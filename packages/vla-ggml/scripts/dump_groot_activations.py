"""Dump GR00T N1.7-3B PyTorch reference activations for ggml-port parity testing.

Runs Isaac-GR00T's `Gr00tPolicy` on a fixed synthetic fixture (embodiment
OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT) and dumps named intermediate tensors to a
safetensors file that the C++ milestone tests (test/unit/test_groot_m*_*.cpp)
diff their ggml sub-graph outputs against.

Must run on a CUDA GPU with Isaac-GR00T installed (flash-attn is a hard
dependency of Qwen3VLForConditionalGeneration's default load path there).

Usage:
    python dump_groot_activations.py \
        --checkpoint /path/to/GR00T-N1.7-3B \
        --out activations.safetensors
"""

import argparse
import json

import numpy as np
import torch
from safetensors.torch import save_file

EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
IMAGE_SIZE = 256
VIDEO_KEYS = ["exterior_image_1_left", "wrist_image_left"]
VIDEO_HISTORY = 2  # delta_indices [-15, 0]
STATE_KEYS = {"eef_9d": 9, "gripper_position": 1, "joint_position": 7}
LANGUAGE_KEY = "annotation.language.language_instruction"
INSTRUCTION = "pick up the red block and place it in the bin"
SEED = 0


def build_fixture():
    rng = np.random.default_rng(SEED)
    video = {
        k: rng.integers(0, 256, size=(1, VIDEO_HISTORY, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        for k in VIDEO_KEYS
    }
    state = {
        k: rng.uniform(-1.0, 1.0, size=(1, 1, d)).astype(np.float32)
        for k, d in STATE_KEYS.items()
    }
    language = {LANGUAGE_KEY: [[INSTRUCTION]]}
    return {"video": video, "state": state, "language": language}


class ActivationRecorder:
    """Registers forward hooks and records each hooked tensor as a float32 CPU tensor.

    Two capture modes:
      * ``attach``       — forward hook; records a single-call module's output.
      * ``attach_input`` — forward PRE hook; records a module's inputs (args +
                           kwargs). Needed when the *input* is the oracle gate
                           (e.g. state_encoder's normalized-state input, which the
                           C++ side can't re-derive without the normalization stats).

    Both modes are step-aware: the flow-matching sampler calls the DiT / action
    coder once per denoising step, so a plain hook would keep only the last step.
    Each capture is suffixed with a per-name call counter (``<name>.callN``) so all
    steps survive — letting the DiT-block and Euler-loop milestones reproduce a
    specific step. Single-call modules also emit a bare ``<name>`` alias (back-compat
    with the v1 dump the M4.1 VL-fusion test consumes).
    """

    def __init__(self):
        self.activations = {}
        self._handles = []
        self._counts = {}

    def _next_index(self, name):
        i = self._counts.get(name, 0)
        self._counts[name] = i + 1
        return i

    def _hook(self, name):
        def fn(module, inputs, output):
            idx = self._next_index(name)
            self._store(f"{name}.call{idx}", output)
            if idx == 0:
                self._store(name, output)  # bare alias for single-call modules

        return fn

    def _pre_hook(self, name):
        def fn(module, args, kwargs):
            idx = self._next_index(name)
            self._store(f"{name}.call{idx}.args", list(args))
            if kwargs:
                self._store(f"{name}.call{idx}.kwargs", kwargs)

        return fn

    def _store(self, name, value):
        if isinstance(value, torch.Tensor):
            # .clone() so no two dumped keys share storage — safetensors'
            # save_file rejects aliased tensors, and the bare-name alias for
            # call0 would otherwise point at the same buffer as `<name>.call0`.
            self.activations[name] = value.detach().float().cpu().clone()
        elif isinstance(value, bool):
            pass  # bool is an int subclass; skip flags, they aren't parity data
        elif isinstance(value, (int, float)):
            # e.g. a per-step timestep scalar — keep as a 1-elem tensor so the
            # schedule is recoverable from the dump.
            self.activations[name] = torch.tensor([float(value)])
        elif isinstance(value, (tuple, list)):
            for i, v in enumerate(value):
                self._store(f"{name}.{i}", v)
        elif hasattr(value, "items"):
            for k, v in value.items():
                self._store(f"{name}.{k}", v)

    def attach(self, name, module):
        self._handles.append(module.register_forward_hook(self._hook(name)))

    def attach_input(self, name, module):
        # with_kwargs=True so keyword-passed tensors (timestep=..., etc.) are also
        # captured — the DiT mixes positional/keyword args and we don't hardcode
        # its signature here.
        self._handles.append(
            module.register_forward_pre_hook(self._pre_hook(name), with_kwargs=True)
        )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    # v4 default: v1 activations feed the M4.1 VL-fusion test; v2/v3 feed the
    # DiT/Euler/backbone milestones. v4 additionally dumps the tokenized backbone
    # INPUT (input_ids + pixel_values + grids) alongside the final actions so the
    # e2e infer() parity test can drive the C++ port from the exact same tokenized
    # input the oracle consumed and diff its actions.
    ap.add_argument("--out", default="activations_v4.safetensors")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import gr00t.model  # noqa: F401 registers Gr00tN1d7 with AutoModel
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    policy = Gr00tPolicy(
        embodiment_tag=EMBODIMENT_TAG,
        model_path=args.checkpoint,
        device=args.device,
    )

    model = policy.model
    action_head = model.action_head

    recorder = ActivationRecorder()
    recorder.attach("backbone_output", model.backbone)
    recorder.attach("vlln_output", action_head.vlln)
    recorder.attach("vl_self_attention_output", action_head.vl_self_attention)
    recorder.attach("state_encoder_output", action_head.state_encoder)
    for i, block in enumerate(action_head.model.transformer_blocks):
        recorder.attach(f"dit_block_{i}_output", block)
    recorder.attach("action_decoder_output", action_head.action_decoder)

    # ── E2E PyTorch-parity gate: tokenized backbone INPUT ────────────────────
    # Dump the backbone's actual forward input (input_ids, attention_mask,
    # pixel_values, image_grid_thw) so the e2e test can drive infer() from the
    # identical input the oracle saw — bypassing tokenizer / image-preprocessor
    # drift between PyTorch and the C++ port — and diff its integrated action
    # sample against the oracle's final normalized sample (reconstructed
    # x_3 + dt·vel_3) at bf16 tolerance. final_action.* below is
    # post-unnormalization, i.e. consumer-side.
    recorder.attach_input("backbone_input", model.backbone)

    # ── Augmented captures for the DiT / Euler / encoder milestones ──────────
    # These stages can't be isolated by output-only hooks: the DiT and action
    # coder run once per denoising step, and several stages are gated by their
    # INPUT, not their output.
    #
    # state_encoder INPUT — the per-embodiment-normalized state vector; the C++
    # side gets the raw state and can't reproduce the normalization without the
    # data-config stats.
    recorder.attach_input("state_encoder_input", action_head.state_encoder)
    # timestep_encoder OUTPUT + INPUT (the raw timestep scalar per step).
    recorder.attach("timestep_encoder_output", action_head.model.timestep_encoder)
    recorder.attach_input("timestep_encoder_input", action_head.model.timestep_encoder)
    # action_encoder INPUT (x_t in ACTION space per step; call0 == initial noise)
    # and OUTPUT (embedded action tokens fed to the DiT).
    recorder.attach_input("action_encoder_input", action_head.action_encoder)
    recorder.attach("action_encoder_output", action_head.action_encoder)
    # DiT stack INPUT per step (hidden_states, timestep, encoder_hidden_states);
    # args+kwargs captured generically rather than hardcoding the signature.
    recorder.attach_input("dit_model_input", action_head.model)
    # action_decoder INPUT per step (final DiT hidden → decoded velocity).
    recorder.attach_input("action_decoder_input", action_head.action_decoder)

    # ── Backbone intermediates (de-risk the M4.5 Qwen3-VL port) ──────────────
    # Split the single coarse backbone_features gate into independently-checkable
    # pieces: vision tower output, text-decoder INPUT (the trickiest interface,
    # post image/text merge + deepstack), and every vision block / text layer.
    qwen = model.backbone.model.model  # Qwen3VLModel
    recorder.attach("vision_output", qwen.visual)
    recorder.attach_input("vision_input", qwen.visual)
    recorder.attach_input("text_model_input", qwen.language_model)
    recorder.attach("text_model_output", qwen.language_model)
    if hasattr(qwen.visual, "blocks"):
        for i, blk in enumerate(qwen.visual.blocks):
            recorder.attach(f"vision_block_{i}", blk)
    if hasattr(qwen.language_model, "layers"):
        for i, lyr in enumerate(qwen.language_model.layers):
            recorder.attach(f"text_layer_{i}", lyr)

    observation = build_fixture()
    policy.check_observation(observation)
    action, _info = policy.get_action(observation)
    policy.check_action(action)

    recorder.remove()

    tensors = dict(recorder.activations)
    for key, value in action.items():
        tensors[f"final_action.{key}"] = torch.from_numpy(value)

    meta = {
        "embodiment_tag": EMBODIMENT_TAG,
        "seed": str(SEED),
        "instruction": INSTRUCTION,
        "video_keys": json.dumps(VIDEO_KEYS),
        "state_keys": json.dumps(STATE_KEYS),
    }
    save_file(tensors, args.out, metadata=meta)

    print(f"Saved {len(tensors)} tensors to {args.out}")
    for name, t in tensors.items():
        print(f"  {name:40s} {tuple(t.shape)} {t.dtype}")


if __name__ == "__main__":
    main()
