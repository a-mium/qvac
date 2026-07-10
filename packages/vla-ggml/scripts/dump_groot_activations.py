"""Dump GR00T N1.7-3B PyTorch reference activations for ggml-port parity testing.

Runs NVIDIA's own Isaac-GR00T `Gr00tPolicy` on a fixed synthetic fixture (one
hardcoded embodiment: OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT) and dumps named
intermediate tensors to a safetensors file. The C++ milestone tests in
test/unit/test_groot_m*_*.cpp diff their ggml sub-graph outputs against these.

Must run on a CUDA GPU with Isaac-GR00T installed (flash-attn is a hard
dependency of Qwen3VLForConditionalGeneration's default load path there) —
see packages/vla-ggml/scripts/README-oracle-groot.md.

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
    """Registers forward hooks and records every hooked tensor as a plain float32 CPU tensor.

    Two capture modes:
      * ``attach``       — forward hook on a module called ONCE (backbone, vlln,
                           vl_self_attention). Records the module output.
      * ``attach_input`` — forward PRE hook. Records the module's inputs (args +
                           kwargs). Needed for stages whose *input* is the oracle
                           gate (e.g. state_encoder's normalized-state input,
                           which the C++ side can't re-derive without the
                           per-embodiment normalization stats).

    Both modes are *step-aware*: the flow-matching sampler calls the DiT / action
    encoder / action decoder once per denoising step. A plain forward hook would
    overwrite and keep only the last step. Instead every capture is suffixed with
    a per-name call counter (``<name>.callN``) so all 4 steps survive — this is
    what lets the DiT-block and Euler-loop milestones reproduce a *specific* step
    rather than only the final one. Single-call modules still emit ``<name>.call0``
    plus a convenience alias at the bare ``<name>`` (back-compat with the v1 dump
    that the M4.1 VL-fusion test already consumes).
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
            # e.g. a timestep passed as a Python scalar — keep it as a 1-elem
            # tensor so the per-step schedule is recoverable from the dump.
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
        # with_kwargs=True so keyword-passed tensors (timestep=..., etc.) are
        # captured too — the DiT is invoked with a mix of positional/keyword args
        # and we don't want to hardcode its signature here.
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
    # v2 default so a re-run doesn't clobber the v1 activations.safetensors the
    # M4.1 VL-fusion test already consumes. Point GROOT_TEST_ACTIVATIONS at this
    # file once it's regenerated for the M4.2+ milestones.
    ap.add_argument("--out", default="activations_v2.safetensors")
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

    # ── Augmented captures for the DiT / Euler / encoder milestones ──────────
    # These stages can't be isolated from output-only hooks: the DiT and action
    # coder run once per denoising step, and several stages are gated by their
    # INPUT, not their output. Capturing them unblocks the M4.2–M4.4 milestones.
    #
    # state_encoder INPUT — the per-embodiment-normalized state vector. The C++
    # side receives the raw state and can't reproduce the policy's normalization
    # without the data-config stats, so we dump the encoder's actual input.
    recorder.attach_input("state_encoder_input", action_head.state_encoder)
    # timestep_encoder OUTPUT + INPUT (the raw timestep scalar per step).
    recorder.attach("timestep_encoder_output", action_head.model.timestep_encoder)
    recorder.attach_input("timestep_encoder_input", action_head.model.timestep_encoder)
    # action_encoder INPUT (x_t in ACTION space per step; call0 == initial noise)
    # and OUTPUT (embedded action tokens fed to the DiT).
    recorder.attach_input("action_encoder_input", action_head.action_encoder)
    recorder.attach("action_encoder_output", action_head.action_encoder)
    # DiT stack INPUT per step — hidden_states (embedded 41-token sample),
    # timestep, and encoder_hidden_states (the VL features). Signature isn't
    # hardcoded; args+kwargs are captured generically and inspected post-hoc.
    recorder.attach_input("dit_model_input", action_head.model)
    # action_decoder INPUT per step (final DiT hidden → decoded velocity).
    recorder.attach_input("action_decoder_input", action_head.action_decoder)

    # ── Backbone intermediates (de-risk the M4.5 Qwen3-VL port) ──────────────
    # The final backbone_features is a single coarse gate; these split the
    # backbone into independently-checkable pieces: the vision tower output, the
    # text-decoder INPUT (post image/text merge + deepstack — the trickiest
    # interface), and every vision block / text layer hidden state.
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
