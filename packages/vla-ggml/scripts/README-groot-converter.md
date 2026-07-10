# GR00T N1.7-3B weight converter — pipeline + quantisation scheme

This document is the **source of truth** for how a GR00T N1.7-3B checkpoint
becomes the single `groot.gguf` (and its quantised `groot-q8_vf16.gguf`) that
`vla-ggml` loads, and for which tensors get quantised to what.

Unlike π₀.₅ (one converter script), GR00T is a **3-stage** pipeline. The split
is forced by the backbone reusing qvac-fabric's external `convert_hf_to_gguf.py`
— do **not** try to collapse it into one script.

## Pipeline

```
GR00T-N1.7-3B/                      (original HF checkpoint)
  │
  │ 1. _repackage_groot_backbone.py         strip backbone.model. prefix,
  │                                         num_hidden_layers -> 16
  ▼
repackaged-qwen3vl-backbone/        (standard Qwen3-VL HF dir)
  │
  │ 2a. fabric convert_hf_to_gguf.py --outtype f16   (text decoder)
  │ 2b. fabric convert_hf_to_gguf.py --mmproj        (vision tower)
  │ 2c. convert_groot_dit_to_gguf.py                 (action head)
  ▼
groot-backbone-text.gguf  +  groot-backbone-vision.gguf  +  groot-action-head.gguf
  │
  │ 3a. _merge_groot_gguf.py                byte-copy 3 parts -> 1 file
  ▼
groot.gguf                          (unified, F16 backbone + F32 head, ~8.9 GB)
  │
  │ 3b. quantize_groot_gguf.py --profile q8_vf16
  ▼
groot-q8_vf16.gguf                  (~3.76 GB — the shipped CI/runtime artefact)
```

### Stage 1 — repackage the backbone

`_repackage_groot_backbone.py` strips the `backbone.model.` attribute prefix
(GR00T wraps a full `Qwen3VLForConditionalGeneration` at that path) so the tensor
names match a standard Qwen3-VL checkpoint, and rewrites
`text_config.num_hidden_layers` to **16** (GR00T's `select_layer` truncation —
only layers 0-15 exist in the checkpoint). It pulls the vision/text hparams +
tokenizer from a real `nvidia/Cosmos-Reason2-2B` config dir (GR00T's own
`config.json` omits them).

```
python _repackage_groot_backbone.py \
    --groot-checkpoint   ~/Documents/groot-checkpoints/GR00T-N1.7-3B \
    --cosmos-config-dir  ~/Documents/groot-checkpoints/Cosmos-Reason2-2B-config \
    --out                ~/Documents/groot-checkpoints/repackaged-qwen3vl-backbone \
    --num-layers 16
```

### Stage 2 — convert to GGUF parts

**Backbone (2a/2b)** reuses qvac-fabric-llm.cpp's own `convert_hf_to_gguf.py`
(`Qwen3VLVisionModel` / `Qwen3VLTextModel`) against the repackaged dir — its
patch-embed Conv3D→2×Conv2D split and deepstack mergers (layers 5/11/17) are
already tested there, so we write no new tensor map:
- `groot-backbone-text.gguf` — 179 tensors, `qwen3vl` arch, 16 layers.
- `groot-backbone-vision.gguf` — 316 tensors, mmproj (vision tower + `mm.*`).

**Action head (2c)** is genuinely new work — `convert_groot_dit_to_gguf.py`
converts `action_head.*` (VL fusion, DiT, timestep + embodiment MLPs). The
`CategorySpecificLinear` embodiment weights are **sliced to one embodiment at
conversion time** and stored dense, so the ggml loader needs no runtime
embodiment-ID input.

```
python convert_groot_dit_to_gguf.py \
    --checkpoint      ~/Documents/groot-checkpoints/GR00T-N1.7-3B \
    --embodiment-tag  oxe_droid_relative_eef_relative_joint \
    --out             ~/Documents/groot-checkpoints/groot-action-head.gguf
```

Output carries `general.architecture="groot"` + all `groot.*` metadata (hidden
sizes, layer counts, `image_token_id`, `embodiment_cat_id`, …).

### Stage 3 — merge + quantise

`_merge_groot_gguf.py` byte-copies the 3 parts into one `groot.gguf`. The three
namespaces are disjoint (`blk.*`/`token_embd` text, `v.*`/`mm.*` vision,
`dit.*`/`vlfusion.*`/`embodiment.*` head), so names are copied **as-is, not
prefixed** — `groot.cpp` looks tensors up by literal name ported verbatim from
fabric's graph code. `general.architecture` + `groot.*` come from the action-head
part; the parts' own `general.`/`tokenizer.`/`clip.`/`vision.` keys are dropped
(the loader doesn't use llama.cpp's model-loading path).

```
python _merge_groot_gguf.py \
    --text        ~/Documents/groot-checkpoints/groot-backbone-text.gguf \
    --vision      ~/Documents/groot-checkpoints/groot-backbone-vision.gguf \
    --action-head ~/Documents/groot-checkpoints/groot-action-head.gguf \
    --out         ~/Documents/groot-checkpoints/groot.gguf

python quantize_groot_gguf.py \
    --in      ~/Documents/groot-checkpoints/groot.gguf \
    --profile q8_vf16
```

## Quantisation scheme

`quantize_groot_gguf.py` is a standalone, re-runnable pass (merge stays a pure
byte-copy) so profiles can be swept and gated against the M4.x parity tests.

### Must stay unquantised (guardrail — quantising these produces silent garbage)

| Pattern | Kept | Reason |
|---|---|---|
| `v.patch_embd.*`, `v.position_emb*` | F16 | Read via **raw host memory** in `grootBuildPatchEmbedLinear` / `grootBuildVisionPosEmbed` (only understands F16/F32; a Q8_0 block read as F16 = garbage). |
| `embodiment.*` | F16 | Consumed via `grootLinearXW` = `ggml_cont(ggml_transpose(W))`; ggml can't make a *transposed* quantised tensor contiguous (blocks span the reduction axis). Single-embodiment, tiny anyway. |
| any 1-D tensor (norms, biases) | as-is | Per-element math; negligible size; Q8_0 needs the blocked axis to be a multiple of 32. |

Everything else flows through `grootLinear` = `ggml_mul_mat(W, x)`, which
dequantises `W` natively — safe to quantise.

### Profiles

| Name | text / DiT / vlfusion / head | vision tower (`v.*`, `mm.*`) | Size | Use when |
|---|---|---|---|---|
| `q8_0` | Q8_0 | Q8_0 | 3.38 GB | rejected — see below |
| **`q8_vf16`** ⬅ shipped | Q8_0 | **F16** | **3.76 GB** | **default** — all parity gates hold |

**Why vision stays F16.** The vision tower is the one quant-sensitive subgraph:
24 LayerNorm blocks accumulate error with no massive-activation outlier to anchor
cosine (unlike the text decoder). Plain `q8_0`-everything pushes the merged-vision
parity gate from cos 0.999 / rel 4.6 % (F16 floor) down to **0.9958 / 9.6 %**,
which cascades to the VL-fusion gate (0.973). The final *actions* stay cos 0.99999
either way (the DiT washes vision drift out), but the intermediate M4.5/M4.6 gates
fail — so we keep vision at F16 (~+0.4 GB) and Q8_0 everything else. `q8_vf16`
passes all 6 M4.x parity gates; final infer-parity **cos 0.999992 / rel 0.0059**.

## Mobile / K-quant follow-up (out of scope for v1)

`q8_vf16` at 3.76 GB is around π₀.₅'s size, which the iOS jetsam per-process cap
already killed — so mobile is deferred. It needs aggressive **Q4_K / Q5_K** on the
DiT + text FFN (per-block scales should handle within-tensor outliers better than
legacy Q8_0) plus a CDN-fronted mirror to stream the model to Device Farm. Add
such a profile as a new branch in `target_type()` and gate it through the same
M4.x parity tests before shipping.
