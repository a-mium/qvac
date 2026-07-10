#include "model-interface/groot.hpp"

#include <cmath>
#include <stdexcept>

#include <ggml-backend.h>
#include <gguf.h>

#include "model-interface/gguf_helpers.hpp"
#include "utils/BackendSelection.hpp"

namespace qvac_lib_infer_vla_ggml {

// ── Small shared graph helpers (mirrors pi05.cpp's static defs) ──────────
namespace {

// Promote non-F32 weights to F32 on-graph so they share a dtype with the
// F32 activations they combine with (biases/norm weights are stored F16).
static struct ggml_tensor* grootToF32(
    struct ggml_context* ctx, struct ggml_tensor* x) {
  if (x != nullptr && x->type != GGML_TYPE_F32) {
    return ggml_cast(ctx, x, GGML_TYPE_F32);
  }
  return x;
}

// LayerNorm with weight + bias (diffusers/torch nn.LayerNorm, eps default 1e-5).
static struct ggml_tensor* grootLayerNorm(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* weight,
    struct ggml_tensor* bias, float eps) {
  x = ggml_norm(ctx, x, eps);
  if (weight != nullptr) {
    x = ggml_mul(ctx, x, grootToF32(ctx, weight));
  }
  if (bias != nullptr) {
    x = ggml_add(ctx, x, grootToF32(ctx, bias));
  }
  return x;
}

// Linear: y = x @ W^T (+ b). ggml_mul_mat(W, x) treats W as ne=[in, out]
// (nn.Linear's PyTorch (out, in) row-major → ggml [in, out]) and x as
// (..., in), producing (out, ...).
static struct ggml_tensor* grootLinear(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* weight,
    struct ggml_tensor* bias) {
  struct ggml_tensor* out = ggml_mul_mat(ctx, weight, x);
  if (bias != nullptr) {
    out = ggml_add(ctx, out, grootToF32(ctx, bias));
  }
  return out;
}

// CategorySpecificLinear semantics: y = x @ W + b, with W stored [in, out]
// (its forward is `torch.bmm(x, W)`, NOT nn.Linear's `x @ W^T`). Sliced to one
// embodiment, the GGUF tensor is that [in, out] matrix → ggml ne=[out, in].
// ggml_mul_mat wants the weight as ne=[in, out], so transpose it here.
// (convert_groot_dit_to_gguf.py stores it untransposed and its "no transpose
// needed" comment is incorrect — this is the compensating transpose, verified
// against CategorySpecificLinear.forward in embodiment_conditioned_mlp.py.)
static struct ggml_tensor* grootLinearXW(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* weight,
    struct ggml_tensor* bias) {
  struct ggml_tensor* wt = ggml_cont(ctx, ggml_transpose(ctx, weight));
  struct ggml_tensor* out = ggml_mul_mat(ctx, wt, x);
  if (bias != nullptr) {
    out = ggml_add(ctx, out, grootToF32(ctx, bias));
  }
  return out;
}

} // namespace

// ── Internal model state ────────────────────────────────────────────────
// CPU-only mmap path for v1 (no_alloc=false — gguf_init_from_file mmaps the
// file directly, tensor `data` pointers point straight at the OS page
// cache). Matches pi05's CPU path; GPU alloc+copy path deferred until v1
// parity is proven on CPU (same staged rollout pi05/smolvla followed).

struct GrootModelInternal {
  // hparams (also mirrored into GrootModel::hparams_)
  int text_num_layers = 16;
  int text_hidden_size = 2048;
  int text_num_heads = 16;
  int text_num_kv_heads = 8;
  int text_head_dim = 128;
  int text_ffn_length = 6144;
  int text_vocab_size = 151936;
  float text_rope_freq_base = 5000000.0f;
  float text_rms_norm_eps = 1e-6f;
  int text_rope_sections[4] = {24, 20, 20, 0};

  int vision_depth = 24;
  int vision_hidden_size = 1024;
  int vision_num_heads = 16;
  int vision_patch_size = 16;
  int vision_spatial_merge_size = 2;
  int vision_temporal_patch_size = 2;
  int vision_num_position_embeddings = 2304;
  int vision_out_hidden_size = 2048;
  std::vector<int> vision_deepstack_indexes; // [5, 11, 17]
  float vision_rms_norm_eps = 1e-6f;

  int hidden_size = 1024;             // action_head.hidden_size
  int input_embedding_dim = 1536;     // DiT inner_dim
  int backbone_embedding_dim = 2048;  // vlfusion inner_dim / DiT cross_attention_dim
  int max_state_dim = 132;
  int max_action_dim = 132;
  int action_horizon = 40;
  int num_inference_timesteps = 4;
  int timestep_proj_channels = 256;
  int position_embed_max_len = 1024;

  int dit_num_layers = 32;
  int dit_num_heads = 32;
  int dit_head_dim = 48;
  int dit_ffn_inner = 6144;
  int dit_output_dim = 1024;
  int dit_attend_text_every_n_blocks = 2;

  int vlfusion_num_layers = 4;
  int vlfusion_num_heads = 32;
  int vlfusion_head_dim = 64;
  int vlfusion_ffn_inner = 8192;

  std::string embodiment_tag;
  int embodiment_cat_id = -1;

  // weight pointers — all owned by `ctx_w` below.
  GrootVisionWeights vision{};
  GrootTextWeights text{};
  GrootVlfusionWeights vlfusion{};
  GrootDitWeights dit{};
  GrootEmbodimentWeights embodiment{};

  // backends + memory
  struct gguf_context* gguf = nullptr;
  struct ggml_context* ctx_w = nullptr;
  ggml_backend_t backend = nullptr;
  ggml_backend_t backend_cpu = nullptr;
  bool has_gpu = false;
  std::string backend_name = "none";

  ~GrootModelInternal() {
    if (gguf != nullptr) {
      gguf_free(gguf);
      gguf = nullptr;
    }
    if (ctx_w != nullptr) {
      ggml_free(ctx_w);
      ctx_w = nullptr;
    }
    if (backend != nullptr && backend != backend_cpu) {
      ggml_backend_free(backend);
    }
    if (backend_cpu != nullptr) {
      ggml_backend_free(backend_cpu);
    }
  }
};

namespace {

std::vector<int> ggufGetI32ArrOr(
    struct gguf_context* g, const char* key, std::vector<int> dflt) {
  const int64_t idx = gguf_find_key(g, key);
  if (idx < 0) {
    return dflt;
  }
  if (gguf_get_kv_type(g, idx) != GGUF_TYPE_ARRAY) {
    return dflt;
  }
  if (gguf_get_arr_type(g, idx) != GGUF_TYPE_INT32) {
    return dflt;
  }
  const size_t n = gguf_get_arr_n(g, idx);
  const int32_t* data = static_cast<const int32_t*>(gguf_get_arr_data(g, idx));
  return std::vector<int>(data, data + n);
}

float ggufGetF32Or(struct gguf_context* g, const char* key, float dflt) {
  const int64_t idx = gguf_find_key(g, key);
  if (idx < 0) {
    return dflt;
  }
  if (gguf_get_kv_type(g, idx) != GGUF_TYPE_FLOAT32) {
    return dflt;
  }
  return gguf_get_val_f32(g, idx);
}

} // namespace

static std::unique_ptr<GrootModelInternal> grootLoadModel(
    const std::string& ggufPath, bool forceCpu,
    const std::string& backendsDir) {
  vla_backend_selection::loadBackendsOnce(backendsDir);
  auto m = std::make_unique<GrootModelInternal>();

  ggml_backend_dev_t cpuDev =
      ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (cpuDev == nullptr) {
    throw std::runtime_error("grootLoadModel: no CPU backend available");
  }
  m->backend_cpu = ggml_backend_dev_init(cpuDev, nullptr);
  if (m->backend_cpu == nullptr) {
    throw std::runtime_error("grootLoadModel: failed to init CPU backend");
  }
  m->backend = m->backend_cpu;
  const char* cpuName = ggml_backend_name(m->backend_cpu);
  m->backend_name = cpuName != nullptr ? cpuName : "CPU";
  m->has_gpu = false;

  if (!forceCpu) {
    ggml_backend_dev_t gpu = vla_backend_selection::pickBestGpuDevice();
    if (gpu != nullptr) {
      ggml_backend_t gpuBackend = ggml_backend_dev_init(gpu, nullptr);
      if (gpuBackend != nullptr) {
        m->backend = gpuBackend;
        m->has_gpu = true;
        const char* bname = ggml_backend_name(gpuBackend);
        m->backend_name = bname != nullptr ? bname : "gpu";
      }
    }
  }

  // v1: CPU-only mmap path regardless of GPU selection above — GPU
  // alloc+copy deferred until parity is proven (see class comment).
  // Force CPU backend for actual compute even if a GPU device was found,
  // by not wiring m->backend into the compute path yet. `infer()` uses
  // m->backend_cpu directly for v1.
  //
  // NOTE: this means hasGpu()/backendName() may report a GPU that isn't
  // actually used for compute yet — acceptable for v1 (matches "CPU-only
  // mmap path for v1" scope), revisit alongside the GPU alloc+copy path.

  struct gguf_init_params gp{};
  gp.no_alloc = false;
  gp.ctx = &m->ctx_w;
  m->gguf = gguf_init_from_file(ggufPath.c_str(), gp);
  if (m->gguf == nullptr) {
    throw std::runtime_error(
        "grootLoadModel: gguf_init_from_file failed for " + ggufPath);
  }

  const std::string arch = ggufGetStrOr(m->gguf, "general.architecture", "");
  if (arch != "groot") {
    throw std::runtime_error(
        "grootLoadModel: expected general.architecture=groot, got '" + arch +
        "'");
  }

  m->text_num_layers = ggufGetU32Or(m->gguf, "groot.text.num_layers", 16);
  m->text_hidden_size = ggufGetU32Or(m->gguf, "groot.text.hidden_size", 2048);
  m->text_num_heads = ggufGetU32Or(m->gguf, "groot.text.num_heads", 16);
  m->text_num_kv_heads = ggufGetU32Or(m->gguf, "groot.text.num_kv_heads", 8);
  m->text_head_dim = ggufGetU32Or(m->gguf, "groot.text.head_dim", 128);
  m->text_ffn_length = ggufGetU32Or(m->gguf, "groot.text.ffn_length", 6144);
  m->text_vocab_size = ggufGetU32Or(m->gguf, "groot.text.vocab_size", 151936);
  m->text_rope_freq_base =
      ggufGetF32Or(m->gguf, "groot.text.rope_freq_base", 5000000.0f);
  m->text_rms_norm_eps =
      ggufGetF32Or(m->gguf, "groot.text.rms_norm_eps", 1e-6f);

  m->vision_depth = ggufGetU32Or(m->gguf, "groot.vision.depth", 24);
  m->vision_hidden_size = ggufGetU32Or(m->gguf, "groot.vision.hidden_size", 1024);
  m->vision_num_heads = ggufGetU32Or(m->gguf, "groot.vision.num_heads", 16);
  m->vision_patch_size = ggufGetU32Or(m->gguf, "groot.vision.patch_size", 16);
  m->vision_spatial_merge_size =
      ggufGetU32Or(m->gguf, "groot.vision.spatial_merge_size", 2);
  m->vision_temporal_patch_size =
      ggufGetU32Or(m->gguf, "groot.vision.temporal_patch_size", 2);
  m->vision_num_position_embeddings =
      ggufGetU32Or(m->gguf, "groot.vision.num_position_embeddings", 2304);
  m->vision_out_hidden_size =
      ggufGetU32Or(m->gguf, "groot.vision.out_hidden_size", 2048);
  m->vision_deepstack_indexes =
      ggufGetI32ArrOr(m->gguf, "groot.vision.deepstack_indexes", {5, 11, 17});
  m->vision_rms_norm_eps =
      ggufGetF32Or(m->gguf, "groot.vision.rms_norm_eps", 1e-6f);

  m->hidden_size = ggufGetU32Or(m->gguf, "groot.hidden_size", 1024);
  m->input_embedding_dim = ggufGetU32Or(m->gguf, "groot.input_embedding_dim", 1536);
  m->backbone_embedding_dim =
      ggufGetU32Or(m->gguf, "groot.backbone_embedding_dim", 2048);
  m->max_state_dim = ggufGetU32Or(m->gguf, "groot.max_state_dim", 132);
  m->max_action_dim = ggufGetU32Or(m->gguf, "groot.max_action_dim", 132);
  m->action_horizon = ggufGetU32Or(m->gguf, "groot.action_horizon", 40);
  m->num_inference_timesteps =
      ggufGetU32Or(m->gguf, "groot.num_inference_timesteps", 4);
  m->timestep_proj_channels =
      ggufGetU32Or(m->gguf, "groot.timestep_proj_channels", 256);
  m->position_embed_max_len =
      ggufGetU32Or(m->gguf, "groot.position_embed_max_len", 1024);

  m->dit_num_layers = ggufGetU32Or(m->gguf, "groot.dit.num_layers", 32);
  m->dit_num_heads = ggufGetU32Or(m->gguf, "groot.dit.num_heads", 32);
  m->dit_head_dim = ggufGetU32Or(m->gguf, "groot.dit.head_dim", 48);
  m->dit_ffn_inner = ggufGetU32Or(m->gguf, "groot.dit.ffn_inner", 6144);
  m->dit_output_dim = ggufGetU32Or(m->gguf, "groot.dit.output_dim", 1024);
  m->dit_attend_text_every_n_blocks =
      ggufGetU32Or(m->gguf, "groot.dit.attend_text_every_n_blocks", 2);

  m->vlfusion_num_layers = ggufGetU32Or(m->gguf, "groot.vlfusion.num_layers", 4);
  m->vlfusion_num_heads = ggufGetU32Or(m->gguf, "groot.vlfusion.num_heads", 32);
  m->vlfusion_head_dim = ggufGetU32Or(m->gguf, "groot.vlfusion.head_dim", 64);
  m->vlfusion_ffn_inner = ggufGetU32Or(m->gguf, "groot.vlfusion.ffn_inner", 8192);

  m->embodiment_tag = ggufGetStrOr(m->gguf, "groot.embodiment_tag", "");
  m->embodiment_cat_id =
      static_cast<int>(ggufGetU32Or(m->gguf, "groot.embodiment_cat_id", 0));

  // Sanity-check hparams — reject zeros (division/scaling UB) and
  // unreasonable upper bounds (OOM / integer overflow from crafted GGUFs).
  if (m->text_num_layers == 0 || m->text_num_layers > 512 ||
      m->vision_depth == 0 || m->vision_depth > 512 ||
      m->dit_num_layers == 0 || m->dit_num_layers > 512 ||
      m->vlfusion_num_layers == 0 || m->vlfusion_num_layers > 512 ||
      m->action_horizon == 0 || m->action_horizon > 1024 ||
      m->max_state_dim == 0 || m->max_state_dim > 4096 ||
      m->max_action_dim == 0 || m->max_action_dim > 4096 ||
      m->text_hidden_size == 0 || m->text_num_heads == 0 ||
      m->text_num_kv_heads == 0 || m->text_head_dim == 0 ||
      m->text_vocab_size == 0 || m->text_vocab_size > 1048576 ||
      m->vision_hidden_size == 0 || m->vision_num_heads == 0 ||
      m->vision_patch_size == 0 || m->hidden_size == 0 ||
      m->input_embedding_dim == 0 || m->backbone_embedding_dim == 0 ||
      m->dit_num_heads == 0 || m->dit_head_dim == 0 ||
      m->vlfusion_num_heads == 0 || m->vlfusion_head_dim == 0 ||
      m->num_inference_timesteps == 0 || m->num_inference_timesteps > 1024 ||
      m->vision_deepstack_indexes.empty() ||
      m->text_hidden_size % m->text_num_heads != 0 ||
      m->dit_num_heads * m->dit_head_dim != m->input_embedding_dim ||
      m->vlfusion_num_heads * m->vlfusion_head_dim != m->backbone_embedding_dim) {
    throw std::runtime_error(
        "grootLoadModel: one or more GGUF hparams are out of expected range "
        "or inconsistent");
  }
  if (m->embodiment_tag.empty() || m->embodiment_cat_id < 0) {
    throw std::runtime_error(
        "grootLoadModel: missing or invalid groot.embodiment_tag/"
        "groot.embodiment_cat_id — this GGUF wasn't produced by "
        "convert_groot_dit_to_gguf.py");
  }

  auto mustGet = [&](const std::string& name) -> struct ggml_tensor* {
    struct ggml_tensor* t = ggml_get_tensor(m->ctx_w, name.c_str());
    if (t == nullptr) {
      throw std::runtime_error("grootLoadModel: tensor missing from GGUF: " + name);
    }
    return t;
  };

  // ── Vision tower (fabric tensor naming: v.*) ──────────────────────────
  m->vision.patch_embd_w = mustGet("v.patch_embd.weight");
  m->vision.patch_embd_w1 = mustGet("v.patch_embd.weight.1");
  m->vision.patch_embd_b = mustGet("v.patch_embd.bias");
  m->vision.position_embd = mustGet("v.position_embd.weight");
  m->vision.blocks.resize(m->vision_depth);
  for (int i = 0; i < m->vision_depth; ++i) {
    const std::string b = "v.blk." + std::to_string(i);
    auto& bw = m->vision.blocks[i];
    bw.ln1_w = mustGet(b + ".ln1.weight");
    bw.ln1_b = mustGet(b + ".ln1.bias");
    bw.attn_qkv_w = mustGet(b + ".attn_qkv.weight");
    bw.attn_qkv_b = mustGet(b + ".attn_qkv.bias");
    bw.attn_out_w = mustGet(b + ".attn_out.weight");
    bw.attn_out_b = mustGet(b + ".attn_out.bias");
    bw.ln2_w = mustGet(b + ".ln2.weight");
    bw.ln2_b = mustGet(b + ".ln2.bias");
    bw.ffn_up_w = mustGet(b + ".ffn_up.weight");
    bw.ffn_up_b = mustGet(b + ".ffn_up.bias");
    bw.ffn_down_w = mustGet(b + ".ffn_down.weight");
    bw.ffn_down_b = mustGet(b + ".ffn_down.bias");
  }
  m->vision.deepstack_mergers.resize(m->vision_deepstack_indexes.size());
  for (size_t i = 0; i < m->vision_deepstack_indexes.size(); ++i) {
    const std::string b = "v.deepstack." + std::to_string(m->vision_deepstack_indexes[i]);
    auto& dw = m->vision.deepstack_mergers[i];
    dw.norm_w = mustGet(b + ".norm.weight");
    dw.norm_b = mustGet(b + ".norm.bias");
    dw.fc1_w = mustGet(b + ".fc1.weight");
    dw.fc1_b = mustGet(b + ".fc1.bias");
    dw.fc2_w = mustGet(b + ".fc2.weight");
    dw.fc2_b = mustGet(b + ".fc2.bias");
  }
  m->vision.post_ln_w = mustGet("v.post_ln.weight");
  m->vision.post_ln_b = mustGet("v.post_ln.bias");
  m->vision.mm_0_w = mustGet("mm.0.weight");
  m->vision.mm_0_b = mustGet("mm.0.bias");
  m->vision.mm_2_w = mustGet("mm.2.weight");
  m->vision.mm_2_b = mustGet("mm.2.bias");

  // ── Text decoder (fabric tensor naming: blk.*, token_embd, output_norm) ─
  m->text.token_embd_w = mustGet("token_embd.weight");
  m->text.output_norm_w = mustGet("output_norm.weight");
  m->text.blocks.resize(m->text_num_layers);
  for (int i = 0; i < m->text_num_layers; ++i) {
    const std::string b = "blk." + std::to_string(i);
    auto& bw = m->text.blocks[i];
    bw.attn_norm_w = mustGet(b + ".attn_norm.weight");
    bw.attn_q_w = mustGet(b + ".attn_q.weight");
    bw.attn_k_w = mustGet(b + ".attn_k.weight");
    bw.attn_v_w = mustGet(b + ".attn_v.weight");
    bw.attn_output_w = mustGet(b + ".attn_output.weight");
    bw.attn_q_norm_w = mustGet(b + ".attn_q_norm.weight");
    bw.attn_k_norm_w = mustGet(b + ".attn_k_norm.weight");
    bw.ffn_norm_w = mustGet(b + ".ffn_norm.weight");
    bw.ffn_gate_w = mustGet(b + ".ffn_gate.weight");
    bw.ffn_up_w = mustGet(b + ".ffn_up.weight");
    bw.ffn_down_w = mustGet(b + ".ffn_down.weight");
  }

  // ── VL fusion ─────────────────────────────────────────────────────────
  m->vlfusion.vlln_w = mustGet("vlfusion.vlln.weight");
  m->vlfusion.vlln_b = mustGet("vlfusion.vlln.bias");
  m->vlfusion.blocks.resize(m->vlfusion_num_layers);
  for (int i = 0; i < m->vlfusion_num_layers; ++i) {
    const std::string b = "vlfusion.blk." + std::to_string(i);
    auto& bw = m->vlfusion.blocks[i];
    bw.norm1_w = mustGet(b + ".norm1.weight");
    bw.norm1_b = mustGet(b + ".norm1.bias");
    bw.norm3_w = mustGet(b + ".norm3.weight");
    bw.norm3_b = mustGet(b + ".norm3.bias");
    bw.attn_q_w = mustGet(b + ".attn_q.weight");
    bw.attn_q_b = mustGet(b + ".attn_q.bias");
    bw.attn_k_w = mustGet(b + ".attn_k.weight");
    bw.attn_k_b = mustGet(b + ".attn_k.bias");
    bw.attn_v_w = mustGet(b + ".attn_v.weight");
    bw.attn_v_b = mustGet(b + ".attn_v.bias");
    bw.attn_out_w = mustGet(b + ".attn_out.weight");
    bw.attn_out_b = mustGet(b + ".attn_out.bias");
    bw.ffn_in_w = mustGet(b + ".ffn_in.weight");
    bw.ffn_in_b = mustGet(b + ".ffn_in.bias");
    bw.ffn_out_w = mustGet(b + ".ffn_out.weight");
    bw.ffn_out_b = mustGet(b + ".ffn_out.bias");
  }

  // ── DiT ───────────────────────────────────────────────────────────────
  m->dit.timestep_embedder_l1_w = mustGet("dit.timestep_embedder.linear_1.weight");
  m->dit.timestep_embedder_l1_b = mustGet("dit.timestep_embedder.linear_1.bias");
  m->dit.timestep_embedder_l2_w = mustGet("dit.timestep_embedder.linear_2.weight");
  m->dit.timestep_embedder_l2_b = mustGet("dit.timestep_embedder.linear_2.bias");
  m->dit.blocks.resize(m->dit_num_layers);
  for (int i = 0; i < m->dit_num_layers; ++i) {
    const std::string b = "dit.blk." + std::to_string(i);
    auto& bw = m->dit.blocks[i];
    bw.norm1_linear_w = mustGet(b + ".norm1_linear.weight");
    bw.norm1_linear_b = mustGet(b + ".norm1_linear.bias");
    bw.attn_q_w = mustGet(b + ".attn_q.weight");
    bw.attn_q_b = mustGet(b + ".attn_q.bias");
    bw.attn_k_w = mustGet(b + ".attn_k.weight");
    bw.attn_k_b = mustGet(b + ".attn_k.bias");
    bw.attn_v_w = mustGet(b + ".attn_v.weight");
    bw.attn_v_b = mustGet(b + ".attn_v.bias");
    bw.attn_out_w = mustGet(b + ".attn_out.weight");
    bw.attn_out_b = mustGet(b + ".attn_out.bias");
    bw.ffn_in_w = mustGet(b + ".ffn_in.weight");
    bw.ffn_in_b = mustGet(b + ".ffn_in.bias");
    bw.ffn_out_w = mustGet(b + ".ffn_out.weight");
    bw.ffn_out_b = mustGet(b + ".ffn_out.bias");
  }
  m->dit.proj_out_1_w = mustGet("dit.proj_out_1.weight");
  m->dit.proj_out_1_b = mustGet("dit.proj_out_1.bias");
  m->dit.proj_out_2_w = mustGet("dit.proj_out_2.weight");
  m->dit.proj_out_2_b = mustGet("dit.proj_out_2.bias");
  m->dit.position_embedding_w = mustGet("dit.position_embedding.weight");

  // ── Embodiment-conditioned encode/decode ─────────────────────────────
  auto getLinear = [&](const std::string& prefix) -> GrootLinearWeights {
    return {mustGet(prefix + ".weight"), mustGet(prefix + ".bias")};
  };
  m->embodiment.state_encoder_layer1 = getLinear("embodiment.state_encoder.layer1");
  m->embodiment.state_encoder_layer2 = getLinear("embodiment.state_encoder.layer2");
  m->embodiment.action_encoder_w1 = getLinear("embodiment.action_encoder.w1");
  m->embodiment.action_encoder_w2 = getLinear("embodiment.action_encoder.w2");
  m->embodiment.action_encoder_w3 = getLinear("embodiment.action_encoder.w3");
  m->embodiment.action_decoder_layer1 = getLinear("embodiment.action_decoder.layer1");
  m->embodiment.action_decoder_layer2 = getLinear("embodiment.action_decoder.layer2");

  return m;
}

// ── M4.1: VL fusion ─────────────────────────────────────────────────────
// One diffusers BasicTransformerBlock, self-attention only (no cross-attn,
// no AdaLN): x + attn(norm1(x)), then x + ffn(norm3(x)). Attention is full
// bidirectional MHA (all 280 tokens valid — oracle attention_mask is all 1s).
static struct ggml_tensor* grootBuildVlfusionBlock(
    struct ggml_context* ctx, struct ggml_tensor* x,
    const GrootVlfusionBlockWeights& w, int nTokens, int dim, int nHeads,
    int headDim, float eps) {
  // ── Pre-attention LayerNorm + MHSA + residual ─────────────────────────
  struct ggml_tensor* residual = x;
  struct ggml_tensor* h = grootLayerNorm(ctx, x, w.norm1_w, w.norm1_b, eps);

  struct ggml_tensor* q = grootLinear(ctx, h, w.attn_q_w, w.attn_q_b);
  struct ggml_tensor* k = grootLinear(ctx, h, w.attn_k_w, w.attn_k_b);
  struct ggml_tensor* v = grootLinear(ctx, h, w.attn_v_w, w.attn_v_b);

  // (dim, nTokens) → (head_dim, n_heads, nTokens, 1) → (head_dim, nTokens,
  // n_heads, 1) so flash-attn sees each head as an independent matmul.
  q = ggml_reshape_4d(ctx, q, headDim, nHeads, nTokens, 1);
  k = ggml_reshape_4d(ctx, k, headDim, nHeads, nTokens, 1);
  v = ggml_reshape_4d(ctx, v, headDim, nHeads, nTokens, 1);
  q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
  k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
  v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

  // Full (non-causal) MHA — mask null. F16 K/V, F32 accumulation. Same FA
  // portability rationale as pi05's SigLIP block (desktop-class target).
  struct ggml_tensor* kf16 = ggml_cast(ctx, k, GGML_TYPE_F16);
  struct ggml_tensor* vf16 = ggml_cast(ctx, v, GGML_TYPE_F16);
  struct ggml_tensor* attnOut = ggml_flash_attn_ext(
      ctx, q, kf16, vf16, /*mask=*/nullptr,
      1.0f / std::sqrt(static_cast<float>(headDim)), /*max_bias=*/0.0f,
      /*logit_softcap=*/0.0f);
  ggml_flash_attn_ext_set_prec(attnOut, GGML_PREC_F32);
  attnOut = ggml_reshape_2d(ctx, attnOut, dim, nTokens);

  struct ggml_tensor* proj = grootLinear(ctx, attnOut, w.attn_out_w, w.attn_out_b);
  h = ggml_add(ctx, proj, residual);

  // ── Post-attention LayerNorm + GELU-approx FFN + residual ─────────────
  residual = h;
  h = grootLayerNorm(ctx, h, w.norm3_w, w.norm3_b, eps);
  h = grootLinear(ctx, h, w.ffn_in_w, w.ffn_in_b);
  // diffusers GELU(approximate="tanh"); ggml_gelu is the tanh approximation.
  h = ggml_gelu(ctx, h);
  h = grootLinear(ctx, h, w.ffn_out_w, w.ffn_out_b);
  return ggml_add(ctx, h, residual);
}

GrootVlfusionOutputs grootBuildVlfusionGraph(
    struct ggml_context* ctx, struct ggml_tensor* backboneFeatures,
    const GrootVlfusionWeights& w, int nTokens, int dim, int nHeads,
    int headDim, float layerNormEps) {
  GrootVlfusionOutputs out{nullptr, nullptr};
  if (ctx == nullptr || backboneFeatures == nullptr || w.vlln_w == nullptr ||
      w.vlln_b == nullptr || w.blocks.empty()) {
    return out;
  }
  if (nHeads <= 0 || dim <= 0 || headDim <= 0 || nHeads * headDim != dim) {
    return out;
  }

  struct ggml_tensor* vlln =
      grootLayerNorm(ctx, backboneFeatures, w.vlln_w, w.vlln_b, layerNormEps);
  out.vlln_out = vlln;

  struct ggml_tensor* x = vlln;
  for (const auto& bw : w.blocks) {
    x = grootBuildVlfusionBlock(
        ctx, x, bw, nTokens, dim, nHeads, headDim, layerNormEps);
  }
  out.fusion_out = x;
  return out;
}

// ── M4.2: timestep encoder + embodiment MLPs ────────────────────────────

// diffusers get_timestep_embedding, Timesteps(256, flip_sin_to_cos=True,
// downscale_freq_shift=1, max_period=10000). Layout: [cos block | sin block].
void grootComputeTimestepProj(float t, int channels, float* out) {
  const int half = channels / 2;
  const double logMax = std::log(10000.0);
  for (int i = 0; i < half; ++i) {
    // downscale_freq_shift=1 → denominator (half - 1).
    const double exponent = -logMax * static_cast<double>(i) /
                            static_cast<double>(half - 1);
    const double freq = std::exp(exponent);
    const double angle = static_cast<double>(t) * freq;
    out[i] = static_cast<float>(std::cos(angle));        // flip: cos first
    out[half + i] = static_cast<float>(std::sin(angle)); // then sin
  }
}

// SinusoidalPositionalEncoding (embodiment_conditioned_mlp.py). Layout:
// [sin block | cos block], freq denominator is half_dim (not half_dim-1).
void grootComputeActionTauEnc(float t, int dim, float* out) {
  const int half = dim / 2;
  const double logMax = std::log(10000.0);
  for (int i = 0; i < half; ++i) {
    const double exponent =
        -static_cast<double>(i) * (logMax / static_cast<double>(half));
    const double freq = std::exp(exponent);
    const double angle = static_cast<double>(t) * freq;
    out[i] = static_cast<float>(std::sin(angle));        // sin first
    out[half + i] = static_cast<float>(std::cos(angle)); // then cos
  }
}

struct ggml_tensor* grootBuildTimestepMlpGraph(
    struct ggml_context* ctx, struct ggml_tensor* proj,
    struct ggml_tensor* l1W, struct ggml_tensor* l1B, struct ggml_tensor* l2W,
    struct ggml_tensor* l2B) {
  if (ctx == nullptr || proj == nullptr || l1W == nullptr || l2W == nullptr) {
    return nullptr;
  }
  struct ggml_tensor* h = grootLinear(ctx, proj, l1W, l1B);
  h = ggml_silu(ctx, h);
  return grootLinear(ctx, h, l2W, l2B);
}

struct ggml_tensor* grootBuildCategoryMlpGraph(
    struct ggml_context* ctx, struct ggml_tensor* x,
    const GrootLinearWeights& layer1, const GrootLinearWeights& layer2) {
  if (ctx == nullptr || x == nullptr || layer1.weight == nullptr ||
      layer2.weight == nullptr) {
    return nullptr;
  }
  struct ggml_tensor* h = grootLinearXW(ctx, x, layer1.weight, layer1.bias);
  h = ggml_relu(ctx, h);
  return grootLinearXW(ctx, h, layer2.weight, layer2.bias);
}

struct ggml_tensor* grootBuildActionEncoderGraph(
    struct ggml_context* ctx, struct ggml_tensor* actions,
    struct ggml_tensor* tauEnc, const GrootLinearWeights& w1,
    const GrootLinearWeights& w2, const GrootLinearWeights& w3, int hidden,
    int nTokens) {
  if (ctx == nullptr || actions == nullptr || tauEnc == nullptr ||
      w1.weight == nullptr || w2.weight == nullptr || w3.weight == nullptr) {
    return nullptr;
  }
  (void)nTokens;
  // a = W1(actions) → [hidden, nTokens].
  struct ggml_tensor* a = grootLinearXW(ctx, actions, w1.weight, w1.bias);
  // Broadcast the single tau vector across all action tokens, then concat on
  // the feature axis: torch.cat([a_emb, tau_emb], dim=-1) → ggml dim0.
  struct ggml_tensor* tau2 = ggml_repeat(ctx, tauEnc, a);
  struct ggml_tensor* x = ggml_concat(ctx, a, tau2, /*dim=*/0);
  // swish(W2(x)) then W3.
  x = grootLinearXW(ctx, x, w2.weight, w2.bias);
  x = ggml_silu(ctx, x); // swish(x) = x·sigmoid(x) = SiLU
  x = grootLinearXW(ctx, x, w3.weight, w3.bias);
  (void)hidden;
  return x;
}

// ── M4.3: DiT (AlternateVLDiT) ───────────────────────────────────────────

namespace {

// AdaLayerNorm modulation: nh = layernorm_noaffine(x) * (1 + scale) + shift,
// where [a, b] = linear(silu(temb)).chunk(2). `scaleFirst` picks the chunk
// order: the per-block AdaLayerNorm uses (scale, shift) (dit.py:95) but the
// output head uses (shift, scale) (dit.py:331) — opposite halves.
static struct ggml_tensor* grootAdaModulate(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* temb,
    struct ggml_tensor* linW, struct ggml_tensor* linB, int dim, float eps,
    bool scaleFirst) {
  struct ggml_tensor* proj = grootLinear(ctx, ggml_silu(ctx, temb), linW, linB);
  const size_t half = static_cast<size_t>(dim) * ggml_element_size(proj);
  struct ggml_tensor* scale = ggml_view_1d(ctx, proj, dim, scaleFirst ? 0 : half);
  struct ggml_tensor* shift = ggml_view_1d(ctx, proj, dim, scaleFirst ? half : 0);
  struct ggml_tensor* normed = ggml_norm(ctx, x, eps); // no affine
  // normed * (1 + scale) + shift = normed*scale + normed + shift, with the
  // (dim,) scale/shift broadcast across the T token axis.
  struct ggml_tensor* out = ggml_mul(ctx, normed, scale);
  out = ggml_add(ctx, out, normed);
  out = ggml_add(ctx, out, shift);
  return out;
}

// Scaled-dot-product attention, unfused F32, optional additive key-mask.
// nh ne=[dim, T]; kvSrc ne=[kvDim, S]; keyMask ne=[S, T] (0/−inf) or null.
static struct ggml_tensor* grootDitAttention(
    struct ggml_context* ctx, struct ggml_tensor* nh, struct ggml_tensor* kvSrc,
    struct ggml_tensor* keyMask, const GrootDitBlockWeights& w, int nHeads,
    int headDim, int dim) {
  const int64_t nQuery = nh->ne[1];
  const int64_t nKv = kvSrc->ne[1];
  struct ggml_tensor* q = grootLinear(ctx, nh, w.attn_q_w, w.attn_q_b);
  struct ggml_tensor* k = grootLinear(ctx, kvSrc, w.attn_k_w, w.attn_k_b);
  struct ggml_tensor* v = grootLinear(ctx, kvSrc, w.attn_v_w, w.attn_v_b);

  q = ggml_reshape_3d(ctx, q, headDim, nHeads, nQuery);
  k = ggml_reshape_3d(ctx, k, headDim, nHeads, nKv);
  v = ggml_reshape_3d(ctx, v, headDim, nHeads, nKv);
  q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [headDim, T, nHeads]
  k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)); // [headDim, S, nHeads]
  v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)); // [headDim, S, nHeads]

  struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q); // [S, T, nHeads]
  kq = ggml_soft_max_ext(
      ctx, kq, keyMask, 1.0f / std::sqrt(static_cast<float>(headDim)), 0.0f);
  // v → [S, headDim, nHeads] so mul_mat(v, kq) sums over S → [headDim, T, nHeads].
  v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
  struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq); // [headDim, T, nHeads]
  kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)); // [headDim, nHeads, nQuery]
  kqv = ggml_reshape_2d(ctx, kqv, dim, nQuery);
  return grootLinear(ctx, kqv, w.attn_out_w, w.attn_out_b);
}

} // namespace

struct ggml_tensor* grootBuildDitBlockGraph(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* temb,
    struct ggml_tensor* encoder, struct ggml_tensor* keyMask,
    const GrootDitBlockWeights& w, int nHeads, int headDim, int dim,
    int crossDim, int ffnInner, float eps) {
  if (ctx == nullptr || x == nullptr || temb == nullptr ||
      w.norm1_linear_w == nullptr || w.attn_q_w == nullptr) {
    return nullptr;
  }
  (void)crossDim;
  (void)ffnInner;
  // AdaLayerNorm (scale, shift order) → attention → residual.
  struct ggml_tensor* nh = grootAdaModulate(
      ctx, x, temb, w.norm1_linear_w, w.norm1_linear_b, dim, eps,
      /*scaleFirst=*/true);
  struct ggml_tensor* kvSrc = (encoder != nullptr) ? encoder : nh;
  struct ggml_tensor* attn =
      grootDitAttention(ctx, nh, kvSrc, keyMask, w, nHeads, headDim, dim);
  struct ggml_tensor* h = ggml_add(ctx, attn, x);

  // norm3 (plain LayerNorm, no affine) → GELU-approx FFN → residual.
  struct ggml_tensor* nh3 = ggml_norm(ctx, h, eps);
  struct ggml_tensor* ff = grootLinear(ctx, nh3, w.ffn_in_w, w.ffn_in_b);
  ff = ggml_gelu(ctx, ff);
  ff = grootLinear(ctx, ff, w.ffn_out_w, w.ffn_out_b);
  return ggml_add(ctx, ff, h);
}

struct ggml_tensor* grootBuildDitGraph(
    struct ggml_context* ctx, struct ggml_tensor* hidden,
    struct ggml_tensor* temb, struct ggml_tensor* encoder,
    struct ggml_tensor* imageKeyMask, struct ggml_tensor* textKeyMask,
    const GrootDitWeights& w, int nLayers, int nHeads, int headDim, int dim,
    int crossDim, int ffnInner, int outputDim, int attendTextEveryN, float eps,
    std::vector<struct ggml_tensor*>* outBlocks) {
  if (ctx == nullptr || hidden == nullptr || temb == nullptr ||
      encoder == nullptr || static_cast<int>(w.blocks.size()) < nLayers) {
    return nullptr;
  }
  struct ggml_tensor* h = hidden;
  for (int idx = 0; idx < nLayers; ++idx) {
    const bool selfAttn = (idx % 2 == 1); // AlternateVLDiT: odd blocks self-attend
    struct ggml_tensor* enc = selfAttn ? nullptr : encoder;
    struct ggml_tensor* mask = nullptr;
    if (!selfAttn) {
      // Even cross-attn blocks alternate text/image every attendTextEveryN
      // cross-attn block: idx % (2*n) == 0 → text tokens, else image tokens.
      const bool text = (idx % (2 * attendTextEveryN) == 0);
      mask = text ? textKeyMask : imageKeyMask;
    }
    h = grootBuildDitBlockGraph(
        ctx, h, temb, enc, mask, w.blocks[idx], nHeads, headDim, dim, crossDim,
        ffnInner, eps);
    if (outBlocks != nullptr) {
      outBlocks->push_back(h);
    }
  }

  // Output head: norm_out (no affine, eps 1e-6) modulated by AdaLN from
  // proj_out_1 (shift, scale order — opposite of the block AdaLayerNorm), then
  // proj_out_2 to the action-space output dim.
  struct ggml_tensor* modulated = grootAdaModulate(
      ctx, h, temb, w.proj_out_1_w, w.proj_out_1_b, dim, /*eps=*/1e-6f,
      /*scaleFirst=*/false);
  (void)outputDim;
  return grootLinear(ctx, modulated, w.proj_out_2_w, w.proj_out_2_b);
}

// ── M4.5: Qwen3-VL text decoder (backbone language side) ────────────────

namespace {

// RMSNorm over dim0 (the feature axis) with a scale weight, no bias — Qwen3
// convention. Weight promoted to F32 to combine with F32 activations.
static struct ggml_tensor* grootRmsNorm(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* weight,
    float eps) {
  x = ggml_rms_norm(ctx, x, eps);
  if (weight != nullptr) {
    x = ggml_mul(ctx, x, grootToF32(ctx, weight));
  }
  return x;
}

// One Qwen3-VL decoder layer's self-attention: per-head Q/K RMSNorm → M-RoPE
// (interleaved, GGML_ROPE_TYPE_IMROPE) → GQA scaled-dot-product with a causal
// additive mask. Unfused F32 (280 tokens is tiny; F32 keeps us strictly more
// precise than the bf16 oracle). `nh` ne=[dim, T] is the pre-normed input.
static struct ggml_tensor* grootTextAttention(
    struct ggml_context* ctx, struct ggml_tensor* nh,
    struct ggml_tensor* positions, struct ggml_tensor* mask,
    const GrootTextBlockWeights& w, int nHead, int nHeadKv, int headDim,
    int nTokens, float ropeFreqBase, const int ropeSections[4], float rmsEps) {
  const int embdQ = nHead * headDim;
  const int embdKv = nHeadKv * headDim;

  struct ggml_tensor* q = ggml_mul_mat(ctx, w.attn_q_w, nh); // [embdQ, T]
  struct ggml_tensor* k = ggml_mul_mat(ctx, w.attn_k_w, nh); // [embdKv, T]
  struct ggml_tensor* v = ggml_mul_mat(ctx, w.attn_v_w, nh); // [embdKv, T]
  (void)embdQ;
  (void)embdKv;

  q = ggml_reshape_3d(ctx, q, headDim, nHead, nTokens);
  k = ggml_reshape_3d(ctx, k, headDim, nHeadKv, nTokens);
  v = ggml_reshape_3d(ctx, v, headDim, nHeadKv, nTokens);

  // Per-head RMSNorm on the head_dim axis, before RoPE (Qwen3 q/k norm).
  q = grootRmsNorm(ctx, q, w.attn_q_norm_w, rmsEps);
  k = grootRmsNorm(ctx, k, w.attn_k_norm_w, rmsEps);

  int sections[4] = {
      ropeSections[0], ropeSections[1], ropeSections[2], ropeSections[3]};
  q = ggml_rope_multi(
      ctx, q, positions, nullptr, headDim, sections, GGML_ROPE_TYPE_IMROPE,
      /*n_ctx_orig=*/32768, ropeFreqBase, /*freq_scale=*/1.0f,
      /*ext_factor=*/0.0f, /*attn_factor=*/1.0f, /*beta_fast=*/32.0f,
      /*beta_slow=*/1.0f);
  k = ggml_rope_multi(
      ctx, k, positions, nullptr, headDim, sections, GGML_ROPE_TYPE_IMROPE,
      32768, ropeFreqBase, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

  // [headDim, T, nHead] / [headDim, T, nHeadKv] — K broadcasts over the GQA
  // group (nHead / nHeadKv) inside ggml_mul_mat.
  q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
  k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
  v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

  struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q); // [T_kv, T_q, nHead]
  kq = ggml_soft_max_ext(
      ctx, kq, mask, 1.0f / std::sqrt(static_cast<float>(headDim)), 0.0f);
  // v → [T_kv, headDim, nHeadKv] so mul_mat sums over keys → [headDim, T_q, nHead].
  v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
  struct ggml_tensor* kqv = ggml_mul_mat(ctx, v, kq); // [headDim, T_q, nHead]
  kqv = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
  kqv = ggml_reshape_2d(ctx, kqv, nHead * headDim, nTokens);
  return ggml_mul_mat(ctx, w.attn_output_w, kqv);
}

} // namespace

struct ggml_tensor* grootBuildTextDecoderGraph(
    struct ggml_context* ctx, struct ggml_tensor* inputsEmbeds,
    struct ggml_tensor* positions, struct ggml_tensor* mask,
    const std::vector<struct ggml_tensor*>& deepstack, const GrootTextWeights& w,
    int nLayers, int nTokens, int nHead, int nHeadKv, int headDim, int ffnLen,
    float ropeFreqBase, const int ropeSections[4], float rmsEps) {
  if (ctx == nullptr || inputsEmbeds == nullptr || positions == nullptr ||
      mask == nullptr || static_cast<int>(w.blocks.size()) < nLayers) {
    return nullptr;
  }
  (void)ffnLen;
  struct ggml_tensor* cur = inputsEmbeds;
  for (int il = 0; il < nLayers; ++il) {
    const auto& bw = w.blocks[il];
    struct ggml_tensor* inpSA = cur;

    struct ggml_tensor* nh = grootRmsNorm(ctx, cur, bw.attn_norm_w, rmsEps);
    struct ggml_tensor* attn = grootTextAttention(
        ctx, nh, positions, mask, bw, nHead, nHeadKv, headDim, nTokens,
        ropeFreqBase, ropeSections, rmsEps);
    struct ggml_tensor* ffnInp = ggml_add(ctx, attn, inpSA);

    // SwiGLU FFN: down(silu(gate(x)) * up(x)).
    struct ggml_tensor* fn = grootRmsNorm(ctx, ffnInp, bw.ffn_norm_w, rmsEps);
    struct ggml_tensor* gate = ggml_mul_mat(ctx, bw.ffn_gate_w, fn);
    struct ggml_tensor* up = ggml_mul_mat(ctx, bw.ffn_up_w, fn);
    struct ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, gate), up);
    struct ggml_tensor* down = ggml_mul_mat(ctx, bw.ffn_down_w, act);
    cur = ggml_add(ctx, down, ffnInp);

    // Deepstack injection: add the merged visual features to the residual
    // stream after layers 0/1/2 (indices 0..deepstack.size()-1).
    if (il < static_cast<int>(deepstack.size()) && deepstack[il] != nullptr) {
      cur = ggml_add(ctx, cur, deepstack[il]);
    }
  }
  // NB: no final output_norm — backbone_features is the raw residual stream
  // (GR00T's select_layer=16 truncation output), see header note.
  return cur;
}

// ── M4.5: Qwen3-VL vision tower ─────────────────────────────────────────

struct ggml_tensor* grootBuildPatchEmbedLinear(
    struct ggml_context* ctx, const struct ggml_tensor* conv0,
    const struct ggml_tensor* conv1, int nEmbd, int inChannels,
    int temporalPatch, int patchSize) {
  if (ctx == nullptr || conv0 == nullptr || conv1 == nullptr ||
      temporalPatch != 2) {
    return nullptr; // v1 fixture: 2 temporal halves (conv0 = t0, conv1 = t1).
  }
  const int patch = patchSize;
  const int numCh = inChannels;
  const int inFlat = numCh * temporalPatch * patch * patch;
  const struct ggml_tensor* conv[2] = {conv0, conv1};

  auto readVal = [](const struct ggml_tensor* t, size_t i) -> float {
    if (t->type == GGML_TYPE_F32) {
      return static_cast<const float*>(t->data)[i];
    }
    return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t*>(t->data)[i]);
  };

  struct ggml_tensor* wlin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inFlat, nEmbd);
  auto* wd = static_cast<float*>(wlin->data);
  // conv ne=[pw, ph, C, OC] contiguous: idx = pw + P*(ph + P*(c + C*oc)).
  // linear flat = ((c*T + t)*P + ph)*P + pw.
  for (int oc = 0; oc < nEmbd; ++oc) {
    for (int c = 0; c < numCh; ++c) {
      for (int ph = 0; ph < patch; ++ph) {
        for (int pw = 0; pw < patch; ++pw) {
          const size_t src =
              static_cast<size_t>(pw) + patch * (ph + patch * (c + numCh * oc));
          for (int t = 0; t < temporalPatch; ++t) {
            const int flat = ((c * temporalPatch + t) * patch + ph) * patch + pw;
            wd[static_cast<size_t>(oc) * inFlat + flat] = readVal(conv[t], src);
          }
        }
      }
    }
  }
  return wlin;
}

namespace {

// Qwen3-VL learned position embedding (get_vision_bilinear_indices_and_weights
// + pos_embed, modeling_qwen3_vl.py). Align-corners bilinear interpolation of
// the √numPosEmbd base grid to the actual gridH×gridW, emitted directly in
// 2×2-merge sequence order via HF's `reorder`. Computed host-side from the F16
// table (exact — ggml_interpolate's antialiased bilinear does NOT match).
// Returns an F32 tensor ne=[nEmbd, gridH*gridW] for one image.
static struct ggml_tensor* grootBuildVisionPosEmbed(
    struct ggml_context* ctx, const struct ggml_tensor* table, int gridH,
    int gridW, int mergeSize, int numPosEmbd, int nEmbd) {
  const int side =
      static_cast<int>(std::lround(std::sqrt(static_cast<double>(numPosEmbd))));
  const int nOut = gridH * gridW;

  auto tableVal = [&](int col, int e) -> float {
    const size_t idx = static_cast<size_t>(col) * nEmbd + e;
    if (table->type == GGML_TYPE_F32) {
      return static_cast<const float*>(table->data)[idx];
    }
    return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t*>(table->data)[idx]);
  };

  auto gridCoord = [](int i, int n, int s) -> double {
    // torch.linspace(0, s-1, n)
    return n > 1 ? static_cast<double>(i) * (s - 1) / (n - 1) : 0.0;
  };

  struct ggml_tensor* pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nEmbd, nOut);
  auto* pd = static_cast<float*>(pe->data);

  const int m = mergeSize;
  int seq = 0;
  // HF `reorder`: iterate merge-block rows (a), cols (cc), then in-block (b,d).
  for (int a = 0; a < gridH / m; ++a) {
    for (int cc = 0; cc < gridW / m; ++cc) {
      for (int b = 0; b < m; ++b) {
        for (int d = 0; d < m; ++d) {
          const int oh = a * m + b;
          const int ow = cc * m + d;
          const double hg = gridCoord(oh, gridH, side);
          const double wg = gridCoord(ow, gridW, side);
          const int hf = static_cast<int>(hg);
          const int wf = static_cast<int>(wg);
          const int hc = std::min(hf + 1, side - 1);
          const int wc = std::min(wf + 1, side - 1);
          const double hfrac = hg - hf;
          const double wfrac = wg - wf;
          const int c0 = hf * side + wf, c1 = hf * side + wc;
          const int c2 = hc * side + wf, c3 = hc * side + wc;
          const double w0 = (1 - hfrac) * (1 - wfrac);
          const double w1 = (1 - hfrac) * wfrac;
          const double w2 = hfrac * (1 - wfrac);
          const double w3 = hfrac * wfrac;
          for (int e = 0; e < nEmbd; ++e) {
            pd[static_cast<size_t>(seq) * nEmbd + e] = static_cast<float>(
                w0 * tableVal(c0, e) + w1 * tableVal(c1, e) +
                w2 * tableVal(c2, e) + w3 * tableVal(c3, e));
          }
          ++seq;
        }
      }
    }
  }
  return pe;
}

// One Qwen3-VL vision block: LayerNorm → fused-QKV self-attention with vision
// M-RoPE and an additive (block-diagonal) mask → residual; LayerNorm → GELU FFN
// → residual. Unfused F32 attention (mirrors clip_graph::build_attn's non-flash
// path). `x` ne=[nEmbd, nPos]; returns same.
static struct ggml_tensor* grootBuildVisionBlock(
    struct ggml_context* ctx, struct ggml_tensor* x,
    struct ggml_tensor* positions, struct ggml_tensor* mask,
    const GrootVisionBlockWeights& w, int nPos, int nEmbd, int nHead,
    int headDim, float eps, float ropeFreqBase) {
  struct ggml_tensor* residual = x;
  struct ggml_tensor* cur = grootLayerNorm(ctx, x, w.ln1_w, w.ln1_b, eps);

  cur = grootLinear(ctx, cur, w.attn_qkv_w, w.attn_qkv_b); // [3*nEmbd, nPos]
  const size_t rowSize = ggml_row_size(cur->type, headDim);
  struct ggml_tensor* q = ggml_view_3d(
      ctx, cur, headDim, nHead, nPos, rowSize, cur->nb[1], 0);
  struct ggml_tensor* k = ggml_view_3d(
      ctx, cur, headDim, nHead, nPos, rowSize, cur->nb[1],
      ggml_row_size(cur->type, nEmbd));
  struct ggml_tensor* v = ggml_view_3d(
      ctx, cur, headDim, nHead, nPos, rowSize, cur->nb[1],
      ggml_row_size(cur->type, 2 * nEmbd));

  int sections[4] = {headDim / 4, headDim / 4, headDim / 4, headDim / 4};
  q = ggml_rope_multi(
      ctx, q, positions, nullptr, headDim / 2, sections, GGML_ROPE_TYPE_VISION,
      32768, ropeFreqBase, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
  k = ggml_rope_multi(
      ctx, k, positions, nullptr, headDim / 2, sections, GGML_ROPE_TYPE_VISION,
      32768, ropeFreqBase, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

  q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)); // [headDim, nPos, nHead]
  k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
  struct ggml_tensor* vt =
      ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)); // [nPos, headDim, nHead]

  struct ggml_tensor* kq = ggml_mul_mat(ctx, k, q); // [nPos_kv, nPos_q, nHead]
  kq = ggml_soft_max_ext(
      ctx, kq, mask, 1.0f / std::sqrt(static_cast<float>(headDim)), 0.0f);
  struct ggml_tensor* kqv = ggml_mul_mat(ctx, vt, kq); // [headDim, nPos_q, nHead]
  kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
  kqv = ggml_cont_2d(ctx, kqv, nEmbd, nPos);
  struct ggml_tensor* attn = grootLinear(ctx, kqv, w.attn_out_w, w.attn_out_b);
  cur = ggml_add(ctx, attn, residual);

  residual = cur;
  struct ggml_tensor* h = grootLayerNorm(ctx, cur, w.ln2_w, w.ln2_b, eps);
  h = grootLinear(ctx, h, w.ffn_up_w, w.ffn_up_b);
  h = ggml_gelu(ctx, h);
  h = grootLinear(ctx, h, w.ffn_down_w, w.ffn_down_b);
  return ggml_add(ctx, h, residual);
}

// 2×2 patch merger MLP (mm.0 → GELU → mm.2, no gate). `x` ne=[nEmbd, nPos]
// where consecutive groups of merge² patches form one output token (GR00T's
// processor already lays patches out in merge order). Returns [outHidden, nPos/m²].
static struct ggml_tensor* grootVisionMerge(
    struct ggml_context* ctx, struct ggml_tensor* x, struct ggml_tensor* fc1W,
    struct ggml_tensor* fc1B, struct ggml_tensor* fc2W, struct ggml_tensor* fc2B,
    int nEmbd, int nPos, int mergeFactor, struct ggml_tensor* normW,
    struct ggml_tensor* normB, float eps) {
  struct ggml_tensor* m =
      ggml_reshape_2d(ctx, x, nEmbd * mergeFactor, nPos / mergeFactor);
  if (normW != nullptr) {
    m = grootLayerNorm(ctx, m, normW, normB, eps);
  }
  m = grootLinear(ctx, m, fc1W, fc1B);
  // Qwen3VLVisionPatchMerger uses nn.GELU() — exact (erf) GELU, unlike the
  // vision blocks' gelu_pytorch_tanh MLP.
  m = ggml_gelu_erf(ctx, m);
  return grootLinear(ctx, m, fc2W, fc2B);
}

} // namespace

struct ggml_tensor* grootBuildVisionGraph(
    struct ggml_context* ctx, struct ggml_tensor* patchInput,
    struct ggml_tensor* patchWLin, struct ggml_tensor* patchBias,
    struct ggml_tensor* positionEmbd, struct ggml_tensor* positions,
    struct ggml_tensor* mask, const GrootVisionWeights& w, int nImages,
    int gridH, int gridW, int nEmbd, int nHead, int headDim, int mergeSize,
    int numPosEmbd, int outHidden, float eps, float ropeFreqBase,
    const std::vector<int>& deepstackIndexes,
    std::vector<struct ggml_tensor*>* outDeepstack,
    std::vector<struct ggml_tensor*>* outBlocks) {
  if (ctx == nullptr || patchInput == nullptr || patchWLin == nullptr ||
      positionEmbd == nullptr || positions == nullptr || mask == nullptr ||
      w.blocks.empty()) {
    return nullptr;
  }
  (void)outHidden;
  const int nPatchesPerImg = gridH * gridW;
  const int nPos = nImages * nPatchesPerImg;
  const int mergeFactor = mergeSize * mergeSize;

  // ── Patch embed (Linear) + bias ────────────────────────────────────────
  struct ggml_tensor* inp = ggml_mul_mat(ctx, patchWLin, patchInput); // [nEmbd, nPos]
  inp = ggml_add(ctx, inp, grootToF32(ctx, patchBias));

  // ── Learned position embedding (Qwen3-VL align-corners bilinear, merge
  // order), computed host-side then tiled per image. ──────────────────────
  struct ggml_tensor* pe = grootBuildVisionPosEmbed(
      ctx, positionEmbd, gridH, gridW, mergeSize, numPosEmbd, nEmbd);
  struct ggml_tensor* peTiled = pe;
  for (int i = 1; i < nImages; ++i) {
    peTiled = ggml_concat(ctx, peTiled, pe, /*dim=*/1);
  }
  inp = ggml_add(ctx, inp, peTiled);

  // ── Transformer blocks + deepstack mergers ─────────────────────────────
  struct ggml_tensor* cur = inp;
  for (int il = 0; il < static_cast<int>(w.blocks.size()); ++il) {
    cur = grootBuildVisionBlock(
        ctx, cur, positions, mask, w.blocks[il], nPos, nEmbd, nHead, headDim,
        eps, ropeFreqBase);
    if (outBlocks != nullptr) {
      outBlocks->push_back(cur);
    }
    if (outDeepstack != nullptr) {
      for (size_t d = 0; d < deepstackIndexes.size(); ++d) {
        if (deepstackIndexes[d] == il && d < w.deepstack_mergers.size()) {
          const auto& dm = w.deepstack_mergers[d];
          outDeepstack->push_back(grootVisionMerge(
              ctx, cur, dm.fc1_w, dm.fc1_b, dm.fc2_w, dm.fc2_b, nEmbd, nPos,
              mergeFactor, dm.norm_w, dm.norm_b, eps));
        }
      }
    }
  }

  // ── post-LayerNorm → 2×2 merge projection (mm.0 → GELU → mm.2). ──────────
  cur = grootLayerNorm(ctx, cur, w.post_ln_w, w.post_ln_b, eps);
  return grootVisionMerge(
      ctx, cur, w.mm_0_w, w.mm_0_b, w.mm_2_w, w.mm_2_b, nEmbd, nPos, mergeFactor,
      /*normW=*/nullptr, /*normB=*/nullptr, eps);
}

// Test hook: run a single vision block in isolation (see test_groot_m4_5_vision).
struct ggml_tensor* grootBuildVisionBlockGraph(
    struct ggml_context* ctx, struct ggml_tensor* x,
    struct ggml_tensor* positions, struct ggml_tensor* mask,
    const GrootVisionBlockWeights& w, int nPos, int nEmbd, int nHead,
    int headDim, float eps, float ropeFreqBase) {
  return grootBuildVisionBlock(
      ctx, x, positions, mask, w, nPos, nEmbd, nHead, headDim, eps,
      ropeFreqBase);
}

// ── GrootModel ────────────────────────────────────────────────────────────

GrootModel::GrootModel(
    const std::string& ggufPath, bool forceCpu, const std::string& backendsDir)
    : impl_(grootLoadModel(ggufPath, forceCpu, backendsDir)) {
  hparams_.chunk_size = impl_->action_horizon;
  hparams_.action_dim = impl_->max_action_dim;
  hparams_.max_action_dim = impl_->max_action_dim;
  hparams_.max_state_dim = impl_->max_state_dim;
  hparams_.tokenizer_max_length = 0; // GR00T's tokenization/templating is consumer-side, no fixed length yet
  hparams_.vision_image_size = 256;  // image_target_size from the source config
  hparams_.num_cameras = 2;          // OXE_DROID: exterior_image_1_left + wrist_image_left
  hparams_.state_input_mode = VlaHparamsGeneric::StateInputMode::Continuous;
}

GrootModel::~GrootModel() = default;

std::string GrootModel::backendName() const {
  return impl_->backend_name;
}

bool GrootModel::hasGpu() const {
  return impl_->has_gpu;
}

bool GrootModel::infer(
    const float** images, int nImages, int imgWidth, int imgHeight,
    const float* state, int stateDim, const int32_t* langTokens,
    const bool* langMask, int langLen, const float* noise, float* actionsOut,
    int* nActionsOut, VlaTimingGeneric* timingOut) {
  // TODO(groot-phase2): graph-building not yet implemented. The GGUF loader
  // above is complete and verified (all ~1032 tensors mapped, hparams
  // validated against the real checkpoint). What's left: vision tower graph
  // (ported from qvac-fabric-llm.cpp's clip_graph_qwen3vl, adapted for our
  // fixed 2-camera/2-frame-history fixture rather than mtmd's general
  // multi-chunk orchestration), text decoder graph with M-RoPE 3-axis
  // position ids + deepstack feature injection at layers 0-2 (ported from
  // src/models/qwen3vl.cpp), VL fusion, DiT (32 alternating self-/cross-
  // attention blocks), and the 4-step Euler action-generation loop —
  // each verified against scripts/dump_groot_activations.py's oracle
  // activations before being wired into this function.
  (void)images;
  (void)nImages;
  (void)imgWidth;
  (void)imgHeight;
  (void)state;
  (void)stateDim;
  (void)langTokens;
  (void)langMask;
  (void)langLen;
  (void)noise;
  (void)actionsOut;
  (void)nActionsOut;
  (void)timingOut;
  return false;
}

} // namespace qvac_lib_infer_vla_ggml
