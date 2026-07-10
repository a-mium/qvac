// M4.1 parity: GR00T VL fusion (vlln + 4-layer SelfAttentionTransformer),
// fed the oracle's dumped backbone_features. Env vars: GROOT_TEST_GGUF,
// GROOT_TEST_ACTIVATIONS.
//
// Tolerance: oracle is bf16 (~0.4% relative noise) and these activations are
// large-magnitude (abs-max ~44/~64), so a fixed absolute bar would measure the
// reference's own rounding. Gate on cosine (> 0.9995, the structural check)
// plus relative max-abs-diff (< 1% of abs-max). Our ggml path runs in F32.

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <ggml.h>
#include <gguf.h>
#include <gtest/gtest.h>

#include "model-interface/groot.hpp"
#include "pi05_compute.hpp"
#include "utils/safetensors_lite.hpp"

namespace {

constexpr int N_TOKENS = 280;
constexpr int DIM = 2048;      // backbone_embedding_dim / vlfusion inner dim
constexpr int N_HEADS = 32;
constexpr int HEAD_DIM = 64;
constexpr int N_BLOCKS = 4;

const char* envOrNull(const char* name) {
  const char* v = std::getenv(name);
  return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

float cosineSim(const float* a, const float* b, size_t n) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (size_t i = 0; i < n; ++i) {
    dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
    nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
  }
  const double denom = std::sqrt(na) * std::sqrt(nb);
  return denom > 0.0 ? static_cast<float>(dot / denom) : 0.0f;
}

float maxAbsDiff(const float* a, const float* b, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float d = std::fabs(a[i] - b[i]);
    if (d > m) {
      m = d;
    }
  }
  return m;
}

float absMax(const float* a, size_t n) {
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    const float v = std::fabs(a[i]);
    if (v > m) {
      m = v;
    }
  }
  return m;
}

// Load one GGUF weight tensor by name (or nullptr).
struct ggml_tensor* get(struct ggml_context* ctx, const char* name) {
  return ggml_get_tensor(ctx, name);
}

} // namespace

TEST(GrootM4_1, VlFusionMatchesPytorch) {
  const char* gguf_path = envOrNull("GROOT_TEST_GGUF");
  const char* activations_path = envOrNull("GROOT_TEST_ACTIVATIONS");
  if (gguf_path == nullptr || activations_path == nullptr) {
    GTEST_SKIP() << "Set GROOT_TEST_GGUF and GROOT_TEST_ACTIVATIONS to run the "
                    "M4.1 VL-fusion parity test.";
  }

  // ── 1. Oracle: backbone features (input) + the two fusion outputs ─────
  qvac_vla_safetensors_lite::Reader activations;
  ASSERT_NO_THROW(activations.open(activations_path));

  const std::vector<float> backbone_features =
      activations.readF32("backbone_output.backbone_features");
  const std::vector<float> expected_vlln =
      activations.readF32("vlln_output");
  const std::vector<float> expected_fusion =
      activations.readF32("vl_self_attention_output");
  ASSERT_EQ(backbone_features.size(), static_cast<size_t>(N_TOKENS * DIM));
  ASSERT_EQ(expected_vlln.size(), static_cast<size_t>(N_TOKENS * DIM));
  ASSERT_EQ(expected_fusion.size(), static_cast<size_t>(N_TOKENS * DIM));

  // ── 2. Load the vlfusion weights from groot.gguf ──────────────────────
  struct ggml_context* ctx_w = nullptr;
  struct gguf_init_params gguf_params{};
  gguf_params.no_alloc = false;
  gguf_params.ctx = &ctx_w;
  struct gguf_context* gguf = gguf_init_from_file(gguf_path, gguf_params);
  ASSERT_NE(gguf, nullptr) << "gguf_init_from_file failed for " << gguf_path;
  ASSERT_NE(ctx_w, nullptr);

  qvac_lib_infer_vla_ggml::GrootVlfusionWeights w{};
  w.vlln_w = get(ctx_w, "vlfusion.vlln.weight");
  w.vlln_b = get(ctx_w, "vlfusion.vlln.bias");
  ASSERT_NE(w.vlln_w, nullptr) << "vlfusion.vlln.weight missing";
  ASSERT_NE(w.vlln_b, nullptr) << "vlfusion.vlln.bias missing";
  w.blocks.resize(N_BLOCKS);
  for (int i = 0; i < N_BLOCKS; ++i) {
    const std::string b = "vlfusion.blk." + std::to_string(i);
    auto& bw = w.blocks[i];
    bw.norm1_w = get(ctx_w, (b + ".norm1.weight").c_str());
    bw.norm1_b = get(ctx_w, (b + ".norm1.bias").c_str());
    bw.norm3_w = get(ctx_w, (b + ".norm3.weight").c_str());
    bw.norm3_b = get(ctx_w, (b + ".norm3.bias").c_str());
    bw.attn_q_w = get(ctx_w, (b + ".attn_q.weight").c_str());
    bw.attn_q_b = get(ctx_w, (b + ".attn_q.bias").c_str());
    bw.attn_k_w = get(ctx_w, (b + ".attn_k.weight").c_str());
    bw.attn_k_b = get(ctx_w, (b + ".attn_k.bias").c_str());
    bw.attn_v_w = get(ctx_w, (b + ".attn_v.weight").c_str());
    bw.attn_v_b = get(ctx_w, (b + ".attn_v.bias").c_str());
    bw.attn_out_w = get(ctx_w, (b + ".attn_out.weight").c_str());
    bw.attn_out_b = get(ctx_w, (b + ".attn_out.bias").c_str());
    bw.ffn_in_w = get(ctx_w, (b + ".ffn_in.weight").c_str());
    bw.ffn_in_b = get(ctx_w, (b + ".ffn_in.bias").c_str());
    bw.ffn_out_w = get(ctx_w, (b + ".ffn_out.weight").c_str());
    bw.ffn_out_b = get(ctx_w, (b + ".ffn_out.bias").c_str());
    ASSERT_NE(bw.ffn_out_b, nullptr) << b << " tensors missing";
  }

  // ── 3. Build the graph on a scratch context ───────────────────────────
  const size_t graph_ctx_mem = 1024u * 1024u * 1024u;
  std::vector<uint8_t> graph_mem(graph_ctx_mem);
  struct ggml_init_params gp{graph_ctx_mem, graph_mem.data(), false};
  struct ggml_context* ctx_g = ggml_init(gp);
  ASSERT_NE(ctx_g, nullptr);

  // backbone_features: numpy (280, 2048) row-major == ggml ne=[2048, 280].
  struct ggml_tensor* feats =
      ggml_new_tensor_2d(ctx_g, GGML_TYPE_F32, DIM, N_TOKENS);
  std::memcpy(feats->data, backbone_features.data(),
              backbone_features.size() * sizeof(float));

  using qvac_lib_infer_vla_ggml::grootBuildVlfusionGraph;
  auto outs = grootBuildVlfusionGraph(
      ctx_g, feats, w, N_TOKENS, DIM, N_HEADS, HEAD_DIM, /*eps=*/1e-5f);
  ASSERT_NE(outs.vlln_out, nullptr);
  ASSERT_NE(outs.fusion_out, nullptr);

  struct ggml_cgraph* graph = ggml_new_graph(ctx_g);
  ggml_build_forward_expand(graph, outs.vlln_out);
  ggml_build_forward_expand(graph, outs.fusion_out);
  ASSERT_EQ(pi05_test::computeGraphCpu(graph), GGML_STATUS_SUCCESS);

  // ── 4. Compare ────────────────────────────────────────────────────────
  const float* got_vlln = static_cast<const float*>(outs.vlln_out->data);
  const float* got_fusion = static_cast<const float*>(outs.fusion_out->data);

  const float cos_vlln =
      cosineSim(got_vlln, expected_vlln.data(), expected_vlln.size());
  const float diff_vlln =
      maxAbsDiff(got_vlln, expected_vlln.data(), expected_vlln.size());
  const float cos_fusion =
      cosineSim(got_fusion, expected_fusion.data(), expected_fusion.size());
  const float diff_fusion =
      maxAbsDiff(got_fusion, expected_fusion.data(), expected_fusion.size());

  const float scale_vlln = absMax(expected_vlln.data(), expected_vlln.size());
  const float scale_fusion =
      absMax(expected_fusion.data(), expected_fusion.size());
  const float rel_vlln = diff_vlln / scale_vlln;
  const float rel_fusion = diff_fusion / scale_fusion;

  std::cerr << "[M4.1] vlln:   cos=" << cos_vlln << " max_abs_diff=" << diff_vlln
            << " rel=" << rel_vlln << "\n"
            << "[M4.1] fusion: cos=" << cos_fusion
            << " max_abs_diff=" << diff_fusion << " rel=" << rel_fusion << "\n";

  EXPECT_GT(cos_vlln, 0.9995f);
  EXPECT_LT(rel_vlln, 0.01f);
  EXPECT_GT(cos_fusion, 0.9995f);
  EXPECT_LT(rel_fusion, 0.01f);

  ggml_free(ctx_g);
  gguf_free(gguf);
  ggml_free(ctx_w);
}
