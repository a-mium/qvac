// GrootModel::infer end-to-end NUMERICAL parity — the DoD "inference output
// matches the PyTorch reference within tolerance" gate. Drives the public
// GrootModel::infer() with the oracle's REAL tokenized prompt (inputIds +
// attention_mask from the v4 dump) and diffs the action sample vs PyTorch.
//
// infer() returns the flow-matching sample in NORMALIZED action space
// ([132,40]) after all 4 Euler steps; unnormalization is consumer-side
// (Gr00tPolicy), so we compare in normalized space. The oracle never hooks the
// final post-loop x_4, so we reconstruct it: x_4 = x_3 + dt·vel_3 (dt=1/4),
// where x_3 is the last action- encoder input and vel_3 is the last decoded
// velocity (action_decoder_output. call3) with its leading state token dropped
// — pred is [41,132]; row 0 is state, rows 1..40 are the actions (mirrors
// infer()'s ggml_view_2d(pred, …, nb[1])).

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "model-interface/groot.hpp"
#include "utils/safetensors_lite.hpp"

using qvac_lib_infer_vla_ggml::GrootModel;
using qvac_lib_infer_vla_ggml::VlaTimingGeneric;

namespace {

constexpr int N_IMAGES = 4; // 2 cameras × 2 frames
constexpr int PATCHES_PER_IMG = 256;
constexpr int IN_FLAT = 1536;
constexpr int T_TOK = 280;
constexpr int STATE_DIM = 132;
constexpr int N_ACT = 40;
constexpr int ACT_DIM = 132;
constexpr int N_STEPS = 4;

const char* envOrNull(const char* n) {
  const char* v = std::getenv(n);
  return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

float cosineSim(const float* a, const float* b, size_t n) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < n; ++i) {
    dot += double(a[i]) * b[i];
    na += double(a[i]) * a[i];
    nb += double(b[i]) * b[i];
  }
  const double d = std::sqrt(na) * std::sqrt(nb);
  return d > 0 ? float(dot / d) : 0.0f;
}

float relMaxDiff(const float* a, const float* b, size_t n) {
  float m = 0, s = 0;
  for (size_t i = 0; i < n; ++i) {
    m = std::max(m, std::fabs(a[i] - b[i]));
    s = std::max(s, std::fabs(b[i]));
  }
  return s > 0 ? m / s : m;
}

} // namespace

TEST(GrootInferParity, FinalActionSampleMatchesPytorch) {
  const char* ggufPath = envOrNull("GROOT_TEST_GGUF");
  const char* actPath = envOrNull("GROOT_TEST_ACTIVATIONS_V4");
  if (ggufPath == nullptr || actPath == nullptr) {
    GTEST_SKIP() << "Set GROOT_TEST_GGUF and GROOT_TEST_ACTIVATIONS_V4 to run "
                    "the GrootModel::infer numerical-parity test.";
  }

  qvac_vla_safetensors_lite::Reader act;
  ASSERT_NO_THROW(act.open(actPath));

  // Real patchified images [1024,1536] → 4 contiguous camera slices.
  const std::vector<float> patches = act.readF32("vision_input.call0.args.0");
  ASSERT_EQ(
      patches.size(),
      static_cast<size_t>(N_IMAGES) * PATCHES_PER_IMG * IN_FLAT);
  std::vector<const float*> images(N_IMAGES);
  for (int i = 0; i < N_IMAGES; ++i) {
    images[i] =
        patches.data() + static_cast<size_t>(i) * PATCHES_PER_IMG * IN_FLAT;
  }

  // Real normalized state and sampled noise (x_0 entering Euler step 0).
  const std::vector<float> state =
      act.readF32("state_encoder_input.call0.args.0");
  ASSERT_EQ(state.size(), static_cast<size_t>(STATE_DIM));
  const std::vector<float> noise =
      act.readF32("action_encoder_input.call0.args.0");
  ASSERT_EQ(noise.size(), static_cast<size_t>(N_ACT) * ACT_DIM);

  // REAL tokenized prompt — the whole point of the v4 dump. inputIds and
  // attention_mask are stored as float32 (the recorder casts every hooked
  // tensor to float); round back to the integer/bool the API expects.
  const std::vector<float> inputIds =
      act.readF32("backbone_input.call0.args.0.input_ids");
  const std::vector<float> attnMask =
      act.readF32("backbone_input.call0.args.0.attention_mask");
  ASSERT_EQ(inputIds.size(), static_cast<size_t>(T_TOK));
  ASSERT_EQ(attnMask.size(), static_cast<size_t>(T_TOK));
  std::vector<int32_t> tokens(T_TOK);
  for (int t = 0; t < T_TOK; ++t) {
    tokens[t] = static_cast<int32_t>(std::lround(inputIds[t]));
  }
  // bool* arg needs a contiguous buffer (std::vector<bool> has no .data()).
  std::vector<char> langMaskBuf(T_TOK);
  for (int t = 0; t < T_TOK; ++t) {
    langMaskBuf[t] = attnMask[t] > 0.5f ? 1 : 0;
  }
  const bool* langMaskPtr = reinterpret_cast<const bool*>(langMaskBuf.data());

  GrootModel model(ggufPath, /*forceCpu=*/true, /*backendsDir=*/"");

  std::vector<float> actionsOut(static_cast<size_t>(N_ACT) * ACT_DIM, 0.0f);
  int nActionsOut = 0;
  VlaTimingGeneric timing{};

  const bool ok = model.infer(
      images.data(),
      N_IMAGES,
      /*imgWidth=*/256,
      /*imgHeight=*/256,
      state.data(),
      STATE_DIM,
      tokens.data(),
      langMaskPtr,
      T_TOK,
      noise.data(),
      actionsOut.data(),
      &nActionsOut,
      &timing);

  ASSERT_TRUE(ok);
  ASSERT_EQ(nActionsOut, N_ACT);
  for (size_t i = 0; i < actionsOut.size(); ++i) {
    ASSERT_TRUE(std::isfinite(actionsOut[i])) << "non-finite action at " << i;
  }

  // ── Reconstruct the oracle's final normalized sample x_4 = x_3 + dt·vel_3 ──
  // x_3 is the last hooked action-encoder input; vel_3 is the last decoded
  // velocity with the leading state token dropped.
  const std::vector<float> x3 =
      act.readF32("action_encoder_input.call3.args.0");
  ASSERT_EQ(x3.size(), static_cast<size_t>(N_ACT) * ACT_DIM);
  const std::vector<float> dec3 = act.readF32("action_decoder_output.call3");
  ASSERT_EQ(dec3.size(), static_cast<size_t>(N_ACT + 1) * ACT_DIM);
  const float dt = 1.0f / static_cast<float>(N_STEPS);
  std::vector<float> oracleFinal(static_cast<size_t>(N_ACT) * ACT_DIM);
  for (size_t i = 0; i < oracleFinal.size(); ++i) {
    // Skip the leading state token: velocity rows start at offset ACT_DIM.
    oracleFinal[i] = x3[i] + dt * dec3[ACT_DIM + i];
  }

  const float cos =
      cosineSim(actionsOut.data(), oracleFinal.data(), oracleFinal.size());
  const float rel =
      relMaxDiff(actionsOut.data(), oracleFinal.data(), oracleFinal.size());
  std::cerr << "[GrootInferParity] final action sample cos=" << cos
            << " rel=" << rel << "\n";

  // End-to-end through the real public pipeline fed the real tokens: infer()'s
  // vl/state features carry the vision tower's ~1% bf16 drift (see M4.6), so
  // the final-sample tolerance matches M4.6's full-pipeline Euler gate rather
  // than the tighter oracle-fed M4.4. Cosine is the strict structural gate.
  EXPECT_GT(cos, 0.9995f);
  EXPECT_LT(rel, 0.05f);
  EXPECT_GT(timing.total_ms, 0.0);
}
