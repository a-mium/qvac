#pragma once

#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <js.h>
#include <inference-addon-cpp/JsInterface.hpp>
#include <inference-addon-cpp/JsUtils.hpp>
#include <inference-addon-cpp/ModelInterfaces.hpp>
#include <inference-addon-cpp/addon/AddonJs.hpp>
#include <inference-addon-cpp/handlers/JsOutputHandlerImplementations.hpp>
#include <inference-addon-cpp/handlers/OutputHandler.hpp>
#include <inference-addon-cpp/queue/OutputCallbackJs.hpp>
#include <tts-cpp/lavasr/denoiser.h>

#include "js-interface/JSAdapter.hpp"
#include "model-interface/chatterbox/ChatterboxModel.hpp"
#include "model-interface/supertonic/SupertonicModel.hpp"

namespace qvac::ttsggml::addon_js {

namespace js = qvac_lib_inference_addon_cpp::js;

using chatterbox::ChatterboxModel;
using supertonic::SupertonicModel;

struct JsAudioOutputHandler
    : qvac_lib_inference_addon_cpp::out_handl::JsBaseOutputHandler<
          std::vector<int16_t>> {
  explicit JsAudioOutputHandler(int sampleRate)
      : qvac_lib_inference_addon_cpp::out_handl::JsBaseOutputHandler<
            std::vector<int16_t>>(
            [this, sampleRate](
                const std::vector<int16_t>& data) -> js_value_t* {
              auto result = js::Object::create(this->env_);
              std::span<const int16_t> outputSpan(data.data(), data.size());
              auto typedArray =
                  js::TypedArray<int16_t>::create(this->env_, outputSpan);
              result.setProperty(this->env_, "outputArray", typedArray);
              result.setProperty(
                  this->env_, "sampleRate",
                  js::Number::create(this->env_, sampleRate));
              return result;
            }) {}
};

struct StreamingPcmChunk {
  std::vector<int16_t> pcm;
  int chunkIndex = 0;
  bool isLast = false;
};

struct JsStreamingPcmHandler
    : qvac_lib_inference_addon_cpp::out_handl::JsBaseOutputHandler<
          StreamingPcmChunk> {
  explicit JsStreamingPcmHandler(int sampleRate)
      : qvac_lib_inference_addon_cpp::out_handl::JsBaseOutputHandler<
            StreamingPcmChunk>(
            [this, sampleRate](const StreamingPcmChunk& chunk) -> js_value_t* {
              auto result = js::Object::create(this->env_);
              std::span<const int16_t> outputSpan(chunk.pcm.data(), chunk.pcm.size());
              auto typedArray =
                  js::TypedArray<int16_t>::create(this->env_, outputSpan);
              result.setProperty(this->env_, "outputArray", typedArray);
              result.setProperty(
                  this->env_, "sampleRate",
                  js::Number::create(this->env_, sampleRate));
              result.setProperty(
                  this->env_, "chunkIndex",
                  js::Number::create(this->env_, chunk.chunkIndex));
              result.setProperty(
                  this->env_, "isLast",
                  js::Boolean::create(this->env_, chunk.isLast));
              return result;
            }) {}
};

inline js_value_t* createInstance(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;
  using namespace std;

  JsArgsParser args(env, info);
  auto configurationParams = args.getJsObject(1, "configurationParams");

  JSAdapter adapter;
  const EngineType engineType = adapter.readEngineType(configurationParams, env);

  unique_ptr<model::IModel> model;
  int sampleRate = 24000;

  // The output sample rate is baked into the JS output handlers at instance
  // creation. Final-rate precedence:
  //   1. outputSampleRate (engine resamples, or the addon resamples after the
  //      enhancer) — always the final emitted rate when set;
  //   2. 48000 when the LavaSR enhancer is active (it always emits 48 kHz);
  //   3. the engine's native rate.
  constexpr int kLavasrEnhancedSampleRate = 48000;

  if (engineType == EngineType::Supertonic) {
    auto cfg = adapter.buildSupertonicConfig(configurationParams, env);
    const bool enhanced = !cfg.enhancerGgufPath.empty();
    const int outSr = cfg.outputSampleRate.value_or(0);
    auto stm = make_unique<SupertonicModel>(std::move(cfg));
    sampleRate =
        outSr > 0 ? outSr
                  : (enhanced ? kLavasrEnhancedSampleRate : stm->sampleRate());
    model = std::move(stm);
  } else {
    auto cfg = adapter.buildChatterboxConfig(configurationParams, env);
    const bool enhanced = !cfg.enhancerGgufPath.empty();
    const int outSr = cfg.outputSampleRate.value_or(0);
    sampleRate =
        outSr > 0 ? outSr : (enhanced ? kLavasrEnhancedSampleRate : 24000);
    model = make_unique<ChatterboxModel>(std::move(cfg));
  }

  out_handl::OutputHandlers<out_handl::JsOutputHandlerInterface> outHandlers;
  outHandlers.add(make_shared<JsAudioOutputHandler>(sampleRate));
  outHandlers.add(make_shared<JsStreamingPcmHandler>(sampleRate));
  unique_ptr<OutputCallBackInterface> callback = make_unique<OutputCallBackJs>(
      env, args.get(0, "jsHandle"), args.getFunction(2, "outputCallback"),
      std::move(outHandlers));

  auto addon = make_unique<AddonJs>(env, std::move(callback), std::move(model));
  return JsInterface::createInstance(env, std::move(addon));
}
JSCATCH

inline js_value_t* runJob(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;
  using namespace std;

  JsArgsParser args(env, info);
  AddonJs& instance = JsInterface::getInstance(env, args.get(0, "instance"));
  auto [type, jsInput] = JsInterface::getInput(args);

  if (type != "text") {
    throw qvac_errors::StatusError(
        qvac_errors::general_error::InvalidArgument,
        "Unknown input type: " + type);
  }

  if (auto* st = dynamic_cast<SupertonicModel*>(&instance.addonCpp->model.get())) {
    SupertonicModel::AnyInput modelInput;
    modelInput.text = js::String(env, jsInput).as<std::string>(env);
    return instance.runJob(std::any(std::move(modelInput)));
  }

  ChatterboxModel::AnyInput modelInput;
  modelInput.text = js::String(env, jsInput).as<std::string>(env);

  auto outputQueue = instance.addonCpp->outputQueue;
  modelInput.chunkCallback = [outputQueue](
      std::vector<int16_t>&& pcm, int chunkIndex, bool isLast) {
    StreamingPcmChunk chunk{std::move(pcm), chunkIndex, isLast};
    outputQueue->queueResult(std::any(std::move(chunk)));
  };

  return instance.runJob(std::any(std::move(modelInput)));
}
JSCATCH

// Async wrapper around AddonCpp::activate() so the deferred GGUF parse
// (ChatterboxModel / SupertonicModel construct without loading; the
// real load happens in waitForLoadInitialization() via IModelAsyncLoad)
// runs on a JsAsyncTask worker thread instead of stalling the JS event
// loop.  Replaces the default sync JsInterface::activate registration in
// binding.cpp.
inline js_value_t* activate(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;

  JsArgsParser args(env, info);
  AddonJs& instance = JsInterface::getInstance(env, args.get(0, "instance"));

  return js::JsAsyncTask::run(
      env, [addonCpp = instance.addonCpp]() { addonCpp->activate(); });
}
JSCATCH

inline js_value_t* reload(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;
  using namespace std;

  JsArgsParser args(env, info);
  AddonJs& instance = JsInterface::getInstance(env, args.get(0, "instance"));
  auto configurationParams = args.getJsObject(1, "configurationParams");
  JSAdapter adapter;

  if (auto* st = dynamic_cast<SupertonicModel*>(&instance.addonCpp->model.get())) {
    auto newCfg = adapter.buildSupertonicConfig(configurationParams, env);
    return js::JsAsyncTask::run(
        env,
        [addonCpp = instance.addonCpp, newCfg = std::move(newCfg)]() mutable {
          auto* stm =
              dynamic_cast<SupertonicModel*>(&addonCpp->model.get());
          if (stm == nullptr) {
            throw qvac_errors::StatusError(
                qvac_errors::general_error::InternalError,
                "reload: model is not a SupertonicModel");
          }
          stm->setConfig(std::move(newCfg));
          stm->reload();
        });
  }

  auto newCfg = adapter.buildChatterboxConfig(configurationParams, env);
  return js::JsAsyncTask::run(
      env,
      [addonCpp = instance.addonCpp, newCfg = std::move(newCfg)]() mutable {
        auto* chatterbox =
            dynamic_cast<ChatterboxModel*>(&addonCpp->model.get());
        if (chatterbox == nullptr) {
          throw qvac_errors::StatusError(
              qvac_errors::general_error::InternalError,
              "reload: model is not a ChatterboxModel");
        }
        chatterbox->setConfig(std::move(newCfg));
        chatterbox->reload();
      });
}
JSCATCH

// denoiserBench(modelPath): A/B the LavaSR denoiser on GPU (OpenCL, fused ops)
// vs scalar CPU over a fixed synthetic input, returning timing + parity so a
// mobile test can log per-device GPU-vs-CPU numbers.  A ggml-CPU "twin"
// (n_gpu_layers=-1) runs the same graph on the CPU backend: gpu-vs-twin nrmse
// is >0 only when OpenCL actually executed (==0 means a silent CPU fallback).
inline js_value_t* denoiserBench(js_env_t* env, js_callback_info_t* info) try {
  using namespace qvac_lib_inference_addon_cpp;
  using clk  = std::chrono::steady_clock;
  using msT  = std::chrono::duration<double, std::milli>;

  JsArgsParser args(env, info);
  auto modelPath = js::String(env, args.get(0, "modelPath")).as<std::string>(env);

  // Force the fused OpenCL ops on for the GPU denoiser graph.
  ::setenv("LAVASR_DN_GRU_OP", "1", 1);
  ::setenv("LAVASR_DN_DW_OP", "1", 1);
  ::setenv("LAVASR_DN_UPS_OP", "1", 1);
  ::setenv("LAVASR_DN_SHUF_OP", "1", 1);
  ::setenv("LAVASR_DN_AP_OP", "1", 1);

  constexpr int    kSr      = 16000;
  constexpr double kSeconds = 10.0;
  constexpr int    kRuns    = 3;
  constexpr double kPi      = 3.14159265358979323846;

  // Deterministic speech-band synthetic input (denoiser RTF is content-independent).
  const size_t nSamp = (size_t) (kSeconds * kSr);
  std::vector<float> pcm(nSamp);
  for (size_t i = 0; i < nSamp; i++) {
    double t = (double) i / kSr;
    pcm[i] = (float) (0.15 * std::sin(2 * kPi * 180.0 * t) +
                      0.10 * std::sin(2 * kPi * 440.0 * t) +
                      0.05 * std::sin(2 * kPi * 900.0 * t));
  }

  auto dnGpu  = tts_cpp::lavasr::Denoiser::load(modelPath, 999);  // OpenCL GPU
  auto dnCpu  = tts_cpp::lavasr::Denoiser::load(modelPath, 0);    // scalar CPU
  auto dnTwin = tts_cpp::lavasr::Denoiser::load(modelPath, -1);   // ggml-CPU twin

  auto medianMs = [&](const tts_cpp::lavasr::Denoiser& d,
                      std::vector<float>& out) -> double {
    out = d.denoise(pcm, kSr);  // warmup
    std::vector<double> t;
    t.reserve(kRuns);
    for (int r = 0; r < kRuns; r++) {
      auto a = clk::now();
      out    = d.denoise(pcm, kSr);
      auto b = clk::now();
      t.push_back(msT(b - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
  };

  std::vector<float> gpuOut, cpuOut, twinOut;
  const double gpuMs = medianMs(*dnGpu, gpuOut);
  const double cpuMs = medianMs(*dnCpu, cpuOut);
  (void) medianMs(*dnTwin, twinOut);  // only for the OpenCL-ran discriminator

  auto nrmseOf = [](const std::vector<float>& a, const std::vector<float>& b,
                    double& cosSimOut) -> double {
    double       se = 0, sr2 = 0, dot = 0, na = 0, nb = 0;
    const size_t m = std::min(a.size(), b.size());
    for (size_t i = 0; i < m; i++) {
      double d = (double) a[i] - (double) b[i];
      se += d * d;
      sr2 += (double) b[i] * b[i];
      dot += (double) a[i] * b[i];
      na += (double) a[i] * a[i];
      nb += (double) b[i] * b[i];
    }
    cosSimOut = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    return std::sqrt(se / (sr2 + 1e-12));
  };

  double       cosSim = 0.0, twinCos = 0.0;
  const double nrmse      = nrmseOf(gpuOut, cpuOut, cosSim);    // GPU vs scalar CPU
  const double twinNrmse  = nrmseOf(gpuOut, twinOut, twinCos);  // GPU vs ggml-CPU twin
  const bool   openclRan  = twinNrmse > 0.0;

  auto result = js::Object::create(env);
  result.setProperty(env, "gpuMs", js::Number::create(env, gpuMs));
  result.setProperty(env, "cpuMs", js::Number::create(env, cpuMs));
  result.setProperty(env, "cosSim", js::Number::create(env, cosSim));
  result.setProperty(env, "nrmse", js::Number::create(env, nrmse));
  result.setProperty(env, "twinNrmse", js::Number::create(env, twinNrmse));
  result.setProperty(env, "openclRan", js::Boolean::create(env, openclRan));
  result.setProperty(
      env, "n",
      js::Number::create(env, (double) std::min(gpuOut.size(), cpuOut.size())));
  return result;
}
JSCATCH

}
