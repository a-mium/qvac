'use strict'

// LavaSR denoiser GPU(OpenCL)-vs-CPU A/B bench.
//
// Android-only: the fused denoiser ops (GRU / conv_2d_dw / zero_upsample /
// channel_shuffle / affine_prelu) ship OpenCL kernels only, so the GPU path is
// meaningful on Adreno (OpenCL); on Metal/Vulkan the single-backend denoiser
// graph would reject a fused op.  Loads the bundled 494 KiB denoiser GGUF and
// runs denoise() on the GPU and on the scalar CPU over a fixed synthetic input,
// then logs a PERF_REPORT the device farm scrapes so we can read per-device
// GPU-vs-CPU numbers.  A ggml-CPU "twin" isolates the backend: twin_nrmse > 0
// only when OpenCL actually executed (0 = silent CPU fallback).

const fs = require('bare-fs')
const os = require('bare-os')
const path = require('bare-path')
const proc = require('bare-process')
const test = require('brittle')

const TTSGgml = require('@qvac/tts-ggml')

const platform = os.platform()
const isAndroid = platform === 'android'
const MODEL_FILE = 'lavasr-denoiser-f16.gguf'

// Resolve the bundled denoiser GGUF: mobile via global.assetPaths, else an env
// override, else the committed test asset (desktop).
function resolveDenoiserModel () {
  if (global.assetPaths) {
    const key = '../../testAssets/' + MODEL_FILE
    if (global.assetPaths[key]) return global.assetPaths[key].replace('file://', '')
  }
  if (proc.env && proc.env.TTS_GGML_DENOISER_MODEL) return proc.env.TTS_GGML_DENOISER_MODEL
  const local = path.join(__dirname, '..', 'mobile', 'testAssets', MODEL_FILE)
  if (fs.existsSync(local)) return local
  return null
}

test('LavaSR denoiser GPU(OpenCL)-vs-CPU bench', { timeout: 600000, skip: !isAndroid }, async (t) => {
  const modelPath = resolveDenoiserModel()
  if (!modelPath || !fs.existsSync(modelPath)) {
    t.fail(`denoiser model not found (${MODEL_FILE}); expected bundled test asset`)
    return
  }
  console.log('[denoiser-bench] model: ' + modelPath)

  let r
  try {
    r = TTSGgml.denoiserBench(modelPath)
  } catch (e) {
    t.fail('denoiserBench threw: ' + (e && e.message ? e.message : e))
    return
  }

  const payload = {
    schema_version: 1,
    addon: 'tts-ggml',
    test: 'lavasr-denoiser-bench',
    platform,
    results: [{
      test: 'lavasr-denoiser',
      execution_provider: r.openclRan ? 'opencl' : 'cpu-fallback',
      model: MODEL_FILE,
      metrics: {
        gpu_ms: r.gpuMs,
        cpu_ms: r.cpuMs,
        speedup_vs_cpu: r.cpuMs / r.gpuMs,
        cos_sim: r.cosSim,
        nrmse: r.nrmse,
        twin_nrmse: r.twinNrmse,
        opencl_ran: r.openclRan,
        samples: r.n
      }
    }]
  }
  console.log('[PERF_REPORT_START]' + JSON.stringify(payload) + '[PERF_REPORT_END]')
  console.log(
    `[denoiser-bench] gpu=${r.gpuMs.toFixed(1)}ms cpu=${r.cpuMs.toFixed(1)}ms ` +
    `speedup=${(r.cpuMs / r.gpuMs).toFixed(3)}x cos_sim=${r.cosSim.toFixed(6)} ` +
    `nrmse=${r.nrmse.toExponential(2)} opencl_ran=${r.openclRan}`)

  t.ok(r.gpuMs > 0 && r.cpuMs > 0, 'denoiser produced finite GPU + CPU timings')
  t.ok(r.cosSim > 0.999, `GPU vs CPU parity (cos_sim=${r.cosSim.toFixed(6)} > 0.999)`)
  t.ok(r.openclRan, 'OpenCL backend actually executed (twin_nrmse > 0)')
})
