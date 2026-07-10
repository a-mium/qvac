'use strict'

// JS integration test for GR00T N1.7-3B.
//
// Loads groot.gguf via the public VlaModel surface and drives one end-to-end
// inference through VlaModel.load()/run()/unload(). This is a PLUMBING /
// finite-shape gate, not a PyTorch-parity test:
//
//   * GR00T's oracle dumps no input_ids and no reference final actions, so —
//     exactly like the C++ test_groot_infer_smoke.cpp — the returned action
//     chunk can't be compared numerically. Numerical correctness of every
//     composed stage is covered by the C++ milestone tests (test_groot_m4_*).
//   * What this DOES catch, and the C++ tests can't:
//       - the JS validator's groot branch (imageInputMode === 'patches' —
//         patchified images bypass the pixel-plane `3·w·h` length check)
//       - binding.runJob argument marshalling for a patch-format image array
//       - VlaModel lifecycle (load → run → unload) + stats/hparams surfacing
//
// Skips cleanly when the artefacts aren't on disk (so CI without the groot
// mirror still passes). Desktop-only; no mobile bundle / S3 download path yet
// (deferred until the GGUF lives on a project-owned mirror):
//   GROOT_TEST_GGUF            — path to groot-q8_vf16.gguf
//   GROOT_TEST_ACTIVATIONS_V4  — path to activations_v4.safetensors

const test = require('brittle')
const fs = require('bare-fs')
const os = require('bare-os')
const path = require('bare-path')
const process = require('bare-process')
const { VlaModel } = require('../..')

// ── Performance reporter (best-effort; same shape as pi05.test.js) ─────────
let createPerformanceReporter
const _scriptBase = path.join('..', '..', '..', '..', 'scripts', 'test-utils')
try {
  const perfReporterMod = require(path.join(_scriptBase, 'performance-reporter'))
  perfReporterMod.configure({ fs, path, process, os })
  createPerformanceReporter = perfReporterMod.createPerformanceReporter
} catch (_) {
  createPerformanceReporter = function () {
    const _results = []
    return {
      record (testName, metrics, extra) { _results.push({ test: testName, metrics, extra }) },
      writeReport () {},
      writeStepSummary () {},
      writeToConsole () {},
      get length () { return _results.length }
    }
  }
}

const _perfReporter = createPerformanceReporter({ addon: 'vla', addonType: 'groot' })
const _reportPath = path.resolve('.', 'test/results/performance-report-groot.json')

process.on('exit', () => {
  if (_perfReporter.length === 0) return
  try {
    _perfReporter.writeReport(_reportPath)
    _perfReporter.writeStepSummary()
    _perfReporter.writeToConsole()
  } catch (err) {
    console.log('[perf-reporter] flush failed: ' + (err && err.message))
  }
})

// ── Fixture dims (match test_groot_infer_smoke.cpp) ────────────────────────
const N_IMAGES = 4 // 2 cameras × 2 frames
const PATCHES_PER_IMG = 256
const IN_FLAT = 1536
const T_TOK = 280
const IMAGE_TOKEN_ID = 151655
const STATE_DIM = 132
const N_ACT = 40
const ACT_DIM = 132
const IMAGE_SIZE = 256

// ── Asset detection ────────────────────────────────────────────────────────
//   HAVE: both env vars set AND files exist → run against local safetensors.
//   SKIP: both unset → local dev convenience.
//   FAIL: set but missing → loud (CI sets them unconditionally; a silent skip
//         would hide a broken mirror path).
const _assetsState = (function detectAssets () {
  const keys = ['GROOT_TEST_GGUF', 'GROOT_TEST_ACTIVATIONS_V4']
  const values = keys.map((k) => process.env[k])
  if (values.every((v) => !v)) return { state: 'SKIP' }
  const missing = keys.filter((k, i) => !values[i] || !fs.existsSync(values[i]))
  if (missing.length > 0) {
    return {
      state: 'FAIL',
      reason: 'Some GROOT_TEST_* env vars point at missing files: ' +
        missing.map((k) => `${k}=${process.env[k] || '<unset>'}`).join(', ')
    }
  }
  return { state: 'HAVE' }
})()

const SKIP_REASON =
  'set GROOT_TEST_GGUF and GROOT_TEST_ACTIVATIONS_V4 to run the groot integration test'

// Mobile is deferred for v1 (same rationale as pi05's _skipMobilePi05): even the
// ~3.76 GB groot-q8_vf16.gguf is around pi05's size, which the iOS jetsam
// per-process cap already killed, and there's no CDN-fronted mirror to stream it
// to Device Farm yet. Aggressive Q4_K is the mobile follow-up. The generated
// mobile bundle globs every *.test.js, so skip explicitly rather than relying
// on the env-vars-unset SKIP path firing by accident.
const _platform = os.platform()
const _skipMobileGroot = _platform === 'ios' || _platform === 'android'

// ── Inline safetensors v1 parser (same as pi05.test.js) ────────────────────
function loadSafetensors (p) {
  const buf = fs.readFileSync(p)
  const headerLen = Number(buf.readBigUInt64LE(0))
  if (headerLen <= 0 || headerLen > buf.length - 8) {
    throw new Error(`safetensors: bad header length in ${p}`)
  }
  const headerJson = buf.subarray(8, 8 + headerLen).toString('utf8')
  const header = JSON.parse(headerJson)
  const blobStart = 8 + headerLen
  return {
    get (name) {
      const rec = header[name]
      if (!rec) throw new Error(`safetensors: missing tensor '${name}' in ${p}`)
      const start = blobStart + rec.data_offsets[0]
      const end = blobStart + rec.data_offsets[1]
      const slice = buf.subarray(start, end)
      switch (rec.dtype) {
        case 'F32':
          return new Float32Array(slice.buffer, slice.byteOffset, slice.byteLength / 4)
        default:
          throw new Error(`safetensors: unsupported dtype ${rec.dtype} for '${name}'`)
      }
    }
  }
}

// Build the { images[4], state, tokens, mask, noise } inputs from the oracle.
// Real patchified images + normalized state + sampled noise; a synthetic prompt
// with an image placeholder at each visual-pos-mask position. This JS gate stays
// plumbing-only (finite/shape), so it uses stand-in text ids; the C++
// infer-parity gtest is the one that feeds v4's real input_ids for numeric parity.
function _loadInputs () {
  const act = loadSafetensors(process.env.GROOT_TEST_ACTIVATIONS_V4)

  const patches = act.get('vision_input.call0.args.0')
  const perImg = PATCHES_PER_IMG * IN_FLAT
  if (patches.length !== N_IMAGES * perImg) {
    throw new Error(`patches length ${patches.length} != ${N_IMAGES}*${perImg}`)
  }
  const images = []
  for (let i = 0; i < N_IMAGES; i++) {
    images.push(patches.subarray(i * perImg, (i + 1) * perImg))
  }

  const state = act.get('state_encoder_input.call0.args.0')
  if (state.length !== STATE_DIM) throw new Error(`state length ${state.length} != ${STATE_DIM}`)
  const noise = act.get('action_encoder_input.call0.args.0')
  if (noise.length !== N_ACT * ACT_DIM) throw new Error(`noise length ${noise.length} != ${N_ACT * ACT_DIM}`)

  const vpm = act.get('text_model_input.call0.kwargs.visual_pos_masks')
  if (vpm.length !== T_TOK) throw new Error(`visual_pos_masks length ${vpm.length} != ${T_TOK}`)
  const tokens = new Int32Array(T_TOK)
  for (let t = 0; t < T_TOK; t++) tokens[t] = vpm[t] > 0.5 ? IMAGE_TOKEN_ID : (1000 + t)
  const mask = new Uint8Array(T_TOK).fill(1)

  return {
    ggufPath: process.env.GROOT_TEST_GGUF,
    images,
    state: Float32Array.from(state),
    tokens,
    mask,
    noise: Float32Array.from(noise)
  }
}

test('groot integration: VlaModel.run() produces finite, correctly-shaped actions', { timeout: 1200000, skip: _skipMobileGroot }, async (t) => {
  if (_assetsState.state === 'SKIP') {
    t.comment('skipping: ' + SKIP_REASON)
    return
  }
  if (_assetsState.state === 'FAIL') {
    t.fail(_assetsState.reason)
    return
  }

  const inputs = _loadInputs()
  const { ggufPath, images, state, tokens, mask, noise } = inputs

  // Input-shape sanity (platform-agnostic).
  t.is(images.length, N_IMAGES, `${N_IMAGES} image buffers`)
  t.is(images[0].length, PATCHES_PER_IMG * IN_FLAT, 'image 0 is a patch buffer')
  t.is(state.length, STATE_DIM, 'state length')
  t.is(tokens.length, T_TOK, 'tokens length')
  t.is(mask.length, T_TOK, 'mask length')
  t.is(noise.length, N_ACT * ACT_DIM, 'noise length')

  const model = new VlaModel({
    files: { model: [path.resolve(ggufPath)] },
    config: { verbosity: 1 }
  })
  try {
    await model.load({ backend: 'cpu' })

    // hparams surface (mirrors the C++ integration test).
    t.ok(model.hparams, 'hparams populated')
    t.is(model.hparams.chunkSize, N_ACT, 'chunk_size')
    t.is(model.hparams.actionDim, ACT_DIM, 'action_dim')
    t.is(model.hparams.maxStateDim, STATE_DIM, 'max_state_dim')
    t.is(model.hparams.visionImageSize, IMAGE_SIZE, 'vision_image_size')
    t.is(model.hparams.numCameras, 2, 'num_cameras')
    t.is(model.hparams.stateInputMode, 'continuous', 'state_input_mode')
    t.is(model.hparams.imageInputMode, 'patches', 'image_input_mode (groot = patches)')
    t.is(model.backendName.toLowerCase(), 'cpu', 'backend name (cpu)')

    const input = {
      images,
      imgWidth: IMAGE_SIZE,
      imgHeight: IMAGE_SIZE,
      state,
      tokens,
      mask,
      noise
    }

    const t0 = Date.now()
    const response = await model.run(input)
    const result = await response.await()
    t.comment(`inference elapsed: ${Date.now() - t0} ms`)

    t.ok(result, 'run() returned a result')
    t.ok(result.actions instanceof Float32Array, 'actions is Float32Array')
    t.is(result.actions.length, N_ACT * ACT_DIM, 'actions length == chunk_size * action_dim')

    let allFinite = true
    for (let i = 0; i < result.actions.length; i++) {
      if (!Number.isFinite(result.actions[i])) { allFinite = false; break }
    }
    t.ok(allFinite, 'all action values are finite')

    const stats = result.stats || {}
    for (const key of ['vision_ms', 'prefill_compute_ms', 'prefill_total_ms', 'ode_ms', 'total_ms']) {
      t.is(typeof stats[key], 'number', `stats.${key} is a number`)
      t.ok(stats[key] >= 0, `stats.${key} >= 0`)
    }
    t.ok(stats.total_ms > 0, 'stats.total_ms > 0')
    console.log(
      `[VLA TIMING groot/cpu] vision=${stats.vision_ms.toFixed(0)}ms ` +
      `prefill_compute=${stats.prefill_compute_ms.toFixed(0)}ms ` +
      `prefill_total=${stats.prefill_total_ms.toFixed(0)}ms ` +
      `ode=${stats.ode_ms.toFixed(0)}ms total=${stats.total_ms.toFixed(0)}ms`
    )

    _perfReporter.record('end-to-end inference (groot/cpu)', {
      total_time_ms: stats.total_ms,
      vision_time_ms: stats.vision_ms,
      prefill_compute_time_ms: stats.prefill_compute_ms,
      prefill_total_time_ms: stats.prefill_total_ms,
      ode_time_ms: stats.ode_ms
    }, { execution_provider: model.backendName || null })
  } finally {
    await model.unload().catch(() => {})
  }
})

// ── Error-path tests (shape symmetry with pi05.test.js) ──────────────────
test('groot integration: module exports expected surface', (t) => {
  t.is(typeof VlaModel, 'function')
})

test('groot integration: VlaModel rejects missing/invalid files.model', (t) => {
  let err1 = null
  try { const m = new VlaModel({ files: { model: [] } }); t.absent(m) } catch (e) { err1 = e }
  t.ok(err1 && /non-empty array/.test(err1.message))

  let err2 = null
  try { const m = new VlaModel({ files: { model: ['relative/path.gguf'] } }); t.absent(m) } catch (e) { err2 = e }
  t.ok(err2 && /absolute path/.test(err2.message))
})
