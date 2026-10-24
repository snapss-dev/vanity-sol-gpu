import { ed25519 } from '@noble/curves/ed25519.js'
import {
  buildShader, resultsBufferBytes, scratchBufferBytes, RESULT_WORDS, RESULTS_HEADER_WORDS, type KernelConfig,
} from './shader'
import { buildBaseTable, compilePattern, pubWordsToBytes, seedWordsToBytes, type CompiledPattern } from './precompute'
import { bytesToBase58, matchesVanityTarget } from '../wallet/solanaAddress'

/** A key whose address matched the pattern on the GPU. Callers must still verify it. */
export type GpuHit = { seed: Uint8Array; pub: Uint8Array }

export type GpuStep = { checked: number; hits: GpuHit[] }

export type GpuVanity = {
  label: string
  /** Begin a search. The engine must be idle (see drain). */
  start(pattern: CompiledPattern): void
  /** Keep the GPU queue full and resolve with the next finished dispatch. */
  step(): Promise<GpuStep>
  /** Wait for in-flight work to finish so the engine is idle again. */
  drain(): Promise<void>
  /** Known-answer test on this device. Resolves to null on success, else a reason. */
  selfTest(): Promise<string | null>
  destroy(): void
}

export type GpuVanityPool = {
  // One independent instance per physical GPU.
  instances: GpuVanity[]
  // Human-readable label per instance, same order as `instances`.
  labels: string[]
  destroy(): void
}

export const DEFAULT_KERNEL_CONFIG: KernelConfig = {
  workgroupSize: 64, keysPerThread: 16, maxThreads: 32768, maxResults: 4096, mulRows: 2,
}
const MAX_IN_FLIGHT = 2
const SLOTS = MAX_IN_FLIGHT + 1
// Keep each dispatch short: the GPU is shared with the compositor (long
// dispatches make the page stutter) and the OS kills devices whose work runs
// for ~2 s. The number of invocations per dispatch adapts toward this target.
const TARGET_DISPATCH_MS = 30
const INITIAL_WINDOW = 256
const SELF_TEST_THREADS = 256

// Build a short, human-readable label from a GPUAdapter's info.
function adapterLabel(adapter: GPUAdapter, index: number): string {
  const info = (adapter as { info?: GPUAdapterInfo }).info
  if (info) {
    const name = info.description || [info.vendor, info.architecture].filter(Boolean).join(' ')
    if (name) return name
  }
  return `GPU ${index + 1}`
}

// Stable identity key so we don't run two devices on the same physical GPU.
function adapterKey(adapter: GPUAdapter, index: number): string {
  const info = (adapter as { info?: GPUAdapterInfo }).info
  if (info) {
    const key = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join('|')
    if (key) return key
  }
  return `adapter-${index}`
}

function isFallbackAdapter(adapter: GPUAdapter): boolean {
  const a = adapter as { info?: { isFallbackAdapter?: boolean }; isFallbackAdapter?: boolean }
  return Boolean(a.info?.isFallbackAdapter ?? a.isFallbackAdapter)
}

// Discover the distinct physical GPUs the browser is willing to expose.
// WebGPU has no "enumerate all adapters" API — the only lever is
// powerPreference, which on multi-GPU machines surfaces different chips
// (except on Windows, where Chrome currently ignores it).
export async function enumerateGpuAdapters(allowFallback = false): Promise<GPUAdapter[]> {
  if (!navigator.gpu) throw new Error('WebGPU not supported')
  const adapters: GPUAdapter[] = []
  const seen = new Set<string>()
  for (const powerPreference of ['high-performance', 'low-power'] as GPUPowerPreference[]) {
    let adapter: GPUAdapter | null = null
    try {
      adapter = await navigator.gpu.requestAdapter({ powerPreference })
    } catch {
      adapter = null
    }
    if (!adapter) continue
    // A software rasterizer would be slower than the CPU workers.
    if (!allowFallback && isFallbackAdapter(adapter)) continue
    const key = adapterKey(adapter, adapters.length)
    if (seen.has(key)) continue
    seen.add(key)
    adapters.push(adapter)
  }
  return adapters
}

type Slot = {
  params: GPUBuffer
  results: GPUBuffer
  readback: GPUBuffer
  bind: GPUBindGroup | null
}

type Dispatch = { slot: Slot; threads: number; done: Promise<void>; submittedAt: number }

let cachedTable: Uint32Array | null = null

export async function createGpuVanity(
  adapter: GPUAdapter,
  label = adapterLabel(adapter, 0),
  overrides: Partial<KernelConfig> = {},
): Promise<GpuVanity> {
  const CONFIG: KernelConfig = { ...DEFAULT_KERNEL_CONFIG, ...overrides }
  const K = CONFIG.keysPerThread
  const WG = CONFIG.workgroupSize
  // Fit the scratch buffer (which bounds invocations per dispatch) to the adapter.
  const maxScratch = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize)
  const perThread = scratchBufferBytes({ ...CONFIG, maxThreads: 1 })
  CONFIG.maxThreads = Math.max(WG, Math.min(CONFIG.maxThreads, Math.floor(maxScratch / perThread / WG) * WG))
  const scratchBytes = scratchBufferBytes(CONFIG)
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBufferBindingSize: scratchBytes, maxBufferSize: scratchBytes },
  })

  let destroyed = false
  // An in-flight mapAsync rejects when the device is lost; callers turn that
  // into a fallback. Log the reason for diagnostics.
  void device.lost.then((info) => {
    if (!destroyed) console.warn(`[vanity] WebGPU device "${label}" lost: ${info.reason ?? 'unknown'} — ${info.message}`)
  })

  const module = device.createShaderModule({ code: buildShader(CONFIG) })
  const compileErrors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error')
  if (compileErrors.length) {
    device.destroy()
    throw new Error('WGSL compile failed: ' + compileErrors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('; '))
  }
  // Async pipeline creation keeps the (large) shader compile off the main thread.
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'search' } })

  const STORAGE = GPUBufferUsage.STORAGE
  cachedTable ??= buildBaseTable()
  const tableBuf = device.createBuffer({ size: cachedTable.byteLength, usage: STORAGE | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(tableBuf, 0, cachedTable as Uint32Array<ArrayBuffer>)
  const scratchBuf = device.createBuffer({ size: scratchBytes, usage: STORAGE })
  let patternBuf: GPUBuffer | null = null

  const resultsBytes = resultsBufferBytes(CONFIG)
  const slots: Slot[] = Array.from({ length: SLOTS }, () => ({
    params: device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    results: device.createBuffer({ size: resultsBytes, usage: STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }),
    readback: device.createBuffer({ size: resultsBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    bind: null,
  }))

  let pattern: CompiledPattern | null = null
  let window = INITIAL_WINDOW
  let nextSlot = 0
  let lastDoneAt = 0
  const inflight: Dispatch[] = []
  const paramData = new Uint32Array(12)

  function loadPattern(p: CompiledPattern): void {
    const bytes = Math.max(p.words.byteLength, 16)
    if (!patternBuf || patternBuf.size < bytes) {
      patternBuf?.destroy()
      patternBuf = device.createBuffer({ size: bytes, usage: STORAGE | GPUBufferUsage.COPY_DST })
      for (const slot of slots) {
        slot.bind = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: slot.params } },
            { binding: 1, resource: { buffer: tableBuf } },
            { binding: 2, resource: { buffer: patternBuf } },
            { binding: 3, resource: { buffer: scratchBuf } },
            { binding: 4, resource: { buffer: slot.results } },
          ],
        })
      }
    }
    device.queue.writeBuffer(patternBuf, 0, p.words as Uint32Array<ArrayBuffer>)
    pattern = p
  }

  /** seed: the dispatch's 8 seed words (fresh randomness unless testing). */
  function submit(threads: number, seed?: Uint32Array): void {
    const p = pattern!
    const slot = slots[nextSlot]
    nextSlot = (nextSlot + 1) % SLOTS
    if (seed) paramData.set(seed, 0)
    else crypto.getRandomValues(paramData.subarray(0, 8))
    paramData[8] = threads
    paramData[9] = 8 / CONFIG.mulRows
    paramData[10] = p.suffixLength
    paramData[11] = p.intervalCount
    device.queue.writeBuffer(slot.params, 0, paramData)
    const encoder = device.createCommandEncoder()
    encoder.clearBuffer(slot.results, 0, RESULTS_HEADER_WORDS * 4)
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, slot.bind!)
    pass.dispatchWorkgroups(Math.ceil(threads / WG))
    pass.end()
    encoder.copyBufferToBuffer(slot.results, 0, slot.readback, 0, resultsBytes)
    device.queue.submit([encoder.finish()])
    const done = slot.readback.mapAsync(GPUMapMode.READ)
    done.catch(() => {}) // observed by step()/drain(); avoid unhandled-rejection noise
    inflight.push({ slot, threads, done, submittedAt: performance.now() })
  }

  // Read a finished dispatch's results and release its readback buffer.
  async function collect(d: Dispatch): Promise<{ hits: GpuHit[]; raw: number }> {
    await d.done
    const words = new Uint32Array(d.slot.readback.getMappedRange()).slice()
    d.slot.readback.unmap()
    const raw = words[0]
    const hits: GpuHit[] = []
    for (let k = 0; k < Math.min(raw, CONFIG.maxResults); k++) {
      const o = RESULTS_HEADER_WORDS + k * RESULT_WORDS
      hits.push({ seed: seedWordsToBytes(words.subarray(o, o + 8)), pub: pubWordsToBytes(words.subarray(o + 8, o + 16)) })
    }
    return { hits, raw }
  }

  function adapt(d: Dispatch, finishedAt: number): void {
    const gpuMs = finishedAt - Math.max(lastDoneAt, d.submittedAt)
    lastDoneAt = finishedAt
    if (gpuMs <= 0) return
    const ideal = (d.threads * TARGET_DISPATCH_MS) / gpuMs
    const next = Math.round(Math.min(window * 2, Math.max(window / 2, ideal)) / WG) * WG
    window = Math.min(CONFIG.maxThreads, Math.max(WG, next))
  }

  async function step(): Promise<GpuStep> {
    if (destroyed) throw new Error('GPU vanity destroyed')
    if (!pattern) throw new Error('GPU search not started')
    while (inflight.length < MAX_IN_FLIGHT) submit(window)
    const d = inflight.shift()!
    const { hits } = await collect(d)
    adapt(d, performance.now())
    return { checked: d.threads * K, hits }
  }

  async function drain(): Promise<void> {
    const pending = inflight.splice(0)
    await Promise.allSettled(pending.map(d => d.done))
    for (const d of pending) if (d.slot.readback.mapState === 'mapped') d.slot.readback.unmap()
  }

  function start(next: CompiledPattern): void {
    if (destroyed) throw new Error('GPU vanity destroyed')
    if (inflight.length) throw new Error('GPU search still running; drain() first')
    loadPattern(next)
    lastDoneAt = 0
  }

  // Known-answer test on the user's actual GPU/driver: with a fixed dispatch
  // seed, a match-everything pattern must report every key, each re-derived
  // with @noble; then prefix-only and suffix-only patterns must report exactly
  // the keys whose addresses match (checked against the CPU string compare).
  async function selfTest(): Promise<string | null> {
    if (inflight.length) throw new Error('GPU search still running; drain() first')
    const seed = new Uint32Array(8)
    crypto.getRandomValues(seed)
    const total = SELF_TEST_THREADS * K
    const run = async (p: CompiledPattern) => {
      loadPattern(p)
      submit(SELF_TEST_THREADS, seed)
      return collect(inflight.shift()!)
    }
    const key = (h: GpuHit) => bytesToBase58(h.pub)
    try {
      const all = await run(matchAll())
      if (all.raw !== total) return `expected ${total} results, got ${all.raw}`
      const addresses = new Set<string>()
      for (const h of all.hits) {
        const want = bytesToBase58(ed25519.getPublicKey(h.seed))
        if (want !== key(h)) return `seed ${bytesToBase58(h.seed)} -> ${want}, GPU said ${key(h)}`
        addresses.add(want)
      }
      if (addresses.size !== total) return 'duplicate keys'
      const sample = [...addresses][0]
      const cases: Array<[string, string, boolean]> = [
        [sample.slice(0, 1), '', false],
        ['', sample.slice(-1), false],
        [sample.slice(0, 2), sample.slice(-1), true],
      ]
      for (const [prefix, suffix, caseSensitive] of cases) {
        const got = new Set((await run(compilePattern(prefix, suffix, caseSensitive))).hits.map(key))
        const want = [...addresses].filter(a => matchesVanityTarget(a, prefix, suffix, caseSensitive))
        if (got.size !== want.length || want.some(a => !got.has(a))) {
          return `pattern ${prefix}…${suffix}: expected ${want.length} matches, GPU found ${got.size}`
        }
      }
      return null
    } catch (err) {
      return err instanceof Error ? err.message : String(err)
    } finally {
      await drain()
      pattern = null
    }
  }

  function destroy() {
    if (destroyed) return
    destroyed = true
    pattern = null
    device.destroy() // also releases every buffer
  }

  return { label, start, step, drain, selfTest, destroy }
}

/** A pattern every key matches: one range covering all of n. */
function matchAll(): CompiledPattern {
  const p = compilePattern('', '', true)
  const words = new Uint32Array(p.words.length + 16)
  words.set(p.words)
  words.fill(0xffffffff, p.words.length + 8)
  return { intervalCount: 1, suffixLength: 0, difficulty: 1, words }
}

// Create one independent GpuVanity instance per distinct physical GPU.
export async function createGpuVanityPool(
  opts: { allowFallbackAdapter?: boolean; kernel?: Partial<KernelConfig> } = {},
): Promise<GpuVanityPool> {
  const adapters = await enumerateGpuAdapters(opts.allowFallbackAdapter)
  if (adapters.length === 0) throw new Error('No WebGPU adapter')
  const instances: GpuVanity[] = []
  const errors: string[] = []
  for (let i = 0; i < adapters.length; i++) {
    try {
      instances.push(await createGpuVanity(adapters[i], adapterLabel(adapters[i], i), opts.kernel))
    } catch (err) {
      // Skip a GPU whose device creation failed; others may still work.
      errors.push(err instanceof Error ? err.message : String(err))
    }
  }
  if (instances.length === 0) throw new Error('Failed to create any GPU device: ' + errors.join('; '))
  return {
    instances,
    labels: instances.map(i => i.label),
    destroy() {
      for (const inst of instances) inst.destroy()
    },
  }
}
