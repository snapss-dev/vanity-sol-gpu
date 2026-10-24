// Runs in the browser (see scripts/gpu-test.mjs): self-test and benchmark the
// kernel on every WebGPU adapter.
import { createGpuVanityPool } from '../../src/gpu/gpuVanity'
import { compilePattern } from '../../src/gpu/precompute'

export async function runGpuCheck(opts: { allowFallbackAdapter: boolean; seconds: number; kernel?: Record<string, number> }) {
  const failures: string[] = []
  const details: Record<string, unknown> = {}
  const t0 = performance.now()
  const pool = await createGpuVanityPool({ allowFallbackAdapter: opts.allowFallbackAdapter, kernel: opts.kernel })
  details.buildMs = Math.round(performance.now() - t0)
  try {
    for (const gpu of pool.instances) {
      const ts = performance.now()
      const err = await gpu.selfTest()
      const selfTestMs = Math.round(performance.now() - ts)
      if (err) { failures.push(`${gpu.label}: ${err}`); continue }
      // A pattern nothing matches, so the benchmark measures pure search speed.
      gpu.start(compilePattern('zzzzzzzzzz', 'zzzzz', true))
      let checked = 0
      await gpu.step() // warm-up
      const tb = performance.now()
      while (performance.now() - tb < opts.seconds * 1000) checked += (await gpu.step()).checked
      const secs = (performance.now() - tb) / 1000
      await gpu.drain()
      details[gpu.label] = { selfTestMs, keysPerSecond: Math.round(checked / secs) }
    }
  } finally {
    pool.destroy()
  }
  return { adapters: pool.labels, details, failures }
}
