import kernelSource from './kernel.wgsl?raw'
import { generateField } from './fieldGen'
import { generateSha512 } from './sha512Gen'

export type KernelConfig = {
  /** Invocations per workgroup. */
  workgroupSize: number
  /** Keys per invocation, sharing one field inversion. */
  keysPerThread: number
  /** Most invocations a dispatch may use (sizes the scratch buffer). */
  maxThreads: number
  /** Capacity of the per-dispatch results buffer. */
  maxResults: number
  /** fe_mul rows per loop iteration: 8 = fully unrolled (fastest, slowest to build). */
  mulRows: number
}

export const RESULT_WORDS = 16
export const RESULTS_HEADER_WORDS = 4
export const SCRATCH_WORDS_PER_KEY = 32 // X, Y, Z, prefix product

export function resultsBufferBytes(cfg: KernelConfig): number {
  return (RESULTS_HEADER_WORDS + cfg.maxResults * RESULT_WORDS) * 4
}

export function scratchBufferBytes(cfg: KernelConfig): number {
  return cfg.maxThreads * cfg.keysPerThread * SCRATCH_WORDS_PER_KEY * 4
}

export function buildShader(cfg: KernelConfig): string {
  const generated = generateField('wgsl', { mulRows: cfg.mulRows, mulItersVar: 'mul_iters' }) + '\n\n' + generateSha512('wgsl')
  const kernel = kernelSource
    .replace('__WG__', String(cfg.workgroupSize))
    .replace('__K__', String(cfg.keysPerThread))
    .replace('__MAX_THREADS__', String(cfg.maxThreads))
    .replace('__MAX_RESULTS__', String(cfg.maxResults))
  return generated + '\n\n' + kernel
}
