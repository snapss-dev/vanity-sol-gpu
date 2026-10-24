// Loads the generated JavaScript twin of the GPU kernel's arithmetic.
import { generateField } from '../../src/gpu/fieldGen'
import { generateSha512 } from '../../src/gpu/sha512Gen'

type Fe = number[]
export type Twin = {
  fe_add(a: Fe, b: Fe): Fe
  fe_sub(a: Fe, b: Fe): Fe
  fe_mul(a: Fe, b: Fe): Fe
  fe_sqr(a: Fe): Fe
  fe_inv(a: Fe): Fe
  sha512_scalar(seed: number[]): Fe
}

export function loadTwin(mulRows = 1): Twin {
  const src = [generateField('js', { mulRows }), generateSha512('js'),
    'return { fe_add, fe_sub, fe_mul, fe_sqr, fe_inv, sha512_scalar }'].join('\n\n')
  return new Function(src)() as Twin
}

export const P = (1n << 255n) - 19n

export function toFe(v: bigint): Fe {
  return Array.from({ length: 8 }, (_, i) => Number((v >> BigInt(32 * i)) & 0xffffffffn))
}

export function fromFe(f: Fe): bigint {
  return f.reduce((acc, w, i) => acc | (BigInt(w >>> 0) << BigInt(32 * i)), 0n)
}
