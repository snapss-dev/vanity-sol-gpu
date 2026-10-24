import { createHash, randomBytes } from 'node:crypto'
import { describe, expect, it } from 'vitest'
import { fromFe, loadTwin, P, toFe } from './twin'

const rand = () => BigInt('0x' + randomBytes(32).toString('hex')) % P
const mod = (v: bigint) => ((v % P) + P) % P
const pow = (b: bigint, e: bigint) => {
  let r = 1n
  for (b = mod(b); e > 0n; e >>= 1n, b = b * b % P) if (e & 1n) r = r * b % P
  return r
}
// Values that stress carries and the final reduction.
const EDGE = [0n, 1n, 2n, 18n, 19n, 37n, 38n, P - 1n, P - 2n, P - 19n, P - 38n, (1n << 255n) - 20n,
  (1n << 128n) - 1n, (1n << 224n), (1n << 254n), (1n << 254n) + 12345n, P >> 1n]

describe.each([1, 2, 8])('field twin (mulRows=%i)', (rows) => {
  const t = loadTwin(rows)
  const pairs: Array<[bigint, bigint]> = []
  for (const a of EDGE) for (const b of EDGE) pairs.push([a, b])
  for (let i = 0; i < 400; i++) pairs.push([rand(), rand()])

  it('add/sub/mul/sqr match BigInt and stay fully reduced', () => {
    for (const [a, b] of pairs) {
      const A = toFe(a), B = toFe(b)
      expect(fromFe(t.fe_add(A, B))).toBe(mod(a + b))
      expect(fromFe(t.fe_sub(A, B))).toBe(mod(a - b))
      expect(fromFe(t.fe_mul(A, B))).toBe(mod(a * b))
      expect(fromFe(t.fe_sqr(A))).toBe(mod(a * a))
    }
  })

  it('inv matches Fermat', () => {
    for (const a of [...EDGE.filter(v => v !== 0n), ...Array.from({ length: 20 }, rand)]) {
      expect(fromFe(t.fe_inv(toFe(a)))).toBe(pow(a, P - 2n))
    }
  })
})

describe('sha512 twin', () => {
  const t = loadTwin()
  it('matches node:crypto SHA-512 + ed25519 clamping', () => {
    for (let i = 0; i < 300; i++) {
      const seed = i === 0 ? Buffer.alloc(32) : i === 1 ? Buffer.alloc(32, 0xff) : randomBytes(32)
      const h = createHash('sha512').update(seed).digest()
      h[0] &= 248; h[31] &= 127; h[31] |= 64
      const words = Array.from({ length: 8 }, (_, k) => seed.readUInt32BE(4 * k))
      const want = Array.from({ length: 8 }, (_, k) => h.readUInt32LE(4 * k))
      expect(t.sha512_scalar(words)).toEqual(want)
    }
  })
})
