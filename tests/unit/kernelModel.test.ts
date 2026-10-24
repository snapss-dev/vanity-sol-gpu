// Mirrors kernel.wgsl step by step on the generated JavaScript twin, so the
// formulas, table layout, encodings and matcher are checked without a GPU.
import { randomBytes } from 'node:crypto'
import { ed25519 } from '@noble/curves/ed25519.js'
import { describe, expect, it } from 'vitest'
import { loadTwin } from './twin'
import {
  buildBaseTable, compilePattern, INTERVAL_WORDS, PATTERN_HEADER_WORDS, pubWordsToBytes, seedWordsToBytes,
  TABLE_ENTRY_WORDS, type CompiledPattern,
} from '../../src/gpu/precompute'
import { bytesToBase58, matchesVanityTarget } from '../../src/wallet/solanaAddress'

const t = loadTwin()
const table = buildBaseTable()
const ZERO = [0, 0, 0, 0, 0, 0, 0, 0]
const ONE = [1, 0, 0, 0, 0, 0, 0, 0]
const fe = (o: number) => Array.from(table.subarray(o, o + 8))
const bswap = (x: number) => (((x & 0xff) << 24) | ((x & 0xff00) << 8) | ((x >>> 8) & 0xff00) | (x >>> 24)) >>> 0

/** search(): K keys, one shared inversion. Returns public-key limbs per key. */
function kernelKeys(seeds: number[][]): number[][] {
  const pts: Array<{ X: number[]; Y: number[]; Z: number[]; pref: number[] }> = []
  let acc = ONE
  for (const seed of seeds) {
    let s = t.sha512_scalar(seed)
    let X = ZERO, Y = ONE, Z = ONE, T = ZERO
    for (let w = 0; w < 32; w++) {
      const o = (w * 256 + (s[0] & 0xff)) * TABLE_ENTRY_WORDS
      s = s.map((v, i) => ((v >>> 8) | (i < 7 ? s[i + 1] << 24 : 0)) >>> 0)
      const pa = t.fe_mul(t.fe_sub(Y, X), fe(o + 8))
      const pb = t.fe_mul(t.fe_add(Y, X), fe(o))
      const pc = t.fe_mul(T, fe(o + 16))
      const pd = t.fe_add(Z, Z)
      const e = t.fe_sub(pb, pa), f = t.fe_sub(pd, pc), g = t.fe_add(pd, pc), h = t.fe_add(pb, pa)
      X = t.fe_mul(e, f); Y = t.fe_mul(g, h); T = t.fe_mul(e, h); Z = t.fe_mul(f, g)
    }
    acc = t.fe_mul(acc, Z)
    pts.push({ X, Y, Z, pref: acc })
  }
  let inv = t.fe_inv(acc)
  const out: number[][] = new Array(seeds.length)
  for (let k = seeds.length - 1; k >= 0; k--) {
    let zi = inv
    if (k > 0) { zi = t.fe_mul(inv, pts[k - 1].pref); inv = t.fe_mul(inv, pts[k].Z) }
    const pk = t.fe_mul(pts[k].Y, zi)
    const x = t.fe_mul(pts[k].X, zi)
    pk[7] = (pk[7] | ((x[0] & 1) << 31)) >>> 0
    out[k] = pk
  }
  return out
}

/** matches() from kernel.wgsl. */
function kernelMatches(pk: number[], p: CompiledPattern): boolean {
  const n = pk.map(bswap)
  const w = p.words
  const le = (o: number) => {
    for (let i = 0; i < 8; i++) if (w[o + i] !== n[i]) return w[o + i] < n[i]
    return true
  }
  if (p.intervalCount > 0) {
    let lo = 0, hi = p.intervalCount
    while (lo < hi) {
      const mid = (lo + hi) >> 1
      if (le(PATTERN_HEADER_WORDS + mid * INTERVAL_WORDS + 8)) lo = mid + 1; else hi = mid
    }
    if (lo === p.intervalCount || !le(PATTERN_HEADER_WORDS + lo * INTERVAL_WORDS)) return false
  }
  for (let j = 0; j < p.suffixLength; j++) {
    let rem = 0
    for (let i = 0; i < 8; i++) {
      const a = (rem * 65536) + (n[i] >>> 16)
      const qa = Math.floor(a / 58)
      const b = ((a - qa * 58) * 65536) + (n[i] & 0xffff)
      const qb = Math.floor(b / 58)
      rem = b - qb * 58
      n[i] = ((qa << 16) | qb) >>> 0
    }
    if (rem !== (w[j] & 0xff) && rem !== (w[j] >>> 8)) return false
  }
  return true
}

const randomSeed = () => Array.from({ length: 8 }, () => randomBytes(4).readUInt32BE())

describe('kernel model', () => {
  it('derives the same public keys as @noble (batched inversion included)', () => {
    for (let batch = 0; batch < 6; batch++) {
      const seeds = Array.from({ length: 8 }, randomSeed)
      if (batch === 0) seeds[0] = ZERO
      const pks = kernelKeys(seeds)
      seeds.forEach((seed, k) => {
        const want = ed25519.getPublicKey(seedWordsToBytes(seed))
        expect(bytesToBase58(pubWordsToBytes(pks[k]))).toBe(bytesToBase58(want))
      })
    }
  })

  it('matcher agrees with the string comparison', () => {
    // Synthetic key bytes, including leading zero bytes (addresses starting with "1").
    const keys: Uint8Array[] = []
    for (let i = 0; i < 3000; i++) {
      const b = randomBytes(32)
      if (i % 5 === 0) b[0] = 0
      if (i % 25 === 0) b[1] = 0
      if (i % 7 === 0) b[0] = 0xff
      keys.push(b)
    }
    const addrs = keys.map(k => bytesToBase58(k))
    const patterns: Array<[string, string, boolean]> = [
      ['', 'z', true], ['', 'Ab', false], ['1', '', true], ['11', '', true], ['1A', '', false],
      ['2', '', true], ['z', '', true], ['Z', '', false], ['ab', 'c', false], ['9', 'L', false],
    ]
    // Plus patterns taken from real addresses, so there are hits to find.
    for (let i = 0; i < 40; i++) {
      const a = addrs[i * 31]
      patterns.push([a.slice(0, 1 + (i % 3)), i % 2 ? a.slice(-1 - (i % 2)) : '', i % 4 < 2])
    }
    for (const [prefix, suffix, cs] of patterns) {
      const p = compilePattern(prefix, suffix, cs)
      keys.forEach((k, i) => {
        const limbs = Array.from({ length: 8 }, (_, j) => new DataView(k.buffer, k.byteOffset).getUint32(4 * j, true))
        const want = matchesVanityTarget(addrs[i], prefix, suffix, cs)
        if (kernelMatches(limbs, p) !== want) throw new Error(`${prefix}…${suffix} cs=${cs}: ${addrs[i]} expected ${want}`)
      })
    }
  })

  it('rejects prefixes no key can produce', () => {
    expect(() => compilePattern('1'.repeat(32), '', true)).toThrow()
    expect(() => compilePattern('z'.repeat(44), '', true)).toThrow()
  })
})

describe('pattern difficulty', () => {
  it('matches the observed match rate on random keys', () => {
    const N = 60000
    const addrs = Array.from({ length: N }, () => bytesToBase58(randomBytes(32)))
    const cases: Array<[string, string, boolean]> = [
      ['A', '', true],   // also reachable by 44-character addresses (~1 in 17)
      ['z', '', true],   // only 43-character addresses start with "z" (~1 in 1000)
      ['', 'x', true],
      ['h', 'a', false],
      ['1', '', true],   // a leading zero byte
    ]
    for (const [prefix, suffix, cs] of cases) {
      const expected = N / compilePattern(prefix, suffix, cs).difficulty
      const observed = addrs.filter(a => matchesVanityTarget(a, prefix, suffix, cs)).length
      // Within 5 standard deviations of the binomial count.
      expect(Math.abs(observed - expected)).toBeLessThan(5 * Math.sqrt(expected) + 1)
    }
    expect(compilePattern('Moon', '', true).difficulty).toBeGreaterThan(58 ** 4 * 10)
  })
})
