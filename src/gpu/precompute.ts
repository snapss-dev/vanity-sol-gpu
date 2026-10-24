// Host-side tables and pattern encoding for the GPU kernel.
import { ed25519 } from '@noble/curves/ed25519.js'
import { BASE58_ALPHABET } from '../wallet/solanaAddress'

const P = (1n << 255n) - 19n
const D2 = (2n * ed25519.Point.CURVE().d) % P

export const WINDOW_BITS = 8
export const WINDOWS = 32 // the clamped scalar is < 2^255
export const TABLE_ENTRY_WORDS = 24

function putFe(out: Uint32Array, offset: number, v: bigint) {
  for (let i = 0; i < 8; i++) out[offset + i] = Number((v >> BigInt(32 * i)) & 0xffffffffn)
}

function modInv(a: bigint): bigint {
  let r = 1n
  let b = a % P
  for (let e = P - 2n; e > 0n; e >>= 1n, b = b * b % P) if (e & 1n) r = r * b % P
  return r
}

/**
 * v * 256^w * B for w < 32 and v < 256, in affine "niels" form
 * (y + x, y - x, 2d*x*y), the operand of the kernel's 7-multiplication mixed
 * addition. Entry v = 0 is the identity (1, 1, 0), so no window is special.
 */
export function buildBaseTable(): Uint32Array {
  const n = WINDOWS << WINDOW_BITS
  const points: InstanceType<typeof ed25519.Point>[] = []
  let base = ed25519.Point.BASE
  for (let w = 0; w < WINDOWS; w++) {
    let acc = ed25519.Point.ZERO
    for (let v = 0; v < 1 << WINDOW_BITS; v++) {
      points.push(acc)
      acc = acc.add(base)
    }
    base = acc // 256 * base
  }
  // Normalize all projective points with one shared inversion (Montgomery's trick).
  const zs = points.map(p => p.Z % P)
  const prefix: bigint[] = new Array(n)
  let run = 1n
  for (let i = 0; i < n; i++) { prefix[i] = run; run = run * zs[i] % P }
  let inv = modInv(run)
  const out = new Uint32Array(n * TABLE_ENTRY_WORDS)
  for (let i = n - 1; i >= 0; i--) {
    const zi = inv * prefix[i] % P
    inv = inv * zs[i] % P
    const x = (points[i].X * zi % P + P) % P
    const y = (points[i].Y * zi % P + P) % P
    const o = i * TABLE_ENTRY_WORDS
    putFe(out, o, (y + x) % P)
    putFe(out, o + 8, (y - x + P) % P)
    putFe(out, o + 16, D2 * x % P * y % P)
  }
  return out
}

/** The kernel's seed words are the 32 seed bytes read as big-endian u32s. */
export function seedWordsToBytes(words: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(32)
  const view = new DataView(out.buffer)
  for (let i = 0; i < 8; i++) view.setUint32(4 * i, words[i] >>> 0, false)
  return out
}

/** The kernel reports public keys as 8 little-endian u32 limbs of the 32 bytes. */
export function pubWordsToBytes(words: ArrayLike<number>): Uint8Array {
  const out = new Uint8Array(32)
  const view = new DataView(out.buffer)
  for (let i = 0; i < 8; i++) view.setUint32(4 * i, words[i] >>> 0, true)
  return out
}

// ── pattern encoding ──
//
// A Solana address is base58 of the 32 public-key bytes: one '1' per leading
// zero byte, then the big-endian value n in base 58. The kernel never builds
// that string. A prefix becomes a sorted list of disjoint ranges of n, and a
// suffix becomes the allowed base-58 digits of n from the least significant
// end, which the kernel peels off with divisions by 58 (most keys fail on the
// first digit).

export const MAX_PATTERN_LEN = 44
export const MAX_INTERVALS = 8192
export const INTERVAL_WORDS = 16 // lo, hi as 8 big-endian-ordered u32 words each
export const PATTERN_HEADER_WORDS = MAX_PATTERN_LEN // one suffix digit pair per position

export type CompiledPattern = {
  /** Number of prefix ranges; 0 means no prefix constraint. */
  intervalCount: number
  suffixLength: number
  /**
   * Expected keys per match: exact, from the share of all keys the prefix
   * ranges cover and the suffix digits allowed. It is not 58^length: only
   * 43-character addresses can start with a character above "J", for example.
   */
  difficulty: number
  /** Suffix digit pairs (header) followed by the ranges. */
  words: Uint32Array
}

const DIGIT = new Map([...BASE58_ALPHABET].map((ch, i) => [ch, i]))

/** Base-58 digits that may appear at a position: both cases when allowed. */
function candidates(ch: string, caseSensitive: boolean): number[] {
  const out = new Set<number>([DIGIT.get(ch)!])
  if (!caseSensitive) {
    for (const alt of [ch.toLowerCase(), ch.toUpperCase()]) {
      const d = DIGIT.get(alt)
      if (d !== undefined) out.add(d)
    }
  }
  return [...out]
}

function putBig(out: Uint32Array, offset: number, v: bigint) {
  // Most significant word first, matching how the kernel orders n.
  for (let i = 0; i < 8; i++) out[offset + i] = Number((v >> BigInt(32 * (7 - i))) & 0xffffffffn)
}

export function compilePattern(prefix: string, suffix: string, caseSensitive: boolean): CompiledPattern {
  if (prefix.length + suffix.length > MAX_PATTERN_LEN) throw new Error('Pattern too long')
  for (const ch of prefix + suffix) if (!DIGIT.has(ch)) throw new Error(`Invalid base58 character "${ch}"`)

  const ranges: Array<[bigint, bigint]> = []
  if (prefix.length > 0) {
    let ones = 0
    while (ones < prefix.length && prefix[ones] === '1') ones++
    if (ones > 31) throw new Error('Prefix cannot be produced by any key')
    const top = 1n << BigInt(256 - 8 * ones)
    const rest = prefix.slice(ones)
    if (rest.length === 0) {
      ranges.push([0n, top]) // at least `ones` leading zero bytes
    } else {
      // Exactly `ones` leading zero bytes, then digits of n starting with `rest`.
      const floor = 1n << BigInt(248 - 8 * ones)
      let values = [0n]
      for (const ch of rest) {
        const ds = candidates(ch, caseSensitive)
        if (values.length * ds.length > MAX_INTERVALS) throw new Error('Too many case variants: turn on case-sensitive matching')
        values = values.flatMap(v => ds.map(d => v * 58n + BigInt(d)))
      }
      const k = rest.length
      for (const v of values) {
        for (let digits = k, scale = 1n; digits <= MAX_PATTERN_LEN; digits++, scale *= 58n) {
          const lo = v * scale > floor ? v * scale : floor
          const hi = (v + 1n) * scale < top ? (v + 1n) * scale : top
          if (lo < hi) ranges.push([lo, hi])
          if (v * scale >= top) break
        }
      }
      if (ranges.length === 0) throw new Error('Prefix cannot be produced by any key')
    }
  }
  ranges.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
  const merged: Array<[bigint, bigint]> = []
  for (const r of ranges) {
    const last = merged[merged.length - 1]
    if (last && r[0] <= last[1]) { if (r[1] > last[1]) last[1] = r[1] } else merged.push([r[0], r[1]])
  }
  if (merged.length > MAX_INTERVALS) throw new Error('Too many case variants: turn on case-sensitive matching')

  const words = new Uint32Array(PATTERN_HEADER_WORDS + merged.length * INTERVAL_WORDS)
  let probability = 1
  for (let j = 0; j < suffix.length; j++) {
    const ds = candidates(suffix[suffix.length - 1 - j], caseSensitive)
    words[j] = ds[0] | ((ds[1] ?? ds[0]) << 8)
    probability *= ds.length / 58
  }
  if (merged.length > 0) {
    // Covered share of 2^256, kept to ~15 significant digits.
    const covered = merged.reduce((sum, [lo, hi]) => sum + (hi - lo), 0n)
    const shift = BigInt(Math.max(0, covered.toString(2).length - 53))
    probability *= Number(covered >> shift) * 2 ** (Number(shift) - 256)
  }
  merged.forEach(([lo, hi], i) => {
    // hi = 2^256 is stored as all-ones; n < 2^256 always, so make it inclusive-safe.
    putBig(words, PATTERN_HEADER_WORDS + i * INTERVAL_WORDS, lo)
    putBig(words, PATTERN_HEADER_WORDS + i * INTERVAL_WORDS + 8, hi >= 1n << 256n ? (1n << 256n) - 1n : hi)
  })
  return { intervalCount: merged.length, suffixLength: suffix.length, difficulty: 1 / probability, words }
}
