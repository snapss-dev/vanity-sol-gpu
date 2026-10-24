// SHA-512 of a 32-byte ed25519 seed, generated for WGSL (which has no u64)
// and as a JavaScript twin for the test-suite. Only the first half of the
// digest matters for the public key: it becomes the clamped secret scalar.
//
// u64 values are [hi, lo] pairs of u32. The 80 rounds run as 5 iterations of
// 16 unrolled rounds, so the body keeps every message word in a named
// register (no runtime-indexed arrays) while staying a fifth of the size of a
// fully unrolled hash: GPU shader build time scales with code size.
import { Code, type Target } from './codegen'

type U64 = [hi: string, lo: string]

const K: bigint[] = [
  0x428a2f98d728ae22n, 0x7137449123ef65cdn, 0xb5c0fbcfec4d3b2fn, 0xe9b5dba58189dbbcn, 0x3956c25bf348b538n,
  0x59f111f1b605d019n, 0x923f82a4af194f9bn, 0xab1c5ed5da6d8118n, 0xd807aa98a3030242n, 0x12835b0145706fben,
  0x243185be4ee4b28cn, 0x550c7dc3d5ffb4e2n, 0x72be5d74f27b896fn, 0x80deb1fe3b1696b1n, 0x9bdc06a725c71235n,
  0xc19bf174cf692694n, 0xe49b69c19ef14ad2n, 0xefbe4786384f25e3n, 0x0fc19dc68b8cd5b5n, 0x240ca1cc77ac9c65n,
  0x2de92c6f592b0275n, 0x4a7484aa6ea6e483n, 0x5cb0a9dcbd41fbd4n, 0x76f988da831153b5n, 0x983e5152ee66dfabn,
  0xa831c66d2db43210n, 0xb00327c898fb213fn, 0xbf597fc7beef0ee4n, 0xc6e00bf33da88fc2n, 0xd5a79147930aa725n,
  0x06ca6351e003826fn, 0x142929670a0e6e70n, 0x27b70a8546d22ffcn, 0x2e1b21385c26c926n, 0x4d2c6dfc5ac42aedn,
  0x53380d139d95b3dfn, 0x650a73548baf63den, 0x766a0abb3c77b2a8n, 0x81c2c92e47edaee6n, 0x92722c851482353bn,
  0xa2bfe8a14cf10364n, 0xa81a664bbc423001n, 0xc24b8b70d0f89791n, 0xc76c51a30654be30n, 0xd192e819d6ef5218n,
  0xd69906245565a910n, 0xf40e35855771202an, 0x106aa07032bbd1b8n, 0x19a4c116b8d2d0c8n, 0x1e376c085141ab53n,
  0x2748774cdf8eeb99n, 0x34b0bcb5e19b48a8n, 0x391c0cb3c5c95a63n, 0x4ed8aa4ae3418acbn, 0x5b9cca4f7763e373n,
  0x682e6ff3d6b2b8a3n, 0x748f82ee5defb2fcn, 0x78a5636f43172f60n, 0x84c87814a1f0ab72n, 0x8cc702081a6439ecn,
  0x90befffa23631e28n, 0xa4506cebde82bde9n, 0xbef9a3f7b2c67915n, 0xc67178f2e372532bn, 0xca273eceea26619cn,
  0xd186b8c721c0c207n, 0xeada7dd6cde0eb1en, 0xf57d4f7fee6ed178n, 0x06f067aa72176fban, 0x0a637dc5a2c898a6n,
  0x113f9804bef90daen, 0x1b710b35131c471bn, 0x28db77f523047d84n, 0x32caab7b40c72493n, 0x3c9ebe0a15c9bebcn,
  0x431d67c49c100d4cn, 0x4cc5d4becb3e42b6n, 0x597f299cfc657e2an, 0x5fcb6fab3ad6faecn, 0x6c44198c4a475817n,
]

const H0: bigint[] = [
  0x6a09e667f3bcc908n, 0xbb67ae8584caa73bn, 0x3c6ef372fe94f82bn, 0xa54ff53a5f1d36f1n,
  0x510e527fade682d1n, 0x9b05688c2b3e6c1fn, 0x1f83d9abfb41bd6bn, 0x5be0cd19137e2179n,
]

const hi32 = (v: bigint) => Number(v >> 32n)
const lo32 = (v: bigint) => Number(v & 0xffffffffn)

function rotr(c: Code, [hi, lo]: U64, n: number): U64 {
  if (n === 32) return [lo, hi]
  if (n > 32) return rotr(c, [lo, hi], n - 32)
  return [
    c.or(c.shr(hi, n), c.shl(lo, 32 - n)),
    c.or(c.shr(lo, n), c.shl(hi, 32 - n)),
  ]
}

function shr(c: Code, [hi, lo]: U64, n: number): U64 {
  return [c.shr(hi, n), c.or(c.shr(lo, n), c.shl(hi, 32 - n))]
}

function xor3(c: Code, a: U64, b: U64, d: U64): U64 {
  return [c.xor(c.xor(a[0], b[0]), d[0]), c.xor(c.xor(a[1], b[1]), d[1])]
}

function bind(c: Code, prefix: string, v: U64): U64 {
  return [c.tmp(prefix + 'h', v[0]), c.tmp(prefix + 'l', v[1])]
}

/** Sum of u64 terms mod 2^64. */
function add(c: Code, prefix: string, terms: U64[]): U64 {
  let [hi, lo] = terms[0]
  for (let i = 1; i < terms.length; i++) {
    const s = c.tmp(prefix + 'l', c.add(lo, terms[i][1]))
    hi = c.tmp(prefix + 'h', c.add(c.add(hi, terms[i][0]), c.lt(s, terms[i][1])))
    lo = s
  }
  return [hi, lo]
}

const bigSigma0 = (c: Code, x: U64) => xor3(c, rotr(c, x, 28), rotr(c, x, 34), rotr(c, x, 39))
const bigSigma1 = (c: Code, x: U64) => xor3(c, rotr(c, x, 14), rotr(c, x, 18), rotr(c, x, 41))
const smallSigma0 = (c: Code, x: U64) => xor3(c, rotr(c, x, 1), rotr(c, x, 8), shr(c, x, 7))
const smallSigma1 = (c: Code, x: U64) => xor3(c, rotr(c, x, 19), rotr(c, x, 61), shr(c, x, 6))

function bswap(c: Code, x: string): string {
  return c.or(
    c.or(c.shl(c.and(x, c.lit(0xff)), 24), c.shl(c.and(x, c.lit(0xff00)), 8)),
    c.or(c.and(c.shr(x, 8), c.lit(0xff00)), c.shr(x, 24)),
  )
}

/**
 * sha512_scalar(seed) -> the clamped ed25519 secret scalar, as 8 little-endian
 * u32 limbs. `seed` holds the 32 seed bytes as 8 big-endian u32 words (the
 * message words SHA-512 reads), so the input needs no byte swapping.
 */
function genSha512Scalar(c: Code): string {
  const kTable = c.target === 'wgsl'
    ? `const K512 = array<u32, 160>(${K.flatMap(k => [c.lit(hi32(k)), c.lit(lo32(k))]).join(', ')});`
    : `const K512 = [${K.flatMap(k => [c.lit(hi32(k)), c.lit(lo32(k))]).join(', ')}];`
  const fn = c.fn('sha512_scalar', [['seed', 'array<u32, 8>']], 'array<u32, 8>', () => {
    const z = c.lit(0)
    const init: U64[] = [
      ['seed[0]', 'seed[1]'], ['seed[2]', 'seed[3]'], ['seed[4]', 'seed[5]'], ['seed[6]', 'seed[7]'],
      [c.lit(0x80000000), z], [z, z], [z, z], [z, z], [z, z], [z, z], [z, z], [z, z], [z, z], [z, z], [z, z],
      [z, c.lit(256)],
    ]
    const W: U64[] = init.map((v, i) => [c.var(`w${i}h`, v[0]), c.var(`w${i}l`, v[1])])
    const S: U64[] = H0.map((v, i) => [c.var(`s${i}h`, c.lit(hi32(v))), c.var(`s${i}l`, c.lit(lo32(v)))])
    c.loop(5, () => {
      // State registers are renamed instead of moved; 16 rounds bring them back.
      let st = [...S]
      for (let j = 0; j < 16; j++) {
        const [a, b, cc, d, e, f, g, h] = st
        const kIdx = c.target === 'wgsl' ? `(rnd * 32u + ${2 * j}u)` : `(rnd * 32 + ${2 * j})`
        const k: U64 = [`K512[${kIdx}]`, `K512[${kIdx} + ${c.lit(1)}]`]
        const ch: U64 = [
          c.xor(c.and(e[0], f[0]), c.and(c.not(e[0]), g[0])),
          c.xor(c.and(e[1], f[1]), c.and(c.not(e[1]), g[1])),
        ]
        const maj: U64 = [
          c.xor(c.xor(c.and(a[0], b[0]), c.and(a[0], cc[0])), c.and(b[0], cc[0])),
          c.xor(c.xor(c.and(a[1], b[1]), c.and(a[1], cc[1])), c.and(b[1], cc[1])),
        ]
        const t1 = add(c, 'a', [h, bind(c, 'S1', bigSigma1(c, e)), bind(c, 'ch', ch), k, W[j]])
        const t2 = add(c, 'b', [bind(c, 'S0', bigSigma0(c, a)), bind(c, 'mj', maj)])
        const ne = add(c, 'e', [d, t1])
        const na = add(c, 'n', [t1, t2])
        c.set(d[0], ne[0]); c.set(d[1], ne[1])
        c.set(h[0], na[0]); c.set(h[1], na[1])
        st = [h, a, b, cc, d, e, f, g]
      }
      // Expand the next 16 message words in place (wasted on the last pass).
      for (let j = 0; j < 16; j++) {
        const s1 = bind(c, 'm1', smallSigma1(c, W[(j + 14) % 16]))
        const s0 = bind(c, 'm0', smallSigma0(c, W[(j + 1) % 16]))
        const nw = add(c, 'w', [W[j], s1, W[(j + 9) % 16], s0])
        c.set(W[j][0], nw[0]); c.set(W[j][1], nw[1])
      }
    }, 'rnd')
    // digest[0..32] = H0..H3 + state, read as little-endian scalar limbs.
    const out: string[] = []
    for (let i = 0; i < 4; i++) {
      const [hi, lo] = add(c, 'o', [[c.lit(hi32(H0[i])), c.lit(lo32(H0[i]))], S[i]])
      out.push(c.tmp('q', bswap(c, hi)), c.tmp('q', bswap(c, lo)))
    }
    // ed25519 clamping: clear the low 3 bits and bit 255, set bit 254.
    out[0] = c.tmp('q', c.and(out[0], c.lit(0xfffffff8)))
    out[7] = c.tmp('q', c.or(c.and(out[7], c.lit(0x7fffffff)), c.lit(0x40000000)))
    c.emit(`return ${c.array('u32', out)};`)
  })
  return kTable + '\n' + fn
}

export function generateSha512(target: Target): string {
  return genSha512Scalar(new Code(target))
}
