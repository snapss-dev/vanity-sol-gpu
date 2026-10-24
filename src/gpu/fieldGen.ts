// Curve25519 field arithmetic (mod p = 2^255 - 19) over 8 little-endian u32
// limbs, generated as straight-line code. Every function takes and returns
// fully reduced elements (< p). See codegen.ts for why this is generated.
import { Code, type Target } from './codegen'

const FE = 'array<u32, 8>'
const LIMBS = [0, 1, 2, 3, 4, 5, 6, 7]

type Pair = [string, string]

/** Split a limb into 16-bit halves (hoisted so each limb is split once). */
function halves(c: Code, x: string): Pair {
  return [c.tmp(x + 'l', c.and(x, c.lit(0xffff))), c.tmp(x + 'h', c.shr(x, 16))]
}

/** Full 32x32 -> 64 multiply from pre-split halves. Returns [lo, hi]. */
function mul32(c: Code, a: string, ah: Pair, b: string, bh: Pair): Pair {
  const [al, aH] = ah
  const [bl, bH] = bh
  const ll = c.tmp('ll', c.mul(al, bl))
  // Neither intermediate can overflow: (2^16-1)^2 + (2^16-1) < 2^32.
  const m1 = c.tmp('m1', c.add(c.mul(aH, bl), c.shr(ll, 16)))
  const m2 = c.tmp('m2', c.add(c.mul(al, bH), c.and(m1, c.lit(0xffff))))
  const hi = c.tmp('hi', c.add(c.add(c.mul(aH, bH), c.shr(m1, 16)), c.shr(m2, 16)))
  const lo = c.tmp('lo', c.mul(a, b)) // native u32 multiply already yields the low word
  return [lo, hi]
}

/** x * k for a small constant k (< 2^16) as [lo, hi] (hi < k). */
function mulSmall(c: Code, x: string, k: number): Pair {
  const lo = c.tmp('ul', c.mul(x, c.lit(k)))
  const part = c.add(c.mul(c.shr(x, 16), c.lit(k)), c.shr(c.mul(c.and(x, c.lit(0xffff)), c.lit(k)), 16))
  const hi = c.tmp('uh', c.shr(part, 16))
  return [lo, hi]
}

/**
 * Sum a column of u32 terms plus an optional incoming carry. Returns the low
 * word and the number of 2^32 wraps (the outgoing carry), which is exact
 * because each single addition of two u32 values wraps at most once.
 */
function column(c: Code, terms: string[], carryIn: string | null): Pair {
  const all = carryIn ? [...terms, carryIn] : terms
  let acc = all[0]
  let carry: string | null = null
  for (let i = 1; i < all.length; i++) {
    const s = c.tmp('s', c.add(acc, all[i]))
    const k = c.lt(s, all[i])
    carry = carry ? c.tmp('k', c.add(carry, k)) : c.tmp('k', k)
    acc = s
  }
  return [acc, carry ?? c.lit(0)]
}

/** Add a small value (< 2^32) to an 8-limb number; the caller guarantees no overflow past limb 7. */
function addSmall(c: Code, r: string[], v: string): string[] {
  const out: string[] = []
  let carry = v
  for (let i = 0; i < 8; i++) {
    const s = c.tmp('t', c.add(r[i], carry))
    if (i < 7) carry = c.tmp('k', c.lt(s, carry))
    out.push(s)
  }
  return out
}

/**
 * Fully reduce v = r + top * 2^255 (r already has bit 255 cleared, top small)
 * using 2^255 == 19. After folding, v < 2^255 + 19 * (top + 1), and one
 * conditional subtraction of p (done as "+19, keep if bit 255 is set") is enough.
 */
function canon(c: Code, r: string[], top: string): string[] {
  const f = addSmall(c, r, c.tmp('ft', c.mul(top, c.lit(19))))
  const s = addSmall(c, f, c.lit(19))
  const ge = c.tmp('ge', c.shr(s[7], 31)) // f + 19 >= 2^255  <=>  f >= p
  return LIMBS.map(i => c.tmp('r', c.sel(ge, i === 7 ? c.and(s[7], c.lit(0x7fffffff)) : s[i], f[i])))
}

/** Split bit 255 off a 9-limb value (limbs r[0..7] plus a small `hi` word). */
function canon9(c: Code, r: string[], hi: string): string[] {
  const top = c.tmp('tp', c.or(c.shl(hi, 1), c.shr(r[7], 31)))
  return canon(c, [...r.slice(0, 7), c.tmp('t7', c.and(r[7], c.lit(0x7fffffff)))], top)
}

/** Reduce a 512-bit product (16 limbs) mod p using 2^256 == 38. */
function reduce(c: Code, p: string[]): string[] {
  const L = p.slice(0, 8)
  const H = p.slice(8, 16)
  // t = L + 38 * H  (< 39 * 2^256, so 8 limbs plus a small t8)
  const u = H.map(h => mulSmall(c, h, 38))
  const t: string[] = []
  let carry: string | null = null
  for (let k = 0; k < 8; k++) {
    const terms = [L[k], u[k][0]]
    if (k > 0) terms.push(u[k - 1][1])
    const [s, k2]: Pair = column(c, terms, carry)
    t.push(s)
    carry = k2
  }
  const t8 = c.tmp('t8', c.add(u[7][1], carry!))
  return canon9(c, t, t8)
}

function loadLimbs(c: Code, name: string): string[] {
  return LIMBS.map(i => c.let(`${name}${i}`, `${name}[${i}]`))
}

/**
 * Add row x*b (x one limb) into the partial product `w` starting at limb
 * `at`. Undefined entries of `w` are zero. Writes w[at..at+8].
 */
function addRow(c: Code, w: Array<string | undefined>, at: number, x: string, xh: Pair, b: string[], bh: Pair[]) {
  let carry: string | null = null
  for (let j = 0; j < 8; j++) {
    const k = at + j
    const [lo, hi] = mul32(c, x, xh, b[j], bh[j])
    const terms = w[k] ? [w[k]!, lo] : [lo]
    const [s, kc] = terms.length === 1 && !carry ? [lo, c.lit(0)] : column(c, terms, carry)
    w[k] = s
    // carry_out = hi + wraps; the exact column total is < 2^64 so this fits.
    carry = c.tmp('c', c.add(hi, kc))
  }
  w[at + 8] = carry!
}

/**
 * fe_mul with `rows` rows of the schoolbook product per loop iteration
 * (8 = fully unrolled). Fewer rows per iteration make the function several
 * times smaller, which matters because GPU compilers inline every call site.
 * The rolled form keeps everything in registers by sliding a window of
 * partial-product limbs and rotating the input/output limbs, instead of
 * indexing arrays with the loop counter.
 *
 * `iterations` is the loop bound expression. Drivers fully unroll a loop with
 * a constant bound (undoing the size win), so the shader passes a value it
 * reads from a uniform; the JS twin passes the constant 8 / rows.
 */
function genMul(c: Code, rows: number, iterations: string): string {
  if (8 % rows !== 0) throw new Error('rows must divide 8')
  return c.fn('fe_mul', [['a', FE], ['b', FE]], FE, () => {
    const b = loadLimbs(c, 'b')
    const bh = b.map(x => halves(c, x))
    if (rows === 8) {
      const a = loadLimbs(c, 'a')
      const p: Array<string | undefined> = new Array(16)
      for (let i = 0; i < 8; i++) addRow(c, p, i, a[i], halves(c, a[i]), b, bh)
      c.emit(`return ${c.array('u32', reduce(c, p as string[]))};`)
      return
    }
    const z = c.lit(0)
    const x = LIMBS.map(i => c.var(`x${i}`, `a[${i}]`)) // a, consumed `rows` limbs per iteration
    const r = LIMBS.map(i => c.var(`r${i}`, z))         // partial product limbs i.. i+7
    const o = LIMBS.map(i => c.var(`o${i}`, z))         // finished low limbs, shifted in from the top
    c.loop(iterations, () => {
      const w: Array<string | undefined> = [...r]
      for (let k = 0; k < rows; k++) addRow(c, w, k, x[k], halves(c, x[k]), b, bh)
      // The lowest `rows` limbs are now final: shift them into o, slide the window.
      const nextO = LIMBS.map(q => (q < 8 - rows ? o[q + rows] : w[q - (8 - rows)]!))
      const nextR = LIMBS.map(q => w[q + rows]!)
      const nextX = LIMBS.map(q => (q + rows < 8 ? x[q + rows] : z))
      const tmpO = nextO.map(v => c.tmp('no', v))
      const tmpR = nextR.map(v => c.tmp('nr', v))
      const tmpX = nextX.map(v => c.tmp('nx', v))
      LIMBS.forEach(q => { c.set(o[q], tmpO[q]); c.set(r[q], tmpR[q]); c.set(x[q], tmpX[q]) })
    }, 'row')
    c.emit(`return ${c.array('u32', reduce(c, [...o, ...r]))};`)
  })
}

function genSqr(c: Code): string {
  return c.fn('fe_sqr', [['a', FE]], FE, () => {
    const a = loadLimbs(c, 'a')
    const ah = a.map(x => halves(c, x))
    // Off-diagonal products a_i*a_j (i < j), accumulated once then doubled.
    const p: Array<string | undefined> = new Array(16)
    for (let i = 0; i < 7; i++) {
      let carry: string | null = null
      for (let j = i + 1; j < 8; j++) {
        const k = i + j
        const [lo, hi] = mul32(c, a[i], ah[i], a[j], ah[j])
        const terms = p[k] ? [p[k]!, lo] : [lo]
        const [s, kc] = terms.length === 1 && !carry ? [lo, c.lit(0)] : column(c, terms, carry)
        p[k] = s
        carry = c.tmp('c', c.add(hi, kc))
      }
      p[i + 8] = carry!
    }
    const z = c.lit(0)
    const cross = Array.from({ length: 16 }, (_, k) => p[k] ?? z)
    const dbl = cross.map((v, k) =>
      c.tmp('d', k === 0 ? c.shl(v, 1) : c.or(c.shl(v, 1), c.shr(cross[k - 1], 31))))
    // Add the diagonal squares a_i^2 at limbs 2i, 2i+1.
    const out: string[] = []
    let carry: string | null = null
    for (let i = 0; i < 8; i++) {
      const [lo, hi] = mul32(c, a[i], ah[i], a[i], ah[i])
      const [s0, k0] = column(c, [dbl[2 * i], lo], carry)
      const [s1, k1] = column(c, [dbl[2 * i + 1], hi], k0)
      out.push(s0, s1)
      carry = k1
    }
    c.emit(`return ${c.array('u32', reduce(c, out))};`)
  })
}

function genAdd(c: Code): string {
  return c.fn('fe_add', [['a', FE], ['b', FE]], FE, () => {
    const a = loadLimbs(c, 'a')
    const b = loadLimbs(c, 'b')
    const s: string[] = []
    let carry: string | null = null
    for (let i = 0; i < 8; i++) {
      const [v, k] = column(c, [a[i], b[i]], carry)
      s.push(v)
      carry = k
    }
    // a + b < 2p < 2^256, so there is no carry out of limb 7.
    c.emit(`return ${c.array('u32', canon9(c, s, c.lit(0)))};`)
  })
}

// 2p = 2^256 - 38, as little-endian limbs.
const TWO_P = [0xffffffda, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]

function genSub(c: Code): string {
  return c.fn('fe_sub', [['a', FE], ['b', FE]], FE, () => {
    const a = loadLimbs(c, 'a')
    const b = loadLimbs(c, 'b')
    // n = 2p - b (no borrow: b < p), then a + n < 3p < 2^257.
    const n: string[] = []
    let borrow: string | null = null
    for (let i = 0; i < 8; i++) {
      const k = c.lit(TWO_P[i])
      const d0 = c.tmp('d', c.sub(k, b[i]))
      const b0 = c.lt(k, b[i])
      if (borrow) {
        n.push(c.tmp('d', c.sub(d0, borrow)))
        borrow = c.tmp('b', c.or(b0, c.lt(d0, borrow)))
      } else {
        n.push(d0)
        borrow = c.tmp('b', b0)
      }
    }
    const s: string[] = []
    let carry: string | null = null
    for (let i = 0; i < 8; i++) {
      const [v, k] = column(c, [a[i], n[i]], carry)
      s.push(v)
      carry = k
    }
    c.emit(`return ${c.array('u32', canon9(c, s, carry!))};`)
  })
}

/**
 * Inversion by Fermat (a^(p-2), p - 2 = 2^255 - 21) with ref10's addition
 * chain: 254 squarings + 11 multiplications. Each step squares pw[src] some
 * number of times and multiplies by pw[mul] (z^2 is computed as z * z).
 * Powers: z^2, z^9, z^11, z^(2^5-1), z^(2^10-1), ..., z^(2^250-1).
 *
 * The chain is driven by a table rather than written out, so fe_sqr and
 * fe_mul each appear once: GPU compilers inline every call site.
 */
const INV_CHAIN: Array<[src: number, squarings: number, mul: number, dst: number]> = [
  [0, 0, 0, 1], [1, 2, 0, 2], [2, 0, 1, 3], [3, 1, 2, 4], [4, 5, 4, 5], [5, 10, 5, 6],
  [6, 20, 6, 7], [7, 10, 5, 8], [8, 50, 8, 9], [9, 100, 9, 10], [10, 50, 8, 11], [11, 5, 3, 11],
]

function genInv(c: Code): string {
  const col = (k: number) => INV_CHAIN.map(step => c.lit(step[k]))
  const table = (name: string, k: number) => c.target === 'wgsl'
    ? `const ${name} = array<u32, ${INV_CHAIN.length}>(${col(k).join(', ')});`
    : `const ${name} = [${col(k).join(', ')}];`
  const fn = c.fn('fe_inv', [['a', FE]], FE, () => {
    c.emit(c.target === 'wgsl' ? `var pw: array<${FE}, 12>;` : 'const pw = new Array(12);')
    c.set('pw[0]', 'a')
    c.loop(INV_CHAIN.length, () => {
      c.var('r', 'pw[INV_SRC[s]]')
      c.loop('INV_SQ[s]', () => c.set('r', 'fe_sqr(r)'))
      c.set('pw[INV_DST[s]]', 'fe_mul(r, pw[INV_MUL[s]])')
    }, 's')
    c.emit('return pw[11];')
  })
  return [table('INV_SRC', 0), table('INV_SQ', 1), table('INV_MUL', 2), table('INV_DST', 3), fn].join('\n')
}

export type FieldOptions = {
  /** Schoolbook rows per loop iteration in fe_mul (1, 2, 4 or 8 = unrolled). */
  mulRows?: number
  /**
   * WGSL only: name of a u32 holding 8 / mulRows at run time, used as the
   * rolled multiply's loop bound (see genMul). Defaults to the constant.
   */
  mulItersVar?: string
}

/** All field functions for the given target, in dependency order. */
export function generateField(target: Target, opts: FieldOptions = {}): string {
  const c = new Code(target)
  const rows = opts.mulRows ?? 8
  const iterations = target === 'wgsl' && opts.mulItersVar ? opts.mulItersVar : c.lit(8 / rows)
  return [genAdd(c), genSub(c), genMul(c, rows, iterations), genSqr(c), genInv(c)].join('\n\n')
}
