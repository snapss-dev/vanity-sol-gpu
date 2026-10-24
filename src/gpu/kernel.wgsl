// Solana vanity search kernel. The field arithmetic (fe_*) and sha512_scalar
// are generated (see fieldGen.ts / sha512Gen.ts) and prepended.
//
// A Solana keypair is a random 32-byte seed; the public key is a*B where a is
// the clamped first half of SHA-512(seed). Because of the hash, keys can't be
// stepped from one to the next, so each candidate costs a full fixed-base
// scalar multiplication: 32 mixed additions from a table of v*256^w*B (entry
// v = 0 is the identity, so the loop has no special cases).
//
// Every invocation derives K seeds from the per-dispatch random seed and its
// own index, computes the K points in extended coordinates, then shares one
// field inversion across them (Montgomery's trick) to get the encoded public
// keys. Points and prefix products wait in `scratch` between the two passes.

alias Fe = array<u32, 8>;

const WG: u32 = __WG__u;
const K: u32 = __K__u;               // keys per invocation
const MAX_THREADS: u32 = __MAX_THREADS__u;
const MAX_RESULTS: u32 = __MAX_RESULTS__u;
const RESULT_WORDS: u32 = 16u;       // seed[8], public key[8]
const TABLE_WORDS: u32 = 24u;        // y + x, y - x, 2d*x*y
const PATTERN_HEADER: u32 = 44u;     // suffix digit pairs, then prefix ranges
const INTERVAL_WORDS: u32 = 16u;

struct Params {
  seed: array<vec4<u32>, 2>,         // fresh random seed for every dispatch
  count: u32,                        // invocations in this dispatch
  mul_iters: u32,
  suffix_len: u32,
  interval_count: u32,
}

struct Results {
  count: atomic<u32>,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
  data: array<u32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> table: array<u32>;
@group(0) @binding(2) var<storage, read> pattern: array<u32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<u32>;
@group(0) @binding(4) var<storage, read_write> results: Results;

// fe_mul's loop count (8 / rows per iteration), copied from a uniform at entry.
// A bound the compiler can't see stops drivers from unrolling the rolled
// multiply back into one huge block per call site, which makes the first
// shader build take many seconds on some GPUs.
var<private> mul_iters: u32;

const FE_ZERO = Fe(0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
const FE_ONE = Fe(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

fn seed_for(tid: u32, k: u32) -> array<u32, 8> {
  let s0 = params.seed[0];
  let s1 = params.seed[1];
  // Unique per (invocation, key); the dispatch seed is fresh randomness.
  return array<u32, 8>(s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z ^ k, s1.w ^ tid);
}

fn table_fe(o: u32) -> Fe {
  return Fe(table[o], table[o + 1u], table[o + 2u], table[o + 3u],
            table[o + 4u], table[o + 5u], table[o + 6u], table[o + 7u]);
}

// scratch layout: [key][field][limb][thread] so a warp's accesses coalesce.
fn scratch_index(k: u32, field: u32, limb: u32, tid: u32) -> u32 {
  return ((k * 4u + field) * 8u + limb) * MAX_THREADS + tid;
}

fn store_fe(k: u32, field: u32, tid: u32, v: Fe) {
  for (var i = 0u; i < 8u; i = i + 1u) {
    scratch[scratch_index(k, field, i, tid)] = v[i];
  }
}

fn load_fe(k: u32, field: u32, tid: u32) -> Fe {
  var v: Fe;
  for (var i = 0u; i < 8u; i = i + 1u) {
    v[i] = scratch[scratch_index(k, field, i, tid)];
  }
  return v;
}

fn bswap(x: u32) -> u32 {
  return ((x & 0xFFu) << 24u) | ((x & 0xFF00u) << 8u) | ((x >> 8u) & 0xFF00u) | (x >> 24u);
}

// a <= n, with a at pattern[o..o+8] and both most-significant word first.
fn range_le(o: u32, n: array<u32, 8>) -> bool {
  for (var i = 0u; i < 8u; i = i + 1u) {
    let a = pattern[o + i];
    if (a != n[i]) { return a < n[i]; }
  }
  return true;
}

// Does base58(pk) match the pattern? pk holds the key bytes as little-endian
// limbs; base58 reads them as one big-endian number n.
fn matches(pk: Fe) -> bool {
  var n: array<u32, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) { n[i] = bswap(pk[i]); }

  let ranges = params.interval_count;
  if (ranges > 0u) {
    // First range whose (exclusive) end is above n, then check its start.
    var lo = 0u;
    var hi = ranges;
    loop {
      if (lo >= hi) { break; }
      let mid = (lo + hi) >> 1u;
      if (range_le(PATTERN_HEADER + mid * INTERVAL_WORDS + 8u, n)) { lo = mid + 1u; } else { hi = mid; }
    }
    if (lo == ranges || !range_le(PATTERN_HEADER + lo * INTERVAL_WORDS, n)) { return false; }
  }

  // Suffix: peel base-58 digits off the low end, 16 bits at a time so every
  // intermediate fits in a u32.
  for (var j = 0u; j < params.suffix_len; j = j + 1u) {
    var rem = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
      let w = n[i];
      let a = (rem << 16u) | (w >> 16u);
      let qa = a / 58u;
      let b = ((a - qa * 58u) << 16u) | (w & 0xFFFFu);
      let qb = b / 58u;
      rem = b - qb * 58u;
      n[i] = (qa << 16u) | qb;
    }
    let want = pattern[j];
    if (rem != (want & 0xFFu) && rem != (want >> 8u)) { return false; }
  }
  return true;
}

@compute @workgroup_size(WG)
fn search(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tid = gid.x;
  if (tid >= params.count) { return; }
  mul_iters = params.mul_iters;

  var acc = FE_ONE;
  for (var k = 0u; k < K; k = k + 1u) {
    let sc = sha512_scalar(seed_for(tid, k));
    // Shift register over the scalar bytes (avoids indexing by the loop counter).
    var s = sc;
    var X = FE_ZERO;
    var Y = FE_ONE;
    var Z = FE_ONE;
    var T = FE_ZERO;
    for (var w = 0u; w < 32u; w = w + 1u) {
      let o = (w * 256u + (s[0] & 0xFFu)) * TABLE_WORDS;
      s = array<u32, 8>(
        (s[0] >> 8u) | (s[1] << 24u), (s[1] >> 8u) | (s[2] << 24u),
        (s[2] >> 8u) | (s[3] << 24u), (s[3] >> 8u) | (s[4] << 24u),
        (s[4] >> 8u) | (s[5] << 24u), (s[5] >> 8u) | (s[6] << 24u),
        (s[6] >> 8u) | (s[7] << 24u), s[7] >> 8u);
      // Extended += affine niels (a = -1 twisted Edwards, complete formula).
      let pa = fe_mul(fe_sub(Y, X), table_fe(o + 8u));
      let pb = fe_mul(fe_add(Y, X), table_fe(o));
      let pc = fe_mul(T, table_fe(o + 16u));
      let pd = fe_add(Z, Z);
      let e = fe_sub(pb, pa);
      let f = fe_sub(pd, pc);
      let g = fe_add(pd, pc);
      let h = fe_add(pb, pa);
      X = fe_mul(e, f);
      Y = fe_mul(g, h);
      T = fe_mul(e, h);
      Z = fe_mul(f, g);
    }
    store_fe(k, 0u, tid, X);
    store_fe(k, 1u, tid, Y);
    store_fe(k, 2u, tid, Z);
    acc = fe_mul(acc, Z);
    store_fe(k, 3u, tid, acc);
  }

  var inv = fe_inv(acc);
  var k = K;
  loop {
    if (k == 0u) { break; }
    k = k - 1u;
    var zi = inv;
    if (k > 0u) {
      zi = fe_mul(inv, load_fe(k - 1u, 3u, tid));
      inv = fe_mul(inv, load_fe(k, 2u, tid));
    }
    var pk = fe_mul(load_fe(k, 1u, tid), zi);
    let x = fe_mul(load_fe(k, 0u, tid), zi);
    pk[7] = pk[7] | ((x[0] & 1u) << 31u);
    if (!matches(pk)) { continue; }
    // Cheap guard so a trivially easy pattern can't hammer the atomic.
    if (atomicLoad(&results.count) >= MAX_RESULTS) { continue; }
    let slot = atomicAdd(&results.count, 1u);
    if (slot >= MAX_RESULTS) { continue; }
    let seed = seed_for(tid, k);
    let o = slot * RESULT_WORDS;
    for (var i = 0u; i < 8u; i = i + 1u) {
      results.data[o + i] = seed[i];
      results.data[o + 8u + i] = pk[i];
    }
  }
}
