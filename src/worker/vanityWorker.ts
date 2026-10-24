import { ed25519 } from '@noble/curves/ed25519.js'
import { bytesToHex } from '../utils/hex'
import { bytesToBase58, matchesVanityTarget } from '../wallet/solanaAddress'
import type { SearchTarget } from '../searchTarget'

type SearchMessage = {
  type: 'search'
  id: number
  batchSize: number
  prefix: string
  suffix: string
  caseSensitive: boolean
  // Program IDs and wallet addresses are both ed25519 public keys, so the
  // search is the same for either target.
  target: SearchTarget
}

type SearchWorkerResult = {
  type: 'result'
  id: number
  checked: number
  found: { privHex: string; address: string } | null
}

self.onmessage = (e: MessageEvent<SearchMessage>) => {
  const { type, id, batchSize, prefix, suffix, caseSensitive } = e.data
  if (type !== 'search') return

  const seeds = new Uint8Array(batchSize * 32)
  crypto.getRandomValues(seeds)

  for (let i = 0; i < batchSize; i++) {
    const seed = seeds.subarray(i * 32, i * 32 + 32)
    const address = bytesToBase58(ed25519.getPublicKey(seed))
    if (!matchesVanityTarget(address, prefix, suffix, caseSensitive)) continue

    const result: SearchWorkerResult = { type: 'result', id, checked: i + 1, found: { privHex: bytesToHex(seed), address } }
    self.postMessage(result)
    return
  }

  const result: SearchWorkerResult = { type: 'result', id, checked: batchSize, found: null }
  self.postMessage(result)
}
