import VanityWorker from './vanityWorker?worker'
import type { SearchTarget } from '../searchTarget'

type SearchResult = {
  checked: number
  found: { privHex: string; address: string } | null
}

type WorkerMessage = { type: 'result'; id: number } & SearchResult

export type WorkerPool = {
  search(
    prefix: string,
    suffix: string,
    batchSize: number,
    target: SearchTarget,
    caseSensitive: boolean
  ): Promise<SearchResult>
  destroy(): void
  workerCount: number
}

export function createWorkerPool(): WorkerPool {
  const workerCount = Math.max(1, navigator.hardwareConcurrency || 4)
  const workers: Worker[] = []
  const pending = new Map<number, (result: SearchResult) => void>()
  let nextId = 0
  let destroyed = false

  for (let i = 0; i < workerCount; i++) {
    const worker = new VanityWorker()
    worker.onmessage = (e: MessageEvent<WorkerMessage>) => {
      const { id, checked, found } = e.data
      const resolve = pending.get(id)
      if (!resolve) return
      pending.delete(id)
      resolve({ checked, found })
    }
    workers.push(worker)
  }

  let workerIndex = 0

  function nextWorker(): Worker {
    const worker = workers[workerIndex]
    workerIndex = (workerIndex + 1) % workerCount
    return worker
  }

  function search(
    prefix: string,
    suffix: string,
    batchSize: number,
    target: SearchTarget,
    caseSensitive: boolean
  ): Promise<SearchResult> {
    if (destroyed) return Promise.reject(new Error('Pool destroyed'))

    return new Promise((resolve) => {
      const id = nextId++
      pending.set(id, resolve)
      nextWorker().postMessage({ type: 'search', id, batchSize, prefix, suffix, caseSensitive, target })
    })
  }

  function destroy() {
    destroyed = true
    workers.forEach(w => w.terminate())
    workers.length = 0
    pending.clear()
  }

  return { search, destroy, workerCount }
}
