import { createReadStream } from 'node:fs'
import { createInterface } from 'node:readline'
import { resolve } from 'node:path'

const file = resolve(process.cwd(), 'datasets/pixelart-dataset/.gitrepo/metadata.jsonl')
const cache = new Map<string, Record<string, unknown> | null>()
export async function findSourceMetadata(path: string) {
  if (cache.has(path)) return cache.get(path)
  const target = path.replace(/^data\//, '')
  try {
    const input = createReadStream(file, { encoding: 'utf8' })
    const lines = createInterface({ input, crlfDelay: Infinity })
    try {
      for await (const line of lines) {
        if (!line.trim()) continue
        const row = JSON.parse(line) as Record<string, unknown>
        const candidate = String(row.image || row.file_name || '').replace(/^\/api\/sprites\//, '').replace(/^data\//, '')
        if (candidate === target) {
          cache.set(path, row)
          return row
        }
      }
    } finally { lines.close(); input.destroy() }
  } catch { return null }
  cache.set(path, null)
  return null
}
