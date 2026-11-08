import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const statusPath = resolve(process.cwd(), 'datasets', 'pixelart-dataset', '.viewer-download-status.json')

export default defineEventHandler(async () => {
  try {
    return JSON.parse(await readFile(statusPath, 'utf8'))
  } catch {
    return { status: 'missing', done: 0, total: 0, message: 'Dataset absent' }
  }
})
