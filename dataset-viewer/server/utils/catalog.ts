import { readFile, stat } from 'node:fs/promises'
import { resolve } from 'node:path'

type ImageRow = { path: string; caption: string; folder: string; name: string }
type Annotation = { source?: { root?: string; relative_path?: string }; model?: string; labels?: Record<string, unknown>; color_measurements?: any; [key: string]: any }
const indexPath = resolve(process.cwd(), 'datasets/pixelart-dataset/viewer-index.json')
const work = resolve(process.cwd(), '../notebook/dataset/taxonomy/taxonomy_work')
const livePath = resolve(work, 'annotations_v3_4_full_color.jsonl')
const palettePath = resolve(work, 'annotations_v3_4_exact_palette.jsonl')
let version = ''
let pending: Promise<Catalog> | null = null
export type Catalog = { images: ImageRow[]; annotations: Map<string, Annotation> }

async function signature(path: string) {
  try { const info = await stat(path); return `${info.mtimeMs}:${info.size}` } catch { return '-' }
}
async function readAnnotations(path: string, rows: Map<string, Annotation>, paletteOnly = false) {
  let text: string
  try { text = await readFile(path, 'utf8') } catch { return }
  const lines = text.split('\n')
  if (!text.endsWith('\n')) lines.pop()
  for (const line of lines) {
    if (!line.trim()) continue
    let row: Annotation
    try { row = JSON.parse(line) } catch { continue }
    if (!String(row.model || '').toLowerCase().includes('qwen3.8') || !String(row.model || '').toLowerCase().includes('gyro')) continue
    if ((row.source?.root || 'data') !== 'data' || !row.source?.relative_path) continue
    const path = `data/${row.source.relative_path.replaceAll('\\', '/')}`
    const previous = rows.get(path)
    if (paletteOnly) {
      if (previous && row.color_measurements?.status === 'measured') previous.color_measurements = row.color_measurements
      else if (!previous) rows.set(path, row)
    } else rows.set(path, row)
  }
}
export async function loadCatalog(): Promise<Catalog> {
  const next = (await Promise.all([indexPath, livePath, palettePath].map(signature))).join('|')
  if (pending && version === next) return pending
  version = next
  pending = (async () => {
    const images = JSON.parse(await readFile(indexPath, 'utf8')) as ImageRow[]
    const annotations = new Map<string, Annotation>()
    await readAnnotations(livePath, annotations)
    await readAnnotations(palettePath, annotations, true)
    return { images, annotations }
  })().catch(error => { pending = null; throw error })
  return pending
}
