import { readFile, stat } from 'node:fs/promises'
import { resolve } from 'node:path'

type Annotation = {
  asset_id?: string
  source?: { root?: string; relative_path?: string }
  labels?: Record<string, unknown>
  caption_candidate?: string
  need_rework?: boolean | null
  review_status?: string
  model?: string
  annotated_at_utc?: string
  [key: string]: unknown
}

const repoId = 'ethanzxv/pixelart-dataset'
const defaultRemoteFile = 'annotations/dataset_lab_sync/20261112T093240Z/annotations_v3_4_color_partial.jsonl'
const candidatePaths = [
  process.env.QWEN_ANNOTATIONS_PATH,
  resolve(process.cwd(), '../notebook/dataset/taxonomy/taxonomy_work/annotations_v3_4_full_color.jsonl'),
  resolve(process.cwd(), 'notebook/dataset/taxonomy/taxonomy_work/annotations_v3_4_full_color.jsonl'),
  resolve(process.cwd(), 'data/qwen_annotations.jsonl')
].filter((value): value is string => Boolean(value))

let cachedRows: Annotation[] = []
let cachedPath = ''
let cachedMtime = 0
let cachedSize = 0
let remoteLoad: Promise<Annotation[]> | null = null

async function parseJsonl(content: string): Promise<Annotation[]> {
  const rows: Annotation[] = []
  const lines = content.split('\n')
  if (!content.endsWith('\n')) lines.pop()
  for (const line of lines) {
    if (!line.trim()) continue
    const row = JSON.parse(line) as Annotation
    const model = String(row.model ?? '').toLowerCase()
    if (model.includes('qwen3.8') && model.includes('gyro')) rows.push(row)
  }
  return rows
}

async function loadRows(): Promise<Annotation[]> {
  if (cachedPath === 'remote' && cachedRows.length) return cachedRows
  for (const filePath of candidatePaths) {
    try {
      const info = await stat(filePath)
      if (cachedPath === filePath && cachedMtime === info.mtimeMs && cachedSize === info.size) return cachedRows
      const rows = await parseJsonl(await readFile(filePath, 'utf8'))
      if (rows.length) {
        cachedRows = rows
        cachedPath = filePath
        cachedMtime = info.mtimeMs
        cachedSize = info.size
        return rows
      }
    } catch {
      // Continue to the next local path or the configured Hub snapshot.
    }
  }

  if (!remoteLoad) {
    remoteLoad = (async () => {
      const file = process.env.QWEN_ANNOTATIONS_HF_FILE || defaultRemoteFile
      const token = process.env.HF_TOKEN || process.env.HUGGINGFACE_HUB_TOKEN || process.env.HF_ACCESS_TOKEN
      const url = `https://huggingface.co/datasets/${repoId}/resolve/main/${file.split('/').map(encodeURIComponent).join('/')}`
      const response = await fetch(url, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        redirect: 'follow'
      })
      if (!response.ok) throw createError({ statusCode: response.status === 401 || response.status === 403 ? 503 : 502, statusMessage: 'Impossible de lire les annotations Qwen3.8 sur Hugging Face.' })
      const rows = await parseJsonl(await response.text())
      if (!rows.length) throw createError({ statusCode: 404, statusMessage: 'Le snapshot distant ne contient aucune annotation Qwen3.8 Gyro-S.' })
      return rows
    })().finally(() => { remoteLoad = null })
  }
  cachedRows = await remoteLoad
  cachedPath = 'remote'
  return cachedRows
}

export default defineEventHandler(async (event) => {
  let rows: Annotation[]
  try {
    rows = await loadRows()
  } catch (error) {
    if (isHttpError(error)) throw error
    throw createError({ statusCode: 503, statusMessage: 'Annotations Qwen3.8 indisponibles. Vérifie le fichier local ou la configuration HF_TOKEN.' })
  }

  const query = getQuery(event)
  const offset = Math.max(0, Number.parseInt(String(query.offset ?? '0'), 10) || 0)
  const limit = Math.min(48, Math.max(1, Number.parseInt(String(query.limit ?? '24'), 10) || 24))
  const term = typeof query.search === 'string' ? query.search.trim().toLocaleLowerCase() : ''
  const assetType = typeof query.asset_type === 'string' ? query.asset_type : ''
  const subjectType = typeof query.subject_type === 'string' ? query.subject_type : ''
  const quality = typeof query.quality === 'string' ? query.quality : ''
  const rework = typeof query.rework === 'string' ? query.rework : ''

  const matches = rows.filter((row) => {
    const labels = row.labels ?? {}
    if (assetType && labels.asset_type !== assetType) return false
    if (subjectType && labels.subject_type !== subjectType) return false
    if (quality && labels.visual_quality !== quality) return false
    const reworkValue = row.need_rework === null || row.need_rework === undefined ? 'unknown' : String(row.need_rework)
    if (rework && reworkValue !== rework) return false
    if (!term) return true
    return JSON.stringify(row).toLocaleLowerCase().includes(term)
  })

  const unique = (field: string) => [...new Set(rows.map((row) => String(row.labels?.[field] ?? '')).filter(Boolean))].sort()
  const facets = {
    asset_type: unique('asset_type'),
    subject_type: unique('subject_type'),
    visual_quality: unique('visual_quality'),
    need_rework: [...new Set(rows.map((row) => String(row.need_rework ?? 'unknown')))].sort()
  }
  return {
    total: rows.length,
    filtered: matches.length,
    offset,
    limit,
    source: cachedPath === 'remote' ? 'huggingface' : 'local',
    facets,
    rows: matches.slice(offset, offset + limit).map((row) => ({
      ...row,
      image_url: row.source?.relative_path
        ? `/api/qwen-image?path=${encodeURIComponent(row.source.relative_path)}`
        : null
    }))
  }
})
