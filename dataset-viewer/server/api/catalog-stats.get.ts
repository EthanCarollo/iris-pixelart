import { loadCatalog } from '../utils/catalog'

export default defineEventHandler(async () => {
  const { images, annotations } = await loadCatalog()
  const existing = new Set(images.map(image => image.path))
  const counts: Record<string, Record<string, number>> = {}
  const colors: Record<string, number> = {}
  const folders: Record<string, number> = {}
  let annotated = 0, measured = 0
  for (const image of images) folders[image.folder] = (folders[image.folder] || 0) + 1
  for (const [path, row] of annotations) {
    if (!existing.has(path)) continue
    annotated++
    for (const [field, raw] of Object.entries(row.labels || {})) {
      if (!counts[field]) counts[field] = {}
      const values = Array.isArray(raw) ? [...new Set(raw.map(String))] : [String(raw ?? 'unknown')]
      for (const value of values) counts[field]![value] = (counts[field]![value] || 0) + 1
    }
    if (row.color_measurements?.status === 'measured') {
      measured++
      const distinct = new Set<string>((row.color_measurements.palette_hex || []).map((entry: any) => String(entry.hex)).filter(Boolean))
      for (const hex of distinct) colors[hex] = (colors[hex] || 0) + 1
    }
  }
  const sorted = (counts: Record<string, number>, max = 20) => Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, max).map(([label, count]) => ({ label, count }))
  return { total: images.length, annotated, measured, sources: sorted(folders), colors: sorted(colors, 24), groups: Object.entries(counts).map(([field, values]) => ({ field, items: sorted(values, 20) })) }
})
