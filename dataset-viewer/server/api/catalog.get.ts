import { loadCatalog } from '../utils/catalog'
import { findSourceMetadata } from '../utils/source-metadata'

export default defineEventHandler(async event => {
  const { images, annotations } = await loadCatalog()
  const query = getQuery(event)
  const term = String(query.search || '').trim().toLocaleLowerCase()
  const annotated = query.annotated === '1'
  const filtered = images.filter(image => (!annotated || annotations.has(image.path)) && (!term || `${image.name} ${image.path} ${image.caption}`.toLocaleLowerCase().includes(term)))
  const offset = Math.max(0, Number.parseInt(String(query.offset || '0'), 10) || 0)
  const limit = Math.min(100, Math.max(1, Number.parseInt(String(query.limit || '24'), 10) || 24))
  const detail = query.detail === '1'
  const page = filtered.slice(offset, offset + limit)
  const metadata = detail ? await Promise.all(page.map(image => findSourceMetadata(image.path))) : []
  return {
    total: images.length, filtered: filtered.length, offset, limit,
    rows: page.map((image, index) => ({
      ...image, position: offset + index, annotated: annotations.has(image.path),
      image_url: `/api/local-dataset/image?path=${encodeURIComponent(image.path)}`,
      ...(detail ? { annotation: annotations.get(image.path) || null, metadata: metadata[index] || null } : {})
    }))
  }
})
