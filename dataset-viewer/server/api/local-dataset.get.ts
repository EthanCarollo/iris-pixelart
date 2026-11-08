import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

type LocalImage = { path: string; caption: string; folder: string; name: string }

const datasetRoot = resolve(process.cwd(), 'datasets', 'pixelart-dataset')
let indexPromise: Promise<LocalImage[]> | null = null

async function loadIndex(): Promise<LocalImage[]> {
  if (!indexPromise) {
    indexPromise = readFile(resolve(datasetRoot, 'viewer-index.json'), 'utf8')
      .then((content) => JSON.parse(content) as LocalImage[])
      .catch((error) => {
        indexPromise = null
        throw error
      })
  }
  return indexPromise
}

export default defineEventHandler(async (event) => {
  const query = getQuery(event)
  const offset = Math.max(0, Number.parseInt(String(query.offset ?? '0'), 10) || 0)
  const length = Math.min(100, Math.max(1, Number.parseInt(String(query.length ?? '100'), 10) || 100))
  const term = typeof query.query === 'string' ? query.query.trim().toLocaleLowerCase() : ''

  try {
    const images = await loadIndex()
    const matches = term
      ? images.filter((image) => `${image.caption} ${image.folder} ${image.name} ${image.path}`.toLocaleLowerCase().includes(term))
      : images
    const features = [
      { name: 'image', type: { _type: 'Image' } },
      { name: 'caption', type: { dtype: 'string', _type: 'Value' } },
      { name: 'folder', type: { dtype: 'string', _type: 'Value' } },
      { name: 'filename', type: { dtype: 'string', _type: 'Value' } }
    ]
    return {
      dataset: 'ethanzxv/pixelart-dataset',
      source: 'local',
      features,
      rows: matches.slice(offset, offset + length).map((image, index) => ({
        row_idx: offset + index,
        row: {
          image: { src: `/api/local-dataset/image?path=${encodeURIComponent(image.path)}` },
          caption: image.caption,
          folder: image.folder,
          filename: image.name
        }
      })),
      num_rows_total: matches.length,
      num_rows_per_page: length
    }
  } catch {
    throw createError({ statusCode: 404, statusMessage: 'Le dataset local n’est pas encore téléchargé.' })
  }
})
