import { createReadStream } from 'node:fs'
import { access } from 'node:fs/promises'
import { resolve, sep } from 'node:path'

const datasetRoot = resolve(process.cwd(), 'datasets', 'pixelart-dataset')

export default defineEventHandler(async (event) => {
  const imagePath = getQuery(event).path
  if (typeof imagePath !== 'string' || !imagePath.startsWith('data/') || !imagePath.toLowerCase().endsWith('.png')) {
    throw createError({ statusCode: 400, statusMessage: 'Chemin d’image invalide.' })
  }

  const filePath = resolve(datasetRoot, imagePath)
  if (!filePath.startsWith(`${datasetRoot}${sep}`)) {
    throw createError({ statusCode: 400, statusMessage: 'Chemin d’image invalide.' })
  }
  try {
    await access(filePath)
  } catch {
    throw createError({ statusCode: 404, statusMessage: 'Image locale introuvable.' })
  }

  setResponseHeader(event, 'Content-Type', 'image/png')
  setResponseHeader(event, 'Cache-Control', 'private, max-age=3600')
  return sendStream(event, createReadStream(filePath))
})
