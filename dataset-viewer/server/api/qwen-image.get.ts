import { open } from 'node:fs/promises'
import { resolve, sep } from 'node:path'

const repoId = 'ethanzxv/pixelart-dataset'

function safeRelativePath(value: string) {
  const normalized = value.replaceAll('\\', '/')
  if (!normalized || normalized.startsWith('/') || normalized.split('/').some((part) => !part || part === '.' || part === '..') || !normalized.toLowerCase().endsWith('.png')) {
    throw createError({ statusCode: 400, statusMessage: 'Chemin d’image Qwen invalide.' })
  }
  return normalized
}

export default defineEventHandler(async (event) => {
  const rawPath = getQuery(event).path
  if (typeof rawPath !== 'string') throw createError({ statusCode: 400, statusMessage: 'Chemin d’image absent.' })
  const relativePath = safeRelativePath(rawPath)
  const roots = [
    resolve(process.cwd(), 'datasets/pixelart-dataset/.gitrepo/data'),
    resolve(process.cwd(), '../dataset-viewer/datasets/pixelart-dataset/.gitrepo/data'),
    resolve(process.cwd(), '../data')
  ]

  for (const root of roots) {
    const file = resolve(root, relativePath)
    if (!file.startsWith(`${root}${sep}`)) continue
    try {
      const handle = await open(file, 'r')
      let valid = false
      try {
        const signature = Buffer.alloc(8)
        const { bytesRead } = await handle.read(signature, 0, 8, 0)
        valid = bytesRead === 8 && signature.equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
      } finally { await handle.close() }
      if (!valid) continue
      setResponseHeader(event, 'Content-Type', 'image/png')
      setResponseHeader(event, 'Cache-Control', 'private, max-age=86400')
      return sendStream(event, createReadStream(file))
    } catch {
      // Try the next local cache location.
    }
  }

  const token = process.env.HF_TOKEN || process.env.HUGGINGFACE_HUB_TOKEN || process.env.HF_ACCESS_TOKEN
  if (!token) throw createError({ statusCode: 503, statusMessage: 'Image absente du cache local et aucun HF_TOKEN configuré côté serveur.' })
  const url = `https://huggingface.co/datasets/${repoId}/resolve/main/data/${relativePath.split('/').map(encodeURIComponent).join('/')}`
  const response = await fetch(url, { headers: { Authorization: `Bearer ${token}` }, redirect: 'follow' })
  if (!response.ok) throw createError({ statusCode: response.status === 404 ? 404 : 502, statusMessage: 'Hugging Face ne peut pas fournir cette image source.' })
  setResponseHeader(event, 'Content-Type', response.headers.get('content-type') || 'image/png')
  setResponseHeader(event, 'Cache-Control', 'private, max-age=86400')
  return Buffer.from(await response.arrayBuffer())
})

import { createReadStream } from 'node:fs'
