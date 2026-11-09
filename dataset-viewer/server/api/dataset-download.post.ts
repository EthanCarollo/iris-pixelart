import { spawn } from 'node:child_process'
import { access, readFile, mkdir, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const projectRoot = process.cwd()
const datasetRoot = resolve(projectRoot, 'datasets', 'pixelart-dataset')
const statusPath = resolve(datasetRoot, '.viewer-download-status.json')
const indexPath = resolve(datasetRoot, 'viewer-index.json')

export default defineEventHandler(async (event) => {
  const syncRequested = getQuery(event).sync === '1'
  const authorization = getHeader(event, 'authorization')
  const token = authorization?.startsWith('Bearer ') ? authorization.slice(7).trim() : ''
  if (!token) throw createError({ statusCode: 401, statusMessage: 'Saisis un jeton Hugging Face.' })

  if (!syncRequested) {
    try {
      await access(indexPath)
      return { status: 'complete', message: 'Le dataset est déjà téléchargé.' }
    } catch {
      // Start a local download below.
    }
  }

  try {
    const status = JSON.parse(await readFile(statusPath, 'utf8')) as { status?: string }
    if (status.status === 'listing' || status.status === 'downloading' || status.status === 'indexing') return status
  } catch {
    // No download is active.
  }

  await mkdir(datasetRoot, { recursive: true })
  await writeFile(statusPath, JSON.stringify({ status: 'starting', done: 0, total: 0, message: 'Démarrage du téléchargement' }), 'utf8')

  const child = spawn('python', [resolve(projectRoot, 'scripts', 'download_dataset.py')], {
    cwd: projectRoot,
    env: { ...process.env, HF_ACCESS_TOKEN: token },
    detached: true,
    stdio: 'ignore',
    windowsHide: true
  })
  child.once('error', async () => {
    await writeFile(statusPath, JSON.stringify({ status: 'error', done: 0, total: 0, message: 'Python est requis pour télécharger le dataset.' }), 'utf8')
  })
  child.unref()

  return { status: 'listing', done: 0, total: 0, message: 'Lecture du manifeste Hugging Face' }
})
