import { spawn } from 'node:child_process'
import { access, readFile, mkdir, writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'

const projectRoot = process.cwd()
const datasetRoot = resolve(projectRoot, 'datasets', 'pixelart-dataset')
const statusPath = resolve(datasetRoot, '.viewer-download-status.json')
const indexPath = resolve(datasetRoot, 'viewer-index.json')

async function startDownloader(token: string) {
  const candidates = process.platform === 'win32'
    ? [['py', '-3'], ['python'], ['python3']]
    : [['python3'], ['python']]

  for (const [command, ...args] of candidates) {
    const child = spawn(command!, [...args, resolve(projectRoot, 'scripts', 'download_dataset.py'), '--metadata-only'], {
      cwd: projectRoot,
      env: { ...process.env, HF_ACCESS_TOKEN: token },
      detached: true,
      stdio: 'ignore',
      windowsHide: true
    })
    const error = await new Promise<NodeJS.ErrnoException | null>((resolveStart) => {
      child.once('spawn', () => resolveStart(null))
      child.once('error', resolveStart)
    })
    if (!error) {
      child.unref()
      return
    }
    if (error.code !== 'ENOENT') throw error
  }

  throw new Error('Python 3 est requis pour télécharger le dataset. Installe-le et vérifie qu’il est accessible dans le PATH.')
}

export default defineEventHandler(async (event) => {
  const syncRequested = getQuery(event).sync === '1'
  const token = process.env.HF_TOKEN?.trim() || ''
  if (!token) throw createError({ statusCode: 503, statusMessage: 'HF_TOKEN absent du fichier .env serveur.' })

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
    if (['starting', 'cloning', 'listing', 'downloading', 'indexing'].includes(status.status || '')) return status
  } catch {
    // No download is active.
  }

  await mkdir(datasetRoot, { recursive: true })
  await writeFile(statusPath, JSON.stringify({ status: 'starting', done: 0, total: 0, message: 'Démarrage du téléchargement' }), 'utf8')

  try {
    await startDownloader(token)
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Impossible de démarrer le téléchargeur.'
    await writeFile(statusPath, JSON.stringify({ status: 'error', done: 0, total: 0, message }), 'utf8')
    return { status: 'error', done: 0, total: 0, message }
  }

  return { status: 'listing', done: 0, total: 0, message: 'Lecture du manifeste Hugging Face' }
})
