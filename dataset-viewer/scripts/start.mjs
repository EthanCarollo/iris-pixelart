import { readFileSync } from 'node:fs'
import { parseEnv } from 'node:util'
import { spawnSync, spawn } from 'node:child_process'

let envFile
try { envFile = parseEnv(readFileSync('.env', 'utf8')) } catch {
  console.error('Fichier dataset-viewer/.env introuvable ou invalide. Ajoute HF_TOKEN dans ce fichier.')
  process.exit(1)
}
const token = envFile.HF_TOKEN?.trim()
if (!token) {
  console.error('HF_TOKEN est obligatoire dans dataset-viewer/.env.')
  process.exit(1)
}
process.env.HF_TOKEN = token
process.env.HF_ACCESS_TOKEN = token
const repo = 'ethanzxv/pixelart-dataset'
let response
try {
  response = await fetch(`https://huggingface.co/api/datasets/${repo}`, {
    headers: { Authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(20000)
  })
} catch {
  console.error('Connexion à Hugging Face impossible : vérifie le réseau.')
  process.exit(1)
}
if (!response.ok) {
  console.error(`Accès au dataset Hugging Face refusé ou indisponible (HTTP ${response.status}).`)
  process.exit(1)
}
console.log('Synchronisation de l’index et des métadonnées Hugging Face…')
const sync = spawnSync(process.platform === 'win32' ? 'python' : 'python3', ['scripts/download_dataset.py', '--metadata-only'], { stdio: 'inherit', env: process.env })
if (sync.status !== 0) {
  console.error('Échec de la synchronisation du dataset. Le viewer ne sera pas lancé.')
  process.exit(sync.status || 1)
}
const mode = process.argv[2] === 'preview' ? 'preview' : 'dev'
const child = spawn('node', ['node_modules/nuxt/bin/nuxt.mjs', mode, ...process.argv.slice(3)], { stdio: 'inherit', env: process.env })
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal))
child.on('exit', (code, signal) => process.exit(code ?? (signal ? 1 : 0)))
child.on('error', () => { console.error('Impossible de démarrer Nuxt.'); process.exit(1) })
