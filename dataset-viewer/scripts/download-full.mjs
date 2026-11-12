import { readFileSync } from 'node:fs'
import { parseEnv } from 'node:util'
import { spawn } from 'node:child_process'

let token
try { token = parseEnv(readFileSync('.env', 'utf8')).HF_TOKEN?.trim() } catch {}
if (!token) {
  console.error('HF_TOKEN est obligatoire dans dataset-viewer/.env.')
  process.exit(1)
}
const child = spawn('python3', ['scripts/download_dataset.py'], {
  stdio: 'inherit', env: { ...process.env, HF_ACCESS_TOKEN: token }
})
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => child.kill(signal))
child.on('exit', (code, signal) => process.exit(code ?? (signal ? 1 : 0)))
child.on('error', () => { console.error('Python 3 est requis.'); process.exit(1) })
