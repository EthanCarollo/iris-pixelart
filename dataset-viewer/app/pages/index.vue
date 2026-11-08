<script setup lang="ts">
type DataRow = Record<string, unknown>
type DatasetView = {
  name: string
  columns: string[]
  rows: DataRow[]
  total: number
  source: 'local' | 'remote'
  size?: number
}
type HuggingFaceSplit = { config: string; split: string }
type HuggingFaceResponse = {
  splits?: HuggingFaceSplit[]
  features?: Array<{ name: string; type?: unknown }>
  rows?: Array<{ row: DataRow }>
  num_rows_total?: number
  error?: string
}

const datasetId = 'ethanzxv/pixelart-dataset'
const pageSize = 100
const tokenInput = ref('')
const sessionToken = ref('')
const configs = ref<string[]>([])
const splits = ref<string[]>([])
const selectedConfig = ref('')
const selectedSplit = ref('')
const connected = ref(false)
const search = ref('')
const currentPage = ref(0)
const dataset = shallowRef<DatasetView | null>(null)
const status = ref('')
const statusError = ref(false)
const loading = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
let requestController: AbortController | null = null
let searchTimer: ReturnType<typeof setTimeout> | undefined

onMounted(() => {
  void connectHuggingFace(true)
})

const localRows = computed(() => {
  if (!dataset.value || dataset.value.source !== 'local') return []
  const term = search.value.trim().toLocaleLowerCase()
  if (!term) return dataset.value.rows
  return dataset.value.rows.filter((row) =>
    Object.values(row).some((value) => stringifyValue(value).toLocaleLowerCase().includes(term))
  )
})
const totalRows = computed(() => {
  if (!dataset.value) return 0
  return dataset.value.source === 'remote' ? dataset.value.total : localRows.value.length
})
const visibleRows = computed(() => {
  if (!dataset.value) return []
  if (dataset.value.source === 'remote') return dataset.value.rows
  const start = currentPage.value * pageSize
  return localRows.value.slice(start, start + pageSize)
})
const pageCount = computed(() => Math.max(1, Math.ceil(totalRows.value / pageSize)))

watch([connected, selectedConfig, selectedSplit, search, currentPage], () => {
  if (!connected.value) return
  clearTimeout(searchTimer)
  requestController?.abort()
  searchTimer = setTimeout(() => loadRemotePage(), search.value.trim() ? 350 : 0)
})

watch(search, () => { currentPage.value = 0 })

async function requestHuggingFace(endpoint: string, query: Record<string, string> = {}) {
  return await $fetch<HuggingFaceResponse>(`/api/huggingface/${endpoint}`, {
    headers: sessionToken.value ? { Authorization: `Bearer ${sessionToken.value}` } : undefined,
    query
  })
}

async function connectHuggingFace(automatic = false) {
  setStatus(automatic ? 'Chargement automatique du dataset…' : 'Connexion au dataset Hugging Face…')
  connected.value = false
  const enteredToken = tokenInput.value.trim()
  sessionToken.value = enteredToken
  try {
    const data = await requestHuggingFace('splits')
    const entries = data.splits ?? []
    if (!entries.length) throw new Error('Aucune configuration ou aucun split disponible.')
    datasetSplits.value = entries
    configs.value = [...new Set(entries.map((entry) => entry.config))]
    selectedConfig.value = configs.value[0] ?? ''
    splits.value = [...new Set(entries.filter((entry) => entry.config === selectedConfig.value).map((entry) => entry.split))]
    selectedSplit.value = splits.value[0] ?? ''
    connected.value = true
    tokenInput.value = ''
    currentPage.value = 0
    search.value = ''
    setStatus('Connexion réussie. Chargement du dataset…')
  } catch (error) {
    sessionToken.value = ''
    if (!enteredToken && isAuthError(error)) {
      setStatus('Ce dataset demande une authentification Hugging Face. Saisis un jeton avec accès en lecture, puis clique sur « Charger le dataset ».', true)
    } else {
      setStatus(errorMessage(error), true)
    }
  }
}

function updateSplitsForConfig() {
  splits.value = [...new Set(
    (datasetSplits.value ?? []).filter((entry) => entry.config === selectedConfig.value).map((entry) => entry.split)
  )]
  selectedSplit.value = splits.value[0] ?? ''
  currentPage.value = 0
}

const datasetSplits = ref<HuggingFaceSplit[]>([])

async function loadRemotePage() {
  if (!connected.value || !selectedConfig.value || !selectedSplit.value) return
  requestController?.abort()
  const controller = new AbortController()
  requestController = controller
  loading.value = true
  setStatus('Chargement des lignes…')
  try {
    const endpoint = search.value.trim() ? 'search' : 'rows'
    const query: Record<string, string> = {
      config: selectedConfig.value,
      split: selectedSplit.value,
      offset: String(currentPage.value * pageSize),
      length: String(pageSize)
    }
    if (search.value.trim()) query.query = search.value.trim()
    const response = await $fetch<HuggingFaceResponse>(`/api/huggingface/${endpoint}`, {
      headers: sessionToken.value ? { Authorization: `Bearer ${sessionToken.value}` } : undefined,
      query,
      signal: controller.signal
    })
    const columns = (response.features ?? []).map((feature) => feature.name)
    const rows = (response.rows ?? []).map((entry) => entry.row)
    dataset.value = {
      name: `${datasetId} · ${selectedConfig.value} / ${selectedSplit.value}`,
      columns,
      rows,
      total: response.num_rows_total ?? rows.length,
      source: 'remote'
    }
    setStatus('Dataset chargé depuis Hugging Face.')
  } catch (error) {
    if (controller.signal.aborted || (error instanceof Error && error.name === 'AbortError')) return
    setStatus(errorMessage(error), true)
  } finally {
    if (requestController === controller) loading.value = false
  }
}

async function loadLocalFile(file?: File) {
  if (!file) return
  try {
    const text = (await file.text()).replace(/^\uFEFF/, '')
    const extension = file.name.split('.').pop()?.toLocaleLowerCase()
    const parsed = extension === 'json' || /^[\s]*[\[{]/.test(text) ? parseJson(text) : parseDelimited(text, extension === 'tsv' ? '\t' : undefined)
    if (!parsed.columns.length) throw new Error('Aucune colonne trouvée dans ce fichier.')
    dataset.value = { name: file.name, columns: parsed.columns, rows: parsed.rows, total: parsed.rows.length, source: 'local', size: file.size }
    connected.value = false
    sessionToken.value = ''
    currentPage.value = 0
    search.value = ''
    setStatus('Fichier local chargé. Les données restent dans ce navigateur.')
  } catch (error) {
    setStatus(errorMessage(error), true)
  }
}

function parseJson(text: string): { columns: string[]; rows: DataRow[] } {
  let value: unknown
  try { value = JSON.parse(text) } catch { throw new Error('Le JSON est invalide. Vérifie la syntaxe du fichier.') }
  if (!Array.isArray(value)) {
    if (value && typeof value === 'object' && Array.isArray((value as { data?: unknown }).data)) value = (value as { data: unknown[] }).data
    else if (value && typeof value === 'object' && Array.isArray((value as { rows?: unknown }).rows)) value = (value as { rows: unknown[] }).rows
    else if (value && typeof value === 'object') value = [value]
    else throw new Error('Le JSON doit contenir un tableau d’objets.')
  }
  const records = value as unknown[]
  const columns = [...new Set(records.flatMap((row) => row && typeof row === 'object' && !Array.isArray(row) ? Object.keys(row) : []))]
  if (!columns.length) return { columns: [], rows: [] }
  return {
    columns,
    rows: records.map((row) => Object.fromEntries(columns.map((column) => [column, (row as DataRow)?.[column] ?? ''])) as DataRow)
  }
}

function parseDelimited(text: string, forcedDelimiter?: string): { columns: string[]; rows: DataRow[] } {
  const firstLine = text.split(/\r?\n/, 1)[0] ?? ''
  const delimiter = forcedDelimiter ?? [',', ';', '\t'].map((candidate) => [candidate, countOutsideQuotes(firstLine, candidate)] as const).sort((a, b) => b[1] - a[1])[0]?.[0] ?? ','
  const records: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  for (let index = 0; index < text.length; index++) {
    const char = text[index]
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') { field += '"'; index++ }
      else if (char === '"') quoted = false
      else field += char
    } else if (char === '"' && field.length === 0) quoted = true
    else if (char === delimiter) { row.push(field); field = '' }
    else if (char === '\n' || char === '\r') {
      if (char === '\r' && text[index + 1] === '\n') index++
      row.push(field); records.push(row); row = []; field = ''
    } else field += char
  }
  if (field.length || row.length) { row.push(field); records.push(row) }
  while (records.length && records[records.length - 1].every((cell) => !cell.trim())) records.pop()
  if (!records.length) return { columns: [], rows: [] }
  const columns = records.shift()!.map((name, index) => name.trim() || `Colonne ${index + 1}`)
  const rows = records.map((cells) => Object.fromEntries(columns.map((column, index) => [column, cells[index] ?? ''])))
  return { columns, rows }
}

function countOutsideQuotes(line: string, delimiter: string) {
  let count = 0
  let quoted = false
  for (let index = 0; index < line.length; index++) {
    if (line[index] === '"' && line[index + 1] === '"' && quoted) index++
    else if (line[index] === '"') quoted = !quoted
    else if (line[index] === delimiter && !quoted) count++
  }
  return count
}

function stringifyValue(value: unknown): string {
  if (value === null || value === undefined) return ''
  return typeof value === 'object' ? JSON.stringify(value) : String(value)
}

function isImageValue(value: unknown): value is { src?: string; url?: string } {
  return Boolean(value && typeof value === 'object' && ('src' in value || 'url' in value))
}

function imageSource(value: unknown) {
  if (!isImageValue(value)) return ''
  return value.src || value.url || ''
}

function onFileSelected(event: Event) {
  const input = event.target as HTMLInputElement
  void loadLocalFile(input.files?.[0])
  input.value = ''
}

function setStatus(message: string, isError = false) {
  status.value = message
  statusError.value = isError
}

function errorMessage(error: unknown) {
  const value = error as { data?: { statusMessage?: string; message?: string }; statusMessage?: string; message?: string }
  return value?.data?.statusMessage || value?.statusMessage || value?.data?.message || value?.message || 'Impossible de charger les données.'
}

function isAuthError(error: unknown) {
  const value = error as { status?: number; statusCode?: number; response?: { status?: number } }
  const status = value?.statusCode ?? value?.status ?? value?.response?.status
  return status === 401 || status === 403
}

function closeDataset() {
  requestController?.abort()
  dataset.value = null
  connected.value = false
  sessionToken.value = ''
  configs.value = []
  splits.value = []
  datasetSplits.value = []
  selectedConfig.value = ''
  selectedSplit.value = ''
  search.value = ''
  currentPage.value = 0
  setStatus('')
}

function onDrop(event: DragEvent) {
  const file = event.dataTransfer?.files?.[0]
  if (file) void loadLocalFile(file)
}

function formatBytes(bytes = 0) {
  if (bytes < 1024) return `${bytes} o`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toLocaleString('fr-FR', { maximumFractionDigits: 1 })} Ko`
  return `${(bytes / (1024 * 1024)).toLocaleString('fr-FR', { maximumFractionDigits: 1 })} Mo`
}

watch(selectedConfig, () => {
  if (connected.value && datasetSplits.value.length) updateSplitsForConfig()
})
</script>

<template>
  <main class="shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">Outil de données · Hugging Face · CSV · TSV · JSON</p>
        <h1>Dataset Viewer</h1>
        <p class="subtitle">Parcours le dataset pixel art et tes fichiers dans une vue claire et filtrable.</p>
      </div>
      <div class="privacy">Le jeton reste en mémoire pendant la session</div>
    </header>

    <section class="hf-card" aria-labelledby="hfHeading">
      <div class="hf-card-head">
        <div><h2 id="hfHeading">Dataset Hugging Face</h2><p>Connecte le dataset distant à ce viewer.</p></div>
        <span class="hf-brand">🤗 HF</span>
      </div>
      <form class="hf-controls" @submit.prevent="connectHuggingFace">
        <div class="field token-field">
          <label for="hfToken">Jeton d'accès</label>
          <input id="hfToken" v-model="tokenInput" type="password" autocomplete="off" placeholder="Jeton avec accès en lecture">
        </div>
        <div class="field">
          <label for="hfConfig">Configuration</label>
          <select id="hfConfig" v-model="selectedConfig" :disabled="!configs.length">
            <option v-for="config in configs" :key="config" :value="config">{{ config }}</option>
          </select>
        </div>
        <div class="field">
          <label for="hfSplit">Split</label>
          <select id="hfSplit" v-model="selectedSplit" :disabled="!splits.length">
            <option v-for="split in splits" :key="split" :value="split">{{ split }}</option>
          </select>
        </div>
        <button class="button" type="submit">{{ connected ? 'Reconnecter' : 'Charger le dataset' }}</button>
      </form>
      <div class="dataset-name">{{ datasetId }}</div>
      <div v-if="status" class="hf-status" :class="{ error: statusError }" role="status">{{ status }}</div>
      <p class="hf-help">Le jeton n'est jamais enregistré dans le dépôt ni dans le stockage du navigateur. Les requêtes passent par le serveur Nuxt vers l'API Hugging Face.</p>
    </section>

    <section class="dropzone" @dragover.prevent @drop.prevent="onDrop">
      <div class="drop-copy">
        <div class="upload-icon" aria-hidden="true">↑</div>
        <div><strong>Ouvre un fichier local</strong><span>CSV, TSV ou JSON · glisse-dépose ou sélectionne</span></div>
      </div>
      <input ref="fileInput" class="hidden-input" type="file" accept=".csv,.tsv,.json" @change="onFileSelected">
      <button class="button secondary" type="button" @click="fileInput?.click()">Choisir un fichier</button>
    </section>

    <section v-if="!dataset" class="empty-state">
      <div class="empty-mark" aria-hidden="true">▦</div>
      <h2>Ton aperçu apparaîtra ici</h2>
      <p>Connecte Hugging Face pour consulter le dataset pixel art ou ouvre un fichier local pour voir ses lignes et ses colonnes.</p>
    </section>

    <section v-else class="workspace" aria-live="polite">
      <div class="dataset-head">
        <div class="file-title"><h2>{{ dataset.name }}</h2><p>{{ dataset.source === 'remote' ? 'Hugging Face' : `${formatBytes(dataset.size)} · fichier local` }}</p></div>
        <button class="button secondary" type="button" @click="closeDataset">Fermer le dataset</button>
      </div>
      <div class="stats">
        <div class="stat"><span>Lignes</span><strong>{{ totalRows.toLocaleString('fr-FR') }}</strong></div>
        <div class="stat"><span>Colonnes</span><strong>{{ dataset.columns.length.toLocaleString('fr-FR') }}</strong></div>
        <div class="stat"><span>Résultats</span><strong>{{ totalRows.toLocaleString('fr-FR') }}</strong></div>
      </div>
      <div class="table-card">
        <div class="table-toolbar">
          <label class="search-wrap"><input v-model="search" class="search" type="search" placeholder="Rechercher dans les données…" autocomplete="off"></label>
          <span class="result-count">{{ loading ? 'Chargement…' : `${totalRows.toLocaleString('fr-FR')} résultat${totalRows === 1 ? '' : 's'}` }}</span>
        </div>
        <div class="table-scroll">
          <table>
            <thead><tr><th v-for="column in dataset.columns" :key="column">{{ column }}</th></tr></thead>
            <tbody>
              <tr v-for="(row, rowIndex) in visibleRows" :key="rowIndex">
                <td v-for="column in dataset.columns" :key="column">
                  <img v-if="isImageValue(row[column])" :src="imageSource(row[column])" alt="Aperçu du visuel" loading="lazy" class="cell-image">
                  <span v-else :title="stringifyValue(row[column])">{{ stringifyValue(row[column]) }}</span>
                </td>
              </tr>
              <tr v-if="!visibleRows.length"><td class="no-results" :colspan="Math.max(dataset.columns.length, 1)">{{ loading ? 'Chargement des lignes…' : 'Aucune ligne ne correspond.' }}</td></tr>
            </tbody>
          </table>
        </div>
        <div class="pagination">
          <span class="page-info">{{ totalRows ? `Lignes ${currentPage * pageSize + 1}–${Math.min((currentPage + 1) * pageSize, totalRows)} sur ${totalRows.toLocaleString('fr-FR')}` : 'Aucun résultat' }}</span>
          <div class="page-actions">
            <button type="button" :disabled="currentPage === 0 || loading" @click="currentPage--">← Précédent</button>
            <button type="button" :disabled="currentPage >= pageCount - 1 || loading" @click="currentPage++">Suivant →</button>
          </div>
        </div>
      </div>
    </section>
    <footer>Dataset Viewer · les fichiers locaux restent dans ton navigateur</footer>
  </main>
</template>
