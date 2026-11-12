<script setup lang="ts">
type DataRow = Record<string, unknown>
type DatasetView = {
  name: string
  columns: string[]
  rows: DataRow[]
  total: number
  source: 'paged' | 'file'
  size?: number
}
type DatasetResponse = {
  features?: Array<{ name: string; type?: unknown }>
  rows?: Array<{ row: DataRow }>
  num_rows_total?: number
  status?: string
  done?: number
  total?: number
  message?: string
  errors?: number
}
type TagStats = {
  total: number
  captionGroups: Array<{ title: string; items: Array<{ key: string; label: string; count: number }>; denominator: number; multiple?: boolean }>
  annotationTotal: number
  annotationCoveragePercent: number
  annotationGroups: Array<{ title: string; items: Array<{ key: string; label: string; count: number }>; denominator: number; multiple?: boolean }>
}

const datasetId = 'ethanzxv/pixelart-dataset'
const pageSize = 100
const search = ref('')
const currentPage = ref(0)
const dataset = shallowRef<DatasetView | null>(null)
const tagStats = shallowRef<TagStats | null>(null)
const localReady = ref(false)
const syncPrompt = ref(false)
const loading = ref(false)
const downloadActive = ref(false)
const downloadDone = ref(0)
const downloadTotal = ref(0)
const downloadStage = ref('')
const status = ref('')
const statusError = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
let searchTimer: ReturnType<typeof setTimeout> | undefined
let downloadPoll: ReturnType<typeof setInterval> | undefined
let requestController: AbortController | null = null

const localFileRows = computed(() => {
  if (!dataset.value || dataset.value.source !== 'file') return []
  const term = search.value.trim().toLocaleLowerCase()
  if (!term) return dataset.value.rows
  return dataset.value.rows.filter((row) =>
    Object.values(row).some((value) => stringifyValue(value).toLocaleLowerCase().includes(term))
  )
})
const totalRows = computed(() => {
  if (!dataset.value) return 0
  return dataset.value.source === 'paged' ? dataset.value.total : localFileRows.value.length
})
const visibleRows = computed(() => {
  if (!dataset.value) return []
  if (dataset.value.source === 'paged') return dataset.value.rows
  const start = currentPage.value * pageSize
  return localFileRows.value.slice(start, start + pageSize)
})
const pageCount = computed(() => Math.max(1, Math.ceil(totalRows.value / pageSize)))
const downloadProgress = computed(() => downloadTotal.value > 0 ? Math.min(100, Math.round(downloadDone.value / downloadTotal.value * 100)) : 0)

onMounted(async () => {
  try {
    await loadDownloadedDataset(0)
    localReady.value = true
    await loadTagStats()
    try {
      const syncState = await $fetch<DatasetResponse>('/api/dataset-download-status')
      if (syncState.status === 'error') setStatus(syncState.message || 'La dernière synchronisation Hugging Face est incomplète.', true)
      else setStatus('Copie locale chargée. Les images sont servies depuis ce dossier.')
    } catch {
      setStatus('Copie locale chargée. Les images sont servies depuis ce dossier.')
    }
  } catch {
    localReady.value = false
    await resumeDownloadIfActive()
  }
})

watch([search, currentPage], () => {
  if (dataset.value?.source !== 'paged' || !localReady.value) return
  clearTimeout(searchTimer)
  requestController?.abort()
  searchTimer = setTimeout(() => loadDownloadedDataset(currentPage.value), search.value.trim() ? 250 : 0)
})
watch(search, () => { currentPage.value = 0 })

async function loadDownloadedDataset(page: number) {
  requestController?.abort()
  const controller = new AbortController()
  requestController = controller
  loading.value = true
  try {
    const response = await $fetch<DatasetResponse>('/api/local-dataset', {
      query: {
        offset: String(page * pageSize),
        length: String(pageSize),
        query: search.value.trim()
      },
      signal: controller.signal
    })
    const columns = (response.features ?? []).map((feature) => feature.name)
    const rows = (response.rows ?? []).map((entry) => entry.row)
    dataset.value = {
      name: `${datasetId} · copie locale`,
      columns,
      rows,
      total: response.num_rows_total ?? rows.length,
      source: 'paged'
    }
  } catch (error) {
    if (controller.signal.aborted) return
    throw error
  } finally {
    if (requestController === controller) loading.value = false
  }
}

async function loadTagStats() {
  try {
    tagStats.value = await $fetch<TagStats>('/api/local-dataset/stats')
  } catch {
    tagStats.value = null
  }
}

function tagPercent(count: number, denominator: number) {
  if (!denominator) return '0,0'
  return (count / denominator * 100).toLocaleString('fr-FR', { maximumFractionDigits: 1 })
}

async function startDownload(sync = false) {
  downloadActive.value = true
  downloadStage.value = 'Démarrage du téléchargement…'
  downloadDone.value = 0
  downloadTotal.value = 0
  setStatus('Synchronisation avec Hugging Face…')
  try {
    const response = await $fetch<DatasetResponse>('/api/dataset-download', {
      method: 'POST',
      query: sync ? { sync: '1' } : undefined
    })
    if (response.status === 'complete') {
      await finishDownload()
      return
    }
    applyDownloadStatus(response)
    downloadPoll = setInterval(() => { void pollDownload() }, 1500)
  } catch (error) {
    downloadActive.value = false
    setStatus(errorMessage(error), true)
  }
}

async function pollDownload() {
  try {
    const response = await $fetch<DatasetResponse>('/api/dataset-download-status')
    applyDownloadStatus(response)
    if (response.status === 'complete') {
      clearInterval(downloadPoll)
      await finishDownload()
    } else if (response.status === 'error') {
      clearInterval(downloadPoll)
      downloadActive.value = false
      setStatus(response.message || 'Le téléchargement a échoué. Vérifie le jeton et réessaie.', true)
    }
  } catch (error) {
    clearInterval(downloadPoll)
    downloadActive.value = false
    setStatus(errorMessage(error), true)
  }
}

async function resumeDownloadIfActive() {
  try {
    const response = await $fetch<DatasetResponse>('/api/dataset-download-status')
    if (response.status === 'listing' || response.status === 'starting' || response.status === 'downloading' || response.status === 'indexing') {
      downloadActive.value = true
      applyDownloadStatus(response)
      downloadPoll = setInterval(() => { void pollDownload() }, 1500)
      return
    }
    if (response.status === 'error') {
      setStatus(response.message || 'Le téléchargement a échoué. Vérifie le jeton et réessaie.', true)
      return
    }
  } catch {
    // Fall back to prompting for a token.
  }
  setStatus('Le dataset n’est pas encore téléchargé. Saisis un jeton Hugging Face avec accès au dépôt pour le récupérer localement.', true)
}

function applyDownloadStatus(response: DatasetResponse) {
  downloadStage.value = response.message || 'Téléchargement en cours…'
  downloadDone.value = response.done ?? 0
  downloadTotal.value = response.total ?? 0
  const count = downloadTotal.value ? ` ${downloadDone.value.toLocaleString('fr-FR')} / ${downloadTotal.value.toLocaleString('fr-FR')} fichiers.` : ''
  setStatus(`${downloadStage.value}${count}`)
}

async function finishDownload() {
  downloadActive.value = false
  localReady.value = true
  syncPrompt.value = false
  search.value = ''
  currentPage.value = 0
  await loadDownloadedDataset(0)
  await loadTagStats()
  setStatus('Index du dataset synchronisé et chargé.')
}

async function loadLocalFile(file?: File) {
  if (!file) return
  try {
    const text = (await file.text()).replace(/^\uFEFF/, '')
    const extension = file.name.split('.').pop()?.toLocaleLowerCase()
    const parsed = extension === 'json' || /^[\s]*[\[{]/.test(text) ? parseJson(text) : parseDelimited(text, extension === 'tsv' ? '\t' : undefined)
    if (!parsed.columns.length) throw new Error('Aucune colonne trouvée dans ce fichier.')
    dataset.value = { name: file.name, columns: parsed.columns, rows: parsed.rows, total: parsed.rows.length, source: 'file', size: file.size }
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

function onDrop(event: DragEvent) {
  const file = event.dataTransfer?.files?.[0]
  if (file) void loadLocalFile(file)
}

function formatBytes(bytes = 0) {
  if (bytes < 1024) return `${bytes} o`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toLocaleString('fr-FR', { maximumFractionDigits: 1 })} Ko`
  return `${(bytes / (1024 * 1024)).toLocaleString('fr-FR', { maximumFractionDigits: 1 })} Mo`
}

onUnmounted(() => {
  requestController?.abort()
  clearTimeout(searchTimer)
  clearInterval(downloadPoll)
})
</script>

<template>
  <main class="shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">Dataset local · Hugging Face · pixel art</p>
        <h1>Dataset Viewer</h1>
        <p class="subtitle">Les images et leurs légendes sont téléchargées sur cet ordinateur puis parcourues localement.</p>
      </div>
      <div class="header-actions"><NuxtLink class="button secondary" to="/qwen">Annotations Qwen3.8 →</NuxtLink><div class="privacy">Données du dataset stockées dans le dépôt local</div></div>
    </header>

    <section class="hf-card" aria-labelledby="hfHeading">
      <div class="hf-card-head">
        <div><h2 id="hfHeading">{{ localReady ? 'Dataset téléchargé' : 'Télécharger le dataset privé' }}</h2><p>{{ localReady ? 'La copie locale est chargée par défaut.' : 'Les métadonnées sont synchronisées au démarrage et les images chargées à la demande.' }}</p></div>
        <div class="hf-head-actions"><button v-if="localReady && !syncPrompt" class="button secondary" type="button" :disabled="downloadActive" @click="syncPrompt = true">Resynchroniser Hugging Face</button><span class="hf-brand">🤗 HF</span></div>
      </div>
      <form v-if="!localReady || syncPrompt" class="hf-controls download-controls" @submit.prevent="startDownload(syncPrompt)">
        <div class="download-actions"><button class="button" type="submit" :disabled="downloadActive">{{ downloadActive ? 'Synchronisation…' : syncPrompt ? 'Resynchroniser' : 'Télécharger localement' }}</button><button v-if="syncPrompt" class="button secondary" type="button" :disabled="downloadActive" @click="syncPrompt = false">Annuler</button></div>
      </form>
      <div class="dataset-name">{{ datasetId }}</div>
      <div v-if="status" class="hf-status" :class="{ error: statusError }" role="status">{{ status }}</div>
      <div v-if="downloadActive" class="download-progress" aria-label="Progression du téléchargement">
        <div class="progress-track"><div class="progress-fill" :style="{ width: `${downloadProgress}%` }"></div></div>
        <span>{{ downloadTotal ? `${downloadProgress}%` : 'Connexion…' }}</span>
      </div>
      <p class="hf-help">Le token est lu côté serveur depuis le fichier .env. Les images sont chargées à la demande.</p>
    </section>

    <section class="dropzone" @dragover.prevent @drop.prevent="onDrop">
      <div class="drop-copy">
        <div class="upload-icon" aria-hidden="true">↑</div>
        <div><strong>Ouvre aussi un fichier local</strong><span>CSV, TSV ou JSON · glisse-dépose ou sélectionne</span></div>
      </div>
      <input ref="fileInput" class="hidden-input" type="file" accept=".csv,.tsv,.json" @change="onFileSelected">
      <button class="button secondary" type="button" @click="fileInput?.click()">Choisir un fichier</button>
    </section>

    <section v-if="!dataset" class="empty-state">
      <div class="empty-mark" aria-hidden="true">▦</div>
      <h2>{{ downloadActive ? 'Préparation du dataset local' : 'Le dataset apparaîtra ici' }}</h2>
      <p>{{ downloadActive ? 'Les images et leurs légendes sont en cours de téléchargement puis d’indexation.' : 'Synchronise le dataset ou ouvre un fichier de données local.' }}</p>
    </section>

    <section v-else class="workspace" aria-live="polite">
      <div class="dataset-head">
        <div class="file-title"><h2>{{ dataset.name }}</h2><p>{{ dataset.source === 'paged' ? 'Fichiers sur le disque local' : `${formatBytes(dataset.size)} · fichier local` }}</p></div>
        <button v-if="dataset.source === 'file' && localReady" class="button secondary" type="button" @click="loadDownloadedDataset(0)">Revenir au dataset</button>
      </div>
      <div class="stats">
        <div class="stat"><span>Images</span><strong>{{ totalRows.toLocaleString('fr-FR') }}</strong></div>
        <div class="stat"><span>Colonnes</span><strong>{{ dataset.columns.length.toLocaleString('fr-FR') }}</strong></div>
        <div class="stat"><span>Résultats</span><strong>{{ totalRows.toLocaleString('fr-FR') }}</strong></div>
      </div>
      <section v-if="dataset.source === 'paged' && tagStats" class="tag-overview" aria-labelledby="tagOverviewTitle">
        <div class="tag-overview-head">
          <div><h2 id="tagOverviewTitle">Répartition globale du dataset</h2><p>{{ tagStats.total.toLocaleString('fr-FR') }} images au total · indépendante de la recherche et de la pagination</p></div>
          <span class="tag-total">{{ tagStats.total.toLocaleString('fr-FR') }} images</span>
        </div>
        <p class="tag-note">Le dépôt contient {{ tagStats.annotationTotal.toLocaleString('fr-FR') }} annotations structurées disponibles ({{ tagStats.annotationCoveragePercent.toLocaleString('fr-FR', { maximumFractionDigits: 1 }) }} % du dataset) ; les graphiques détaillés ci-dessous portent sur ces images annotées. Le rôle de jeu est exclu des statistiques. Le reste du dataset est résumé d’après les caractéristiques explicitement mentionnées dans les légendes, sans déduire de tags absents.</p>
        <h3 class="tag-section-title">Tags structurés des images annotées</h3>
        <div class="tag-grid">
          <article v-for="group in tagStats.annotationGroups" :key="`annotation-${group.title}`" class="tag-card">
            <h3>{{ group.title }}</h3>
            <ul class="tag-list">
              <li v-for="item in group.items" :key="item.key" class="tag-row">
                <div class="tag-row-label"><span>{{ item.label }}</span><strong>{{ item.count.toLocaleString('fr-FR') }} · {{ tagPercent(item.count, group.denominator) }} %</strong></div>
                <div class="tag-track"><span :style="{ width: `${Math.max(item.count ? 0.5 : 0, item.count / group.denominator * 100)}%` }"></span></div>
              </li>
            </ul>
          </article>
        </div>
        <p class="tag-note tag-note-bottom">Pour les couleurs et équipements, une image peut porter plusieurs tags : la somme de ces pourcentages peut dépasser 100 %. Le terme source « character » est affiché comme « humanoid » pour la forme du sujet.</p>
        <h3 class="tag-section-title">Caractéristiques explicites sur toutes les images</h3>
        <div class="tag-grid">
          <article v-for="group in tagStats.captionGroups" :key="`caption-${group.title}`" class="tag-card">
            <h3>{{ group.title }}</h3>
            <ul class="tag-list">
              <li v-for="item in group.items" :key="item.key" class="tag-row">
                <div class="tag-row-label"><span>{{ item.label }}</span><strong>{{ item.count.toLocaleString('fr-FR') }} · {{ tagPercent(item.count, group.denominator) }} %</strong></div>
                <div class="tag-track"><span :style="{ width: `${Math.max(item.count ? 0.5 : 0, item.count / group.denominator * 100)}%` }"></span></div>
              </li>
            </ul>
          </article>
        </div>
      </section>
      <div class="table-card">
        <div class="table-toolbar">
          <label class="search-wrap"><input v-model="search" class="search" type="search" placeholder="Rechercher dans les légendes et noms…" autocomplete="off"></label>
          <span class="result-count">{{ loading ? 'Chargement…' : `${totalRows.toLocaleString('fr-FR')} résultat${totalRows === 1 ? '' : 's'}` }}</span>
        </div>
        <div class="table-scroll">
          <table>
            <thead><tr><th v-for="column in dataset.columns" :key="column">{{ column }}</th></tr></thead>
            <tbody>
              <tr v-for="(row, rowIndex) in visibleRows" :key="rowIndex">
                <td v-for="column in dataset.columns" :key="column">
                  <img v-if="isImageValue(row[column])" :src="imageSource(row[column])" alt="Pixel art" loading="lazy" class="cell-image">
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
    <footer>Dataset Viewer · la copie Hugging Face est lue depuis le disque local</footer>
  </main>
</template>
