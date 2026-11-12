<script setup lang="ts">
type Annotation = {
  asset_id?: string
  source?: { relative_path?: string; root?: string }
  labels?: Record<string, unknown>
  caption_candidate?: string
  need_rework?: boolean | null
  review_status?: string
  model?: string
  annotated_at_utc?: string
  image_url?: string | null
}
type AnnotationResponse = {
  total: number
  filtered: number
  offset: number
  limit: number
  source: 'local' | 'huggingface'
  facets: { asset_type: string[]; subject_type: string[]; visual_quality: string[]; need_rework: string[] }
  rows: Annotation[]
}

const pageSize = 24
const search = ref('')
const assetType = ref('')
const subjectType = ref('')
const quality = ref('')
const rework = ref('')
const offset = ref(0)
const result = shallowRef<AnnotationResponse | null>(null)
const loading = ref(false)
const error = ref('')
let searchTimer: ReturnType<typeof setTimeout> | undefined

const page = computed(() => result.value ? Math.floor(result.value.offset / pageSize) + 1 : 1)
const pages = computed(() => result.value ? Math.max(1, Math.ceil(result.value.filtered / pageSize)) : 1)
const visibleStart = computed(() => result.value?.filtered ? offset.value + 1 : 0)
const visibleEnd = computed(() => Math.min(offset.value + pageSize, result.value?.filtered ?? 0))

async function load() {
  loading.value = true
  error.value = ''
  try {
    result.value = await $fetch<AnnotationResponse>('/api/qwen-annotations', {
      query: {
        offset: String(offset.value), limit: String(pageSize), search: search.value.trim(),
        asset_type: assetType.value, subject_type: subjectType.value,
        quality: quality.value, rework: rework.value
      }
    })
  } catch (cause) {
    const value = cause as { data?: { statusMessage?: string }; message?: string }
    error.value = value.data?.statusMessage || value.message || 'Impossible de charger les annotations.'
  } finally {
    loading.value = false
  }
}

function resetAndLoad() {
  offset.value = 0
  void load()
}
function pageBy(delta: number) {
  offset.value = Math.max(0, Math.min((pages.value - 1) * pageSize, offset.value + delta * pageSize))
  void load()
}
function displayValue(value: unknown) {
  if (value === null || value === undefined || value === '' || value === 'not_applicable') return '—'
  return Array.isArray(value) ? value.join(', ') : String(value)
}
function prettify(value: string) {
  return value.replaceAll('_', ' ')
}
function labelTitle(key: string) {
  const titles: Record<string, string> = {
    asset_type: 'Type d’asset', subject_type: 'Type de sujet', view: 'Vue', pose: 'Pose', action: 'Action',
    background_type: 'Fond', render_family: 'Rendu', body_plan: 'Morphologie', wings: 'Ailes',
    background_scene: 'Scène de fond', visual_quality: 'Qualité visuelle',
    dominant_color_families: 'Couleurs principales', palette_size: 'Taille de palette',
    palette_temperature: 'Température', palette_structure: 'Structure de palette', palette_contrast: 'Contraste'
  }
  return titles[key] || prettify(key)
}

onMounted(() => { void load() })
watch([assetType, subjectType, quality, rework], resetAndLoad)
watch(search, () => {
  clearTimeout(searchTimer)
  searchTimer = setTimeout(resetAndLoad, 250)
})
onUnmounted(() => clearTimeout(searchTimer))
</script>

<template>
  <main class="shell qwen-shell">
    <header class="page-header">
      <div>
        <p class="eyebrow">Dataset Viewer · annotations automatiques</p>
        <h1>Annotations Qwen3.8</h1>
        <p class="subtitle">Parcours les images sources, captions et taxonomies produites par Qwen3.8 Gyro-S.</p>
      </div>
      <NuxtLink class="button secondary qwen-back" to="/">← Dataset Viewer</NuxtLink>
    </header>

    <section class="qwen-controls" aria-label="Recherche et filtres">
      <label class="qwen-search"><span>Recherche</span><input v-model="search" type="search" placeholder="ID, jeu, caption, valeur de taxonomie…"></label>
      <label><span>Type d’asset</span><select v-model="assetType"><option value="">Tous</option><option v-for="value in result?.facets.asset_type || []" :key="value" :value="value">{{ prettify(value) }}</option></select></label>
      <label><span>Type de sujet</span><select v-model="subjectType"><option value="">Tous</option><option v-for="value in result?.facets.subject_type || []" :key="value" :value="value">{{ prettify(value) }}</option></select></label>
      <label><span>Qualité</span><select v-model="quality"><option value="">Toutes</option><option v-for="value in result?.facets.visual_quality || []" :key="value" :value="value">{{ prettify(value) }}</option></select></label>
      <label><span>Rework</span><select v-model="rework"><option value="">Tous</option><option value="false">À garder</option><option value="true">À retravailler</option><option value="unknown">Non renseigné</option></select></label>
    </section>

    <div class="qwen-summary" aria-live="polite">
      <span>{{ result?.total.toLocaleString('fr-FR') || '…' }} annotations Qwen3.8</span>
      <span>{{ result?.filtered.toLocaleString('fr-FR') || 0 }} résultat{{ result?.filtered === 1 ? '' : 's' }}</span>
      <span v-if="result">Source {{ result.source === 'local' ? 'locale' : 'Hugging Face' }}</span>
      <span v-if="loading">Chargement…</span>
    </div>

    <section v-if="error" class="qwen-error" role="alert">
      <h2>Annotations indisponibles</h2><p>{{ error }}</p>
      <p>En local, lance le viewer depuis <code>dataset-viewer/</code>. Pour un Space privé, configure le secret HF_TOKEN et le chemin du snapshot Qwen3.8.</p>
    </section>

    <section v-else class="qwen-grid" aria-label="Images annotées">
      <article v-for="row in result?.rows || []" :key="row.asset_id" class="qwen-card">
        <div class="qwen-image-wrap">
          <img v-if="row.image_url" :src="row.image_url" :alt="row.caption_candidate || `Sprite ${row.asset_id}`" loading="lazy">
          <span v-else class="qwen-image-missing">Image source absente</span>
        </div>
        <div class="qwen-card-body">
          <div class="qwen-card-head">
            <code>{{ row.asset_id }}</code>
            <span class="quality-pill" :class="`quality-${row.labels?.visual_quality || 'unknown'}`">{{ prettify(String(row.labels?.visual_quality || 'unknown')) }}</span>
          </div>
          <p class="qwen-caption">{{ row.caption_candidate || 'Aucune caption disponible.' }}</p>
          <p class="qwen-source">{{ row.source?.relative_path || 'Source inconnue' }}</p>
          <div class="qwen-taxonomy">
            <div v-for="(value, key) in row.labels || {}" :key="String(key)" class="qwen-tag-row">
              <span>{{ labelTitle(String(key)) }}</span><strong>{{ displayValue(value) }}</strong>
            </div>
          </div>
          <div class="qwen-card-foot">
            <span>{{ row.need_rework === false ? 'Pas de rework' : row.need_rework === true ? 'Need rework' : 'Rework inconnu' }}</span>
            <span>{{ row.review_status === 'human_approved' ? 'Validé' : 'Auto annoté' }}</span>
          </div>
        </div>
      </article>
      <div v-if="!loading && result?.rows.length === 0" class="qwen-empty">Aucune annotation ne correspond à ces filtres.</div>
    </section>

    <nav v-if="result && !error" class="qwen-pagination" aria-label="Pagination">
      <span>{{ visibleStart.toLocaleString('fr-FR') }}–{{ visibleEnd.toLocaleString('fr-FR') }} sur {{ result.filtered.toLocaleString('fr-FR') }}</span>
      <div><button class="button secondary" :disabled="offset === 0 || loading" @click="pageBy(-1)">← Précédent</button><strong>Page {{ page }} / {{ pages }}</strong><button class="button secondary" :disabled="page >= pages || loading" @click="pageBy(1)">Suivant →</button></div>
    </nav>
  </main>
</template>
