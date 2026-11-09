import { readFile, stat } from 'node:fs/promises'
import { resolve } from 'node:path'

type LocalImage = { path: string; caption: string; folder: string; name: string }
type Annotation = { labels?: Record<string, unknown> }
type TagItem = { key: string; label: string; count: number }
type TagGroup = { title: string; items: TagItem[]; denominator: number; multiple?: boolean }
type DatasetStats = {
  total: number
  captionGroups: TagGroup[]
  annotationTotal: number
  annotationCoveragePercent: number
  annotationGroups: TagGroup[]
}

const datasetRoot = resolve(process.cwd(), 'datasets', 'pixelart-dataset')
const indexPath = resolve(datasetRoot, 'viewer-index.json')
const annotationPath = resolve(datasetRoot, '.gitrepo', 'annotations', 'qwen3vl_fp8_annotations_2000.jsonl')
let statsPromise: Promise<DatasetStats> | null = null
let cachedVersion = ''
const valueLabels: Record<string, string> = {
  single_sprite: 'Sprite individuel', tile_set: 'Tileset', sprite_sheet: 'Planche de sprites', animation_strip: 'Bande d’animation',
  character: 'humanoid', creature: 'Créature', object: 'Objet', environment: 'Environnement', effect: 'Effet', typography: 'Typographie',
  front: 'Face', side: 'Profil', back: 'Dos', three_quarter: 'Trois-quarts', top_down: 'Vue du dessus',
  front_right: 'Avant droite', front_left: 'Avant gauche', back_right: 'Arrière droite', back_left: 'Arrière gauche', mixed: 'Mixte',
  standing: 'Debout', airborne: 'En suspension', crouching: 'Accroupi', flying: 'En vol', lying: 'Allongé', kneeling: 'À genoux',
  idle: 'Inactif', run: 'Course', walk: 'Marche', attack_melee: 'Attaque au corps à corps', jump: 'Saut',
  static_still: 'Image fixe', single_frame_unconfirmed: 'Frame unique non confirmée',
  not_applicable: 'Sans objet', unknown: 'Non précisé', transparent: 'Transparent', solid_color: 'Couleur unie', pixel_art: 'Pixel art',
  no_outline: 'Sans contour', dark_outline: 'Contour sombre', flat: 'Aplat', flat_shading: 'Ombrage en aplats',
  A_native_clean: 'A · natif propre', B_usable: 'B · utilisable'
}

function matchesAny(text: string, patterns: RegExp[]) {
  return patterns.some((pattern) => pattern.test(text))
}

function toGroups(counts: Map<string, Map<string, number>>, denominator: number, limits: Record<string, number> = {}, multipleGroups = new Set<string>()): TagGroup[] {
  return [...counts.entries()].map(([title, values]) => {
    const sorted = [...values.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    const limit = limits[title]
    const shown = limit ? sorted.slice(0, limit) : sorted
    if (limit && sorted.length > limit) shown.push(['Autres valeurs', sorted.slice(limit).reduce((sum, [, count]) => sum + count, 0)])
    return {
      title,
      denominator,
      multiple: multipleGroups.has(title),
      items: shown.map(([label, count]) => ({ key: label, label, count }))
    }
  })
}

function getStats(images: LocalImage[], annotations: Annotation[]): DatasetStats {
  const total = images.length
  const captionCounts = new Map<string, Map<string, number>>()
  const annotationCounts = new Map<string, Map<string, number>>()
  const count = (target: Map<string, Map<string, number>>, group: string, label: string) => {
    const values = target.get(group) ?? new Map<string, number>()
    values.set(label, (values.get(label) ?? 0) + 1)
    target.set(group, values)
  }

  const shapeRules: Array<[string, RegExp[]]> = [
    ['Humanoïde', [/\bhumanoid\b/i, /\bhuman(?:oid)?\b/i, /\bcharacter\b/i, /\bperson\b/i, /\bwarrior\b/i, /\bknight\b/i, /\bsoldier\b/i, /\bfighter\b/i]],
    ['Créature', [/\bcreature\b/i, /\bmonster\b/i, /\bbeast\b/i, /\banimal\b/i, /\bdragon\b/i, /\bwolf\b/i, /\bdog\b/i, /\bcat\b/i, /\bbird\b/i, /\bfish\b/i, /\binsect\b/i, /\bslime\b/i]],
    ['Mécanique', [/\brobot\b/i, /\brobotic\b/i, /\bmechanical\b/i, /\bmachine\b/i]]
  ]
  const viewRules: Array<[string, RegExp[]]> = [
    ['Face', [/\bfront(?:-facing)? view\b/i, /\bfront-facing\b/i]],
    ['Profil', [/\bside view\b/i, /\bprofile view\b/i, /\bside-facing\b/i]],
    ['Dos', [/\bback view\b/i, /\bviewed from the back\b/i, /\brear view\b/i]],
    ['Dessus / isométrique', [/\btop[- ]down\b/i, /\bisometric\b/i]],
    ['Trois-quarts', [/\bthree[- ]quarter\b/i]]
  ]
  const poseRules: Array<[string, RegExp[]]> = [
    ['Debout / immobile', [/\bstanding\b/i, /\bidle pose\b/i]],
    ['Marche / course', [/\bwalking\b/i, /\brunning\b/i]],
    ['Attaque / combat', [/\battacking\b/i, /\battack pose\b/i, /\bfighting\b/i]],
    ['Saut / en l’air', [/\bjumping\b/i, /\bleaping\b/i, /\bin mid-air\b/i, /\bairborne\b/i]],
    ['Vol', [/\bflying\b/i, /\bin flight\b/i]],
    ['Autre action décrite', [/\bcasting\b/i, /\bsitting\b/i, /\bcrouching\b/i, /\bkneeling\b/i, /\blying\b/i]]
  ]

  for (const image of images) {
    const caption = image.caption ?? ''
    const lower = caption.toLocaleLowerCase()
    const shapes = shapeRules.filter(([, patterns]) => matchesAny(caption, patterns)).map(([label]) => label)
    count(captionCounts, 'Forme indiquée par la légende', shapes.length ? [...new Set(shapes)].join(' + ') : 'Non précisée')

    const views = viewRules.filter(([, patterns]) => matchesAny(caption, patterns)).map(([label]) => label)
    count(captionCounts, 'Vue indiquée par la légende', views.length ? [...new Set(views)].join(' + ') : 'Non précisée')

    const poses = poseRules.filter(([, patterns]) => matchesAny(caption, patterns)).map(([label]) => label)
    count(captionCounts, 'Pose / action indiquée par la légende', poses.length ? [...new Set(poses)].join(' + ') : 'Non précisée')

    let background = 'Non précisé'
    if (matchesAny(lower, [/transparent background/, /transparent backdrop/, /alpha background/])) background = 'Transparent'
    else if (matchesAny(lower, [/white background/, /plain white/])) background = 'Blanc'
    else if (matchesAny(lower, [/black background/, /plain black/])) background = 'Noir'
    else if (matchesAny(lower, [/solid[- ]color background/, /colored background/, /colour background/])) background = 'Uni / coloré'
    count(captionCounts, 'Fond indiqué par la légende', background)

    const dimension = caption.match(/\b(\d{1,4})\s*[x×]\s*(\d{1,4})\s*(?:pixel|px)\b/i)
    count(captionCounts, 'Dimensions écrites dans la légende', dimension ? `${dimension[1]} × ${dimension[2]}` : 'Non précisées')

    let asset = 'Autre / non précisé'
    if (/\btileset\b|\btile set\b/i.test(caption)) asset = 'Tileset'
    else if (/\bsprite sheet\b|\bspritesheet\b|\bsprite atlas\b/i.test(caption)) asset = 'Planche de sprites'
    else if (/\bportrait\b/i.test(caption)) asset = 'Portrait'
    else if (/\bframe\b/i.test(caption)) asset = 'Frame d’animation'
    else if (/\bsprite\b/i.test(caption)) asset = 'Sprite'
    count(captionCounts, 'Type d’image indiqué par la légende', asset)
    count(captionCounts, 'Jeu / dossier', image.folder || 'Non classé')
  }

  // Exclude labels.role and asset_category: they contain game roles such as “enemy”.
  // Map the source's generic `character` subject label to the requested shape term.
  const fields: Array<{ field: string; title: string; multi?: boolean }> = [
    { field: 'asset_type', title: 'Type d’actif' },
    { field: 'subject_type', title: 'Forme du sujet' },
    { field: 'view', title: 'Vue caméra' },
    { field: 'facing_direction', title: 'Orientation' },
    { field: 'pose', title: 'Pose' },
    { field: 'action', title: 'Action' },
    { field: 'motion_representation', title: 'Représentation du mouvement' },
    { field: 'animation_status', title: 'État de l’animation' },
    { field: 'render_family', title: 'Style graphique' },
    { field: 'background_type', title: 'Fond' },
    { field: 'outline_style', title: 'Contour' },
    { field: 'shading_style', title: 'Ombrage' },
    { field: 'dominant_color', title: 'Couleurs dominantes', multi: true },
    { field: 'visible_equipment', title: 'Équipement visible', multi: true },
    { field: 'quality_tier', title: 'Qualité visuelle' }
  ]
  for (const annotation of annotations) {
    const labels = annotation.labels ?? {}
    for (const { field, title, multi } of fields) {
      const raw = labels[field]
      const values = multi ? (Array.isArray(raw) ? raw : []) : [raw]
      for (const entry of values) {
        if (typeof entry !== 'string' || !entry.trim()) continue
        let value = entry
        if (field === 'subject_type' && entry === 'character') value = 'humanoid'
        else value = valueLabels[entry] ?? entry.replaceAll('_', ' ')
        count(annotationCounts, title, value)
      }
    }
  }

  const annotationTotal = annotations.length
  const captionLimits = { 'Jeu / dossier': 12, 'Dimensions écrites dans la légende': 10 }
  const annotationLimits = { 'Couleurs dominantes': 10, 'Équipement visible': 10 }
  return {
    total,
    captionGroups: toGroups(captionCounts, total, captionLimits),
    annotationTotal,
    annotationCoveragePercent: total ? annotationTotal * 100 / total : 0,
    annotationGroups: toGroups(annotationCounts, annotationTotal, annotationLimits, new Set(['Couleurs dominantes', 'Équipement visible']))
  }
}

export default defineEventHandler(async () => {
  try {
    const [indexInfo, annotationInfo] = await Promise.all([
      stat(indexPath),
      stat(annotationPath).catch(() => null)
    ])
    const version = `${indexInfo.mtimeMs}:${annotationInfo?.mtimeMs ?? 0}`
    if (!statsPromise || version !== cachedVersion) {
      cachedVersion = version
      statsPromise = Promise.all([
        readFile(indexPath, 'utf8'),
        annotationInfo ? readFile(annotationPath, 'utf8') : Promise.resolve('')
      ]).then(([indexContent, annotationContent]) => {
        const images = JSON.parse(indexContent) as LocalImage[]
        const annotations = annotationContent.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line) as Annotation)
        return getStats(images, annotations)
      }).catch((error) => {
        statsPromise = null
        cachedVersion = ''
        throw error
      })
    }
    return await statsPromise
  } catch {
    throw createError({ statusCode: 404, statusMessage: 'Les statistiques locales ne sont pas encore disponibles.' })
  }
})
