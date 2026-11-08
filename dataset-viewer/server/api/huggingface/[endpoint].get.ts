const datasetId = 'ethanzxv/pixelart-dataset'
const allowedEndpoints = new Set(['splits', 'rows', 'search'])
const allowedQueryKeys = new Set(['config', 'split', 'offset', 'length', 'query'])

export default defineEventHandler(async (event) => {
  const endpoint = getRouterParam(event, 'endpoint')
  if (!endpoint || !allowedEndpoints.has(endpoint)) {
    throw createError({ statusCode: 404, statusMessage: 'Endpoint Hugging Face inconnu.' })
  }

  const authorization = getHeader(event, 'authorization')
  const headers: Record<string, string> = {}
  if (authorization?.startsWith('Bearer ')) headers.Authorization = authorization

  const incomingQuery = getQuery(event)
  const query: Record<string, string> = { dataset: datasetId }
  for (const key of allowedQueryKeys) {
    const value = incomingQuery[key]
    if (typeof value === 'string') query[key] = value
  }

  setResponseHeader(event, 'Cache-Control', 'no-store')
  try {
    return await $fetch(`https://datasets-server.huggingface.co/${endpoint}`, {
      headers,
      query,
      retry: 0,
      timeout: 30_000
    })
  } catch (error) {
    const upstream = error as { response?: { status?: number; _data?: { error?: string } }; statusCode?: number }
    const statusCode = upstream.response?.status ?? upstream.statusCode ?? 502
    const statusMessage = upstream.response?._data?.error ?? (statusCode === 401 || statusCode === 403
      ? 'Accès refusé. Vérifie le jeton et ses droits de lecture sur ce dataset.'
      : 'La requête Hugging Face a échoué.')
    throw createError({ statusCode, statusMessage })
  }
})
