export default defineEventHandler(async (event) => {
  const imagePath = getQuery(event).path
  if (typeof imagePath !== 'string' || !imagePath.startsWith('data/')) {
    throw createError({ statusCode: 400, statusMessage: 'Chemin d’image invalide.' })
  }
  const relative = imagePath.slice(5)
  if (!relative.toLowerCase().endsWith('.png') || relative.split('/').some(part => !part || part === '.' || part === '..') || relative.includes('\\')) {
    throw createError({ statusCode: 400, statusMessage: 'Chemin d’image invalide.' })
  }
  return sendRedirect(event, `/api/qwen-image?path=${encodeURIComponent(relative)}`, 302)
})
