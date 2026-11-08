export default defineNuxtConfig({
  compatibilityDate: '2026-11-08',
  css: ['~/assets/css/main.css'],
  devtools: { enabled: true },
  app: {
    head: {
      title: 'Dataset Viewer · Iris Pixelart',
      meta: [
        { name: 'description', content: 'Visualiseur local et Hugging Face pour le dataset pixel art.' },
        { name: 'theme-color', content: '#f5f5f2' }
      ]
    }
  }
})
