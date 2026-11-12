FROM node:22-bookworm-slim
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY app ./app
COPY server ./server
COPY nuxt.config.ts ./
COPY data/qwen_annotations.jsonl ./data/qwen_annotations.jsonl
ENV HOST=0.0.0.0
ENV PORT=7860
ENV NITRO_HOST=0.0.0.0
ENV NITRO_PORT=7860
EXPOSE 7860
CMD ["node", ".output/server/index.mjs"]
