FROM node:22-alpine

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG API_INTERNAL_BASE_URL=http://api:8002
ENV API_INTERNAL_BASE_URL=${API_INTERNAL_BASE_URL}
RUN npm run build

ENV NODE_ENV=production
EXPOSE 3000
CMD ["npm", "run", "start"]
