# ---------- stage 1: build the static site ----------
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .

# IMPORTANT: Vite bakes this URL into the JavaScript at BUILD time.
# It is the address the *browser* uses, so it must be reachable from the host
# (http://localhost:8002), NOT the Docker-internal name "orchestrator".
ARG VITE_API_BASE_URL=http://localhost:8002
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN npm run build

# ---------- stage 2: serve it with nginx (runs as a non-root user, port 8080) ----------
FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/healthz || exit 1
