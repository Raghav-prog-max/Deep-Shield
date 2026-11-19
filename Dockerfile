FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEEPSHIELD_DEVICE=cpu \
    DEEPSHIELD_M02_BACKEND=hf \
    PORT=8002

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD sh -c 'curl -fsS "http://127.0.0.1:${PORT:-8002}/api/v1/m01/health" || exit 1'

CMD ["python", "scripts/cloud_entrypoint.py"]
