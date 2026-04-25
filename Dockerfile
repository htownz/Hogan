FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY hogan_bot ./hogan_bot
COPY scripts ./scripts
COPY diagnostics ./diagnostics
COPY docs ./docs
COPY AGENTS.md .

RUN mkdir -p /app/data /app/models /app/reports

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python - <<'PY' || exit 1
import os
import urllib.request
port = int(os.getenv("HOGAN_METRICS_PORT", "8000"))
urllib.request.urlopen(f"http://127.0.0.1:{port}", timeout=3).read(256)
PY

CMD ["python", "-m", "hogan_bot.main"]
