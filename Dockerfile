# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user -- principle of least privilege
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ src/
COPY pyproject.toml .

# mlruns/ is mounted as a volume at runtime (see docker-compose.yml).
# Create the mount point so Docker does not auto-create it as root.
RUN mkdir -p mlruns && chown appuser:appuser mlruns

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.serving.serve:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
