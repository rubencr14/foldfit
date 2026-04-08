# Stage 1: Build dependencies
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Install OpenFold (Python-only, no CUDA extensions — those come from nvidia base)
RUN git clone --depth 1 https://github.com/aqlaboratory/openfold.git /tmp/openfold && \
    cp -r /tmp/openfold/openfold /install/lib/python3.10/site-packages/ && \
    pip install --no-cache-dir --prefix=/install dm-tree modelcif && \
    rm -rf /tmp/openfold

# Install foldfit
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install --no-deps .

# Stage 2: Runtime
FROM python:3.10-slim

# Security: non-root user
RUN groupadd -r foldfit && useradd -r -g foldfit -d /app -s /sbin/nologin foldfit

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code and scripts
COPY --chown=foldfit:foldfit src/ /app/src/
COPY --chown=foldfit:foldfit scripts/ /app/scripts/
COPY --chown=foldfit:foldfit config.yaml /app/config.yaml

# Create data directories owned by foldfit
RUN mkdir -p /app/data /app/checkpoints && chown -R foldfit:foldfit /app

WORKDIR /app

# Security: drop all capabilities, read-only filesystem compatible
USER foldfit

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "foldfit.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
