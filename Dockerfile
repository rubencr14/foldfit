# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# Stage 2: Runtime
FROM python:3.11-slim

RUN groupadd -r foldfit && useradd -r -g foldfit foldfit

COPY --from=builder /install /usr/local
COPY src/ /app/src/
COPY config.yaml /app/config.yaml

WORKDIR /app
USER foldfit

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "foldfit.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
