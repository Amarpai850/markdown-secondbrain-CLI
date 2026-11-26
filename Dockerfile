# ==========================================
# Stage 1: Build & Automated Test (Python 3.11)
# ==========================================
FROM python:3.11-slim AS test

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project specification, sources, tests, and sample notes
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY web/ ./web/
COPY sample_notes/ ./sample_notes/
COPY tests/ ./tests/

# Install with web, semantic, and development dependencies
RUN pip install --no-cache-dir ".[web,semantic,dev]"

# Execute the complete test suite on Python 3.11
# The container build will strictly FAIL if any test fails
RUN pytest -v --tb=short && touch /tmp/.tests_passed

# ==========================================
# Stage 2: Minimal Secure Runtime
# ==========================================
FROM python:3.11-slim AS runtime

# Enforce dependency on Stage 1 passing tests (prevents BuildKit from pruning the test stage)
COPY --from=test /tmp/.tests_passed /tmp/.tests_passed
RUN rm -f /tmp/.tests_passed

# Create an unprivileged non-root user
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

# Copy project specification and runtime assets
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY web/ ./web/
COPY sample_notes/ ./sample_notes/

# Non-editable install of package with web and semantic extras
RUN pip install --no-cache-dir ".[web,semantic]"

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    BRAIN_NOTES_DIR=/app/sample_notes \
    BRAIN_WEB_DIR=/app/web

# Pre-build inverted index and knowledge graph at build time (baked in)
RUN brain index /app/sample_notes

# Secure ownership and switch to unprivileged user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Launch uvicorn on 0.0.0.0:$PORT with proxy headers enabled
CMD ["sh", "-c", "uvicorn brain.api:create_app --factory --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers --forwarded-allow-ips='*'"]
