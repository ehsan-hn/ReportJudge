FROM python:3.11-slim

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLM_PROVIDER=fake

# Set container working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create dedicated non-root user for security hardening
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

# Copy application source code and evaluation benchmark suite
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser eval/ ./eval/

# Switch to non-root execution context
USER appuser

# Expose FastAPI service port
EXPOSE 8000

# Liveness and readiness healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

# Default container startup command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
