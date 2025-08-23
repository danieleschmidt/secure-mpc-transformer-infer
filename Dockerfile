# Production Dockerfile for Secure MPC Transformer
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Copy requirements and install Python dependencies
COPY --chown=app:app pyproject.toml .
COPY --chown=app:app src/ src/

# Install dependencies
RUN pip install --user --no-cache-dir -e .

# Copy application code
COPY --chown=app:app . .

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "-m", "secure_mpc_transformer.server", "--host", "0.0.0.0", "--port", "8080"]
