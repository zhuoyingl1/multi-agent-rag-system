FROM python:3.12-slim

ARG INSTALL_PRODUCTION_EXTRAS=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --create-home app

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples

RUN if [ "$INSTALL_PRODUCTION_EXTRAS" = "true" ]; then \
      python -m pip install ".[production]"; \
    else \
      python -m pip install "."; \
    fi \
    && mkdir -p /app/output/uploads "$HF_HOME" \
    && chown -R app:app /app /home/app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/liveness', timeout=3)"

CMD ["uvicorn", "multi_agent_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

