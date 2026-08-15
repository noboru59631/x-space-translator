FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORKER_TEMP_DIR=/tmp/x-space-worker \
    HF_HOME=/home/appuser/.cache/huggingface \
    PORT=7860

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY cloud/worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app /workspace/app
COPY cloud/__init__.py /workspace/cloud/__init__.py
COPY cloud/worker /workspace/cloud/worker

RUN mkdir -p /tmp/x-space-worker /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /tmp/x-space-worker /home/appuser

USER appuser
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "cloud.worker.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
