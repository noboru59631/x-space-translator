FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/service \
    BRIDGE_TEMP_DIR=/tmp/x-space-translator-render-bridge

WORKDIR /service

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

COPY cloud/render-bridge/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY app/__init__.py /service/app/__init__.py
COPY app/services/__init__.py /service/app/services/__init__.py
COPY app/services/downloader.py /service/app/services/downloader.py
COPY app/services/errors.py /service/app/services/errors.py
COPY cloud/render-bridge/render_bridge_app /service/render_bridge_app

RUN mkdir -p /tmp/x-space-translator-render-bridge \
    && chown -R appuser:appuser /tmp/x-space-translator-render-bridge

USER appuser
EXPOSE 10000

CMD ["python", "-m", "render_bridge_app"]
