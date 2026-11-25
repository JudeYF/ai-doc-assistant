FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is used by the Compose healthcheck. libgomp1 is needed by Chroma/ONNX.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY .streamlit .streamlit
COPY app app
COPY scripts scripts
COPY data/中华人民共和国治安管理处罚法.pdf data/中华人民共和国治安管理处罚法.pdf

RUN chmod +x scripts/docker-entrypoint.sh

EXPOSE 8000 8501

ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["api"]
