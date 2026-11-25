#!/bin/sh
set -e

cd /app

role="${1:-api}"

case "$role" in
  api)
    python scripts/init_db.py
    exec uvicorn app.main:app --host 0.0.0.0 --port 8000
    ;;
  web)
    exec streamlit run app/streamlit_app.py \
      --server.address 0.0.0.0 \
      --server.port 8501 \
      --server.headless true \
      --browser.gatherUsageStats false
    ;;
  init)
    exec python scripts/init_db.py
    ;;
  *)
    exec "$@"
    ;;
esac
