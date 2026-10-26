FROM python:3.10-slim-bookworm
WORKDIR /app
RUN python -m pip install --no-cache-dir --upgrade pip==26.2.2
RUN python -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY src/requirements-rerank.txt .
RUN python -m pip install --no-cache-dir -r requirements-rerank.txt
COPY src/retrieval/ /app/retrieval/
COPY scripts/download_reranker.py /app/download_reranker.py
ENV RERANK_MODEL_PATH=/app/assets/models/reranker
CMD ["python", "-m", "retrieval.server"]
