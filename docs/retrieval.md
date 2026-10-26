# Retrieval options

The current API and indexes can stay in place. The additions below are opt-in and do not reset or migrate an existing collection.

## Page-aware chunks

For a new test project, upload a PDF or text file using the normal upload endpoint. Send this body to `/api/v1/data/process-and-push/{project_id}`:

```json
{
  "chunk_size": 800,
  "overlap_size": 120,
  "chunk_strategy": "page_overlap",
  "do_reset": 0
}
```

Sizes are characters, not tokens. Each chunk stays within one loader page and records the original metadata, file ID, zero-based page index, and start/end character offsets. The PDF loader's original `page` field is also retained. The splitter uses fixed windows; it can split a sentence. It is deliberately simple and does not claim semantic segmentation.

Use a new project ID to compare strategies. Do not reprocess an existing project repeatedly without deciding how to handle its existing chunks. Resetting is a separate explicit operation and can remove the old index. Existing indexes do not gain page metadata automatically.

## Optional local reranker

Model: `cross-encoder/ms-marco-MiniLM-L6-v2`, revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`. This is an English retrieval model. Arabic prompt support in the application does not establish Arabic reranking quality.

The model runs on CPU. Each query-passage pair has a 512-token budget; long passages are truncated for scoring only. The original retrieved text is retained in the response. Scores are ranking logits, not probabilities or factuality ratings. Query relevance does not certify answer correctness.

From the repository root, build the isolated reranker and updated application images:

```bash
docker compose -f docker/docker-compose.yml -f docker/compose.rerank.yml build reranker fastapi celery-worker celery-beat flower
docker compose -f docker/docker-compose.yml -f docker/compose.rerank.yml up -d --no-deps reranker fastapi celery-worker celery-beat flower
docker compose -f docker/docker-compose.yml -f docker/compose.rerank.yml exec reranker python /app/download_reranker.py --output /app/assets/models/reranker
```

This installs CPU PyTorch and reranking dependencies in a separate reranker image. Existing API and Celery dependency pins remain unchanged. It is a substantial one-time download even though it costs no service fees. The weights live in the existing shared assets volume. The script downloads only the pinned safetensors model and tokenizer; inference uses local files only. No API key is needed for the model download.

The first reranked request loads one model in the private reranker service. API workers call it over the internal Docker network. Normal requests do not load it. Missing files or optional dependencies return HTTP 503 on the new endpoints; the original endpoints remain available. The initial load is not included in the reported retrieval/reranking timings.

Call either `/api/v1/nlp/index/search-reranked/{project_id}` or `/api/v1/nlp/index/answer-reranked/{project_id}`:

```json
{"text": "What are the main parts of the architecture?", "candidate_limit": 20, "limit": 5}
```

The system retrieves up to 20 vector candidates and reranks them into five results. The response includes chunk IDs, text, source metadata, vector scores, reranking scores and stage timings. The answer route uses the same existing answer prompt and generation settings. Its `results` are supplied context passages, not verified claim-level citations; the existing generation client can shorten each passage according to its configured character limit.

## Paired evaluation

Write questions against an indexed project and manually label the chunk IDs that contain the evidence. Do this before inspecting the reranked order. Freeze the questions, project/index version, model settings and labels. Keep tuning questions separate from the final evaluation set.

Question file format:

```json
[
  {
    "id": "architecture-1",
    "project_id": 101,
    "text": "What are the main parts of the architecture?",
    "relevant_ids": ["42", "43"],
    "candidate_limit": 20,
    "k": 5
  }
]
```

The IDs above illustrate the format; replace them with real evidence IDs. Use one k across the run. The collector calls the new search route and saves each completed question immediately:

```bash
python scripts/collect_retrieval_evaluation.py questions.json --output reports/retrieval/rankings.json
python scripts/evaluate_retrieval.py reports/retrieval/rankings.json --k 5 --output reports/retrieval/metrics.json
```

A request failure stops collection rather than silently dropping the question. Rerun the same command to resume; changed questions with an existing ID are rejected. Complete every question before publishing aggregate results. Copy the frozen index/model configuration alongside the records; the collector does not fingerprint the live deployment.

Recall@k is the fraction of labelled relevant chunks present in the first k results. Mean reciprocal rank at k averages the reciprocal position of the first relevant hit, with zero for a miss. Latency compares vector retrieval with retrieval plus reranking from the same request and candidate pool. It excludes model loading and answer generation; it is not a separate end-to-end benchmark of both endpoints. Warm the model before a timed run and report dataset size and hardware.

This measures retrieval, not hallucination rate. Answer factuality needs source-based review. No benchmark scores are included until a real labelled run is complete.

## Returning to the original path

Keep using `/index/search` and `/index/answer` with `chunk_strategy="legacy"` for newly processed files. Neither model setup nor a reranked request modifies existing chunks. To disable the new path, restart the API with only `docker/docker-compose.yml`, then stop the `reranker` service using the optional overlay; keep all data volumes intact. Do not use `down -v`.
