# Retrieval change checks

Checked on 26 October 2026 against source commit `ce8d60f`.

- Unit checks cover overlap offsets, page isolation, metadata preservation, stable reranking, malformed scores, request bounds, missing-model behavior, retrieval metrics, expanded candidates and answer context selection.
- A compatibility check locks the existing legacy splitter implementation. Legacy request defaults and the original search response fields are retained.
- Python source compilation and Git whitespace checks passed.
- The pinned MiniLM model was downloaded and run on CPU using PyTorch 2.11.0 and Transformers 4.57.6. In a small synthetic fixture, the passage listing the architecture components ranked above unrelated passages. This is a smoke check, not a project benchmark.
- The HTTP adapter was checked separately with the real local model. The Docker image isolates these model dependencies from the original application's dependencies.

The complete Docker stack and the user's live database/provider credentials were not available in this review environment. A live ingestion/search/answer check is still required after installing the change locally. Existing data and configuration have not been migrated or reset. No answer-accuracy or retrieval-improvement percentage is reported.
