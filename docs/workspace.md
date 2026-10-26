# Document workspace

Start the existing Docker stack, including the optional reranker if you want reranked answers. From the repository root:

```bash
python run_ui.py
```

Open `http://127.0.0.1:7861`. Ctrl+C closes the interface server; it does not stop Docker. You can change the UI port with `--port 7862`. The API address defaults to `http://127.0.0.1:8000`; `--api` accepts a local HTTP address only.

## Existing documents

Enter the project ID used when the file was indexed. Choose the retrieval method and passage limit, then ask a question. The initial ID, 1063, corresponds to the local example used during development; another installation should select its own indexed project.

The reranked answer displays the passages returned by that endpoint, along with available file IDs, page numbers and chunk IDs. The original answer endpoint has no structured source response, so the UI does not invent source metadata for vector-only answers. Older indexes can have empty metadata.

Ranking scores are not confidence percentages. Source passages are context supplied to generation, not verified sentence-by-sentence citations. The configured generation provider can apply its existing passage-length limit.

## New documents

Choose **New**, select one PDF or UTF-8 TXT file, then **Upload & prepare**. The interface chooses a random positive project ID, uploads the file and starts the existing processing/indexing workflow using 800-character windows with 120-character overlap. It never sends a reset request. Existing projects are used for questions, not reprocessing through this UI.

The upload button is disabled after submission to avoid accidental duplicate processing. If an upload succeeded but processing fails, the status explicitly says not to upload again. Use the API/logs to inspect or retry that workflow. This interface intentionally does not expose bulk operations, deletion or reset controls.

The preparation status polls index availability every four seconds for up to four minutes. It does not inspect Celery task state. For large documents an index can become available before every batch finishes; use worker logs to establish full task completion. A timeout reports uncertainty rather than claiming success. **Check index** can be used again without uploading.

## Local operation

The interface server binds to localhost. Its relay has a fixed local API target and allows only the upload, process-and-push, index-info and answer routes needed by the page. Browser requests to a different origin are rejected. It does not read `.env` files, connect to the database, manage Docker or store provider keys. The standard answer's internal prompt/history are removed from the UI response.

The interface has no CDN, analytics, remote font or frontend framework dependency. This does not make the full pipeline offline: your existing embedding/generation providers receive the input/context required by their requests. The interface does not publish a public sharing link.

Checks cover the relay's routes, upload passthrough, debug-field removal, unavailable-API errors and cross-origin rejection. Run `python -m unittest discover -s tests -v` with the lightweight test requirements already documented in the README.
