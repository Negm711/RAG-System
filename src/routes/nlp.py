from fastapi import FastAPI, APIRouter, status, Request
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushRequest, SearchRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController
from models import ResponseSignal
from tqdm.auto import tqdm

import logging

logger = logging.getLogger("uvicorn.error")

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)


@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: int, push_request: PushRequest):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)

    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    await nlp_controller.process_project_files(
        project=project, do_reset=push_request.do_reset
    )

    return JSONResponse(content={"signal": "data_indexed_successfully"})


@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: int):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)

    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    collection_info = await nlp_controller.get_vector_db_collection_info(
        project=project
    )

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info,
        }
    )


@nlp_router.post("/index/search/{project_id}")
async def search_index(
    request: Request, project_id: int, search_request: SearchRequest
):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)

    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    results = await nlp_controller.search_vector_db_collection(
        project=project, text=search_request.text, limit=search_request.limit
    )

    if not results:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value},
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "results": [{"text": result.text, "score": result.score} for result in results],
        }
    )


@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request: Request, project_id: int, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)

    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
    )

    if not answer:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.RAG_ANSWER_ERROR.value},
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history,
        }
    )


# Optional path: existing /search and /answer behavior is unchanged.
import asyncio
from time import perf_counter
from fastapi import HTTPException
from routes.schemes.nlp import RerankedSearchRequest
from retrieval.reranking import reranker, rank_documents, RerankerUnavailable


async def _reranked_documents(request, project_id, body):
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)
    controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )
    # Fail before spending an embedding request when the model is not provisioned.
    try:
        await asyncio.to_thread(reranker.ready)
    except RerankerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    started = perf_counter()
    candidates = await controller.search_vector_db_collection(
        project=project, text=body.text, limit=body.candidate_limit
    )
    retrieval_ms = (perf_counter() - started) * 1000
    if not candidates:
        raise HTTPException(status_code=404, detail="No indexed passages found for this project")
    started = perf_counter()
    try:
        ranked = await asyncio.to_thread(
            rank_documents, body.text, candidates, body.limit, reranker.score
        )
    except RerankerUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    result = {
        "candidate_count": len(candidates),
        "retrieval_ms": retrieval_ms,
        "rerank_ms": (perf_counter() - started) * 1000,
        "baseline_ids": [doc.chunk_id for doc in candidates],
        "results": [dict(chunk_id=doc.chunk_id, text=doc.text, metadata=doc.metadata,
                         vector_score=doc.score, rerank_score=score) for doc, score in ranked],
    }
    return controller, ranked, result


@nlp_router.post("/index/search-reranked/{project_id}")
async def search_reranked(request: Request, project_id: int, body: RerankedSearchRequest):
    _, _, result = await _reranked_documents(request, project_id, body)
    return dict(signal="reranked_search_success", **result)


@nlp_router.post("/index/answer-reranked/{project_id}")
async def answer_reranked(request: Request, project_id: int, body: RerankedSearchRequest):
    controller, ranked, result = await _reranked_documents(request, project_id, body)
    started = perf_counter()
    answer, _, _ = await asyncio.to_thread(
        controller.answer_from_documents, body.text, [doc for doc, _ in ranked]
    )
    if not answer:
        raise HTTPException(status_code=502, detail="Generation provider returned no answer")
    result["generation_ms"] = (perf_counter() - started) * 1000
    return dict(signal="reranked_answer_success", answer=answer, **result)
