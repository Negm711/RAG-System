from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from models.ChunkModel import ChunkModel
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json
import logging

logger = logging.getLogger("uvicorn.error")


class NLPController(BaseController):

    def __init__(
        self, vectordb_client, generation_client, embedding_client, template_parser
    ):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

    def create_collection_name(self, project_id: str):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()

    async def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(
            collection_name=collection_name
        )

    async def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vectordb_client.get_collection_info(
            collection_name=collection_name
        )

        return json.loads(json.dumps(collection_info, default=lambda x: x.__dict__))

    async def process_project_files(self, project: Project, do_reset: bool = False):

        if not hasattr(self, "db_client") or self.db_client is None:
            from celery_app import get_setup_utils

            _, db_client, _, _, _, _, _, _ = await get_setup_utils()
            self.db_client = db_client

        chunk_model = await ChunkModel.create_instance(db_client=self.db_client)

        chunks_ids = []
        chunks = []

        project_chunks = await chunk_model.get_poject_chunks(
            project_id=project.project_id
        )

        logger.error(
            f"--> Project Chunks Found: {len(project_chunks) if project_chunks else 0}"
        )

        if not project_chunks:
            return False

        for chunk in project_chunks:
            chunk_id = getattr(chunk, "chunk_id", getattr(chunk, "id", None))
            chunks_ids.append(chunk_id)
            chunks.append(chunk)

        if not chunks:
            return False

        await self.index_into_vector_db(
            project=project, chunks=chunks, chunks_ids=chunks_ids, do_reset=do_reset
        )

        return True

    async def index_into_vector_db(
        self,
        project: Project,
        chunks: List[DataChunk],
        chunks_ids: List[int],
        do_reset: bool = False,
    ):

        collection_name = self.create_collection_name(project_id=project.project_id)

        texts = [c.chunk_text for c in chunks]
        metadata = [c.chunk_metadata for c in chunks]
        vectors = self.embedding_client.embed_text(
            text=texts, document_type=DocumentTypeEnum.DOCUMENT.value
        )

        _ = await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        _ = await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )

        return True

    async def search_vector_db_collection(
        self, project: Project, text: str, limit: int = 10
    ):

        query_vector = None
        collection_name = self.create_collection_name(project_id=project.project_id)

        vectors = self.embedding_client.embed_text(
            text=text, document_type=DocumentTypeEnum.QUERY.value
        )

        if not vectors or len(vectors) == 0:
            return False

        if isinstance(vectors, list) and len(vectors) > 0:
            query_vector = vectors[0]

        if not query_vector:
            return False

        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name, vector=query_vector, limit=limit
        )

        if not results:
            return False

        return results

    async def answer_rag_question(self, project: Project, query: str, limit: int = 10):

        answer, full_prompt, chat_history = None, None, None

        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, chat_history

        return self.answer_from_documents(query, retrieved_documents)

    def answer_from_documents(self, query, retrieved_documents):
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join(
            [
                self.template_parser.get(
                    "rag",
                    "document_prompt",
                    {
                        "doc_num": idx + 1,
                        "chunk_text": self.generation_client.process_text(doc.text),
                    },
                )
                for idx, doc in enumerate(retrieved_documents)
            ]
        )

        footer_prompt = self.template_parser.get(
            "rag", "footer_prompt", {"query": query}
        )

        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([documents_prompts, footer_prompt])

        answer = self.generation_client.generate_text(
            prompt=full_prompt, chat_history=chat_history
        )

        return answer, full_prompt, chat_history
