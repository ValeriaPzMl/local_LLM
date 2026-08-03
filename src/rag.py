import hashlib
from pathlib import Path

import chromadb

from src.embeddings import OllamaEmbeddings
from src.loaders import DocumentLoader
from src.utils import split_into_chunks


class RAGService:
    def __init__(
        self,
        database_path: str = "data/chroma",
    ):
        Path(database_path).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=database_path,
        )

        self.embeddings = OllamaEmbeddings()

    def get_collection(self, channel_id: int):
        collection_name = f"channel_{channel_id}"

        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": (
                    "Documentos asociados a un canal de Discord"
                )
            },
        )

    @staticmethod
    def calculate_file_hash(file_path: str) -> str:
        """
        Genera un identificador SHA-256 basado en el contenido
        completo del archivo.
        """
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as file:
            while chunk := file.read(1024 * 1024):
                sha256.update(chunk)

        return sha256.hexdigest()

    def document_exists(
        self,
        channel_id: int,
        document_id: str,
    ) -> bool:
        collection = self.get_collection(channel_id)

        result = collection.get(
            where={
                "document_id": document_id,
            },
            include=[],
        )

        return len(result["ids"]) > 0

    async def index_document(
        self,
        channel_id: int,
        file_path: str,
        replace_existing: bool = False,
    ) -> dict:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"No existe el archivo: {file_path}"
            )

        document_id = self.calculate_file_hash(
            str(path)
        )

        collection = self.get_collection(channel_id)

        already_exists = self.document_exists(
            channel_id=channel_id,
            document_id=document_id,
        )

        if already_exists and not replace_existing:
            return {
                "status": "already_exists",
                "document_id": document_id,
                "file_name": path.name,
                "chunks": 0,
            }

        if already_exists and replace_existing:
            collection.delete(
                where={
                    "document_id": document_id,
                }
            )

        text = DocumentLoader.load(str(path))

        chunks = split_into_chunks(
            text,
            chunk_size=1000,
            overlap=150,
        )

        if not chunks:
            raise ValueError(
                "El documento no contiene texto utilizable."
            )

        embeddings = await self.embeddings.embed_texts(
            chunks
        )

        if len(embeddings) != len(chunks):
            raise ValueError(
                "La cantidad de embeddings no coincide "
                "con la cantidad de chunks."
            )

        ids = [
            f"{document_id}_{index}"
            for index in range(len(chunks))
        ]

        metadatas = [
            {
                "document_id": document_id,
                "source": path.name,
                "chunk_index": index,
                "channel_id": str(channel_id),
            }
            for index in range(len(chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return {
            "status": (
                "replaced"
                if already_exists
                else "indexed"
            ),
            "document_id": document_id,
            "file_name": path.name,
            "chunks": len(chunks),
        }

    async def search(
        self,
        channel_id: int,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        collection = self.get_collection(channel_id)

        if collection.count() == 0:
            return []

        query_embedding = await self.embeddings.embed_query(
            query
        )

        result_limit = min(
            limit,
            collection.count(),
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=result_limit,
        )

        documents = results.get(
            "documents",
            [[]],
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]

        distances = results.get(
            "distances",
            [[]],
        )[0]

        matches = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            matches.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return matches
    async def build_context(
        self,
        channel_id: int,
        query: str,
        limit: int = 5,
    ) -> str:

        results = await self.search(
            channel_id,
            query,
            limit,
        )

        if not results:
            return ""

        context = []

        for result in results:
            context.append(result["text"])

        return "\n\n".join(context)

    def delete_document(
        self,
        channel_id: int,
        document_id: str,
    ) -> None:
        collection = self.get_collection(channel_id)

        collection.delete(
            where={
                "document_id": document_id,
            }
        )
    async def answer_question(
        self,
        channel_id: int,
        question: str,
        ollama_client,
        limit: int = 5,
    ) -> str:
        context = await self.build_context(
            channel_id=channel_id,
            query=question,
            limit=limit,
        )

        if not context:
            return "No encontré información relacionada en los documentos."

        system_prompt = """
    Eres un asistente que responde preguntas usando únicamente
    la información proporcionada en el contexto.

    Si la respuesta no aparece en el contexto, responde:
    "No encontré esa información en los documentos."

    No inventes datos.
    Responde de forma clara y directa.
    """.strip()

        user_prompt = f"""
    CONTEXTO:

    {context}

    PREGUNTA:

    {question}
    """.strip()

        response = await ollama_client.generate(
            system=system_prompt,
            prompt=user_prompt,
        )

        return response
    def list_documents(
        self,
        channel_id: int,
    ) -> list[dict]:
        collection = self.get_collection(channel_id)

        results = collection.get(
            include=["metadatas"],
        )

        documents = {}

        for metadata in results.get("metadatas", []):
            if not metadata:
                continue

            document_id = metadata["document_id"]

            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "file_name": metadata["source"],
                    "chunks": 0,
                }

            documents[document_id]["chunks"] += 1

        return list(documents.values())