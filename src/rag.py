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
        self.last_sources: dict[int, list[dict]] = {}

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
                "kind": "document",
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

    async def index_text(
        self,
        channel_id: int,
        text: str,
        source_name: str,
        document_id: str | None = None,
        kind: str = "text",
        replace_existing: bool = False,
    ) -> dict:
        text = text.strip()

        if not text:
            raise ValueError(
                "No se recibió texto utilizable para indexar."
            )

        if document_id is None:
            document_id = hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest()

        collection = self.get_collection(channel_id)

        already_exists = self.document_exists(
            channel_id=channel_id,
            document_id=document_id,
        )

        if already_exists and not replace_existing:
            return {
                "status": "already_exists",
                "document_id": document_id,
                "file_name": source_name,
                "chunks": 0,
                "kind": kind,
            }

        if already_exists and replace_existing:
            collection.delete(
                where={
                    "document_id": document_id,
                }
            )

        chunks = split_into_chunks(
            text,
            chunk_size=1000,
            overlap=150,
        )

        if not chunks:
            raise ValueError(
                "El texto no produjo ningún fragmento."
            )

        embeddings = await self.embeddings.embed_texts(
            chunks
        )

        if len(embeddings) != len(chunks):
            raise ValueError(
                "La cantidad de embeddings no coincide "
                "con la cantidad de fragmentos."
            )

        ids = [
            f"{document_id}_{index}"
            for index in range(len(chunks))
        ]

        metadatas = [
            {
                "document_id": document_id,
                "source": source_name,
                "chunk_index": index,
                "channel_id": str(channel_id),
                "kind": kind,
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
            "file_name": source_name,
            "chunks": len(chunks),
            "kind": kind,
        }

    async def search(
        self,
        channel_id: int,
        query: str,
        limit: int = 5,
        max_per_source: int = 2,
    ) -> list[dict]:
        collection = self.get_collection(channel_id)
        total_chunks = collection.count()

        if total_chunks == 0:
            return []

        query_embedding = await self.embeddings.embed_query(
            query
        )

        # Recuperamos más candidatos de los que mostraremos.
        # Esto permite encontrar información en documentos pequeños.
        candidate_limit = min(
            max(limit * 5, limit),
            total_chunks,
        )

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_limit,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        candidates = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            candidates.append(
                {
                    "text": document,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )

        # Evita que un PDF grande ocupe todos los resultados.
        selected = []
        source_counts: dict[str, int] = {}

        for candidate in candidates:
            source = candidate["metadata"].get(
                "source",
                "Documento desconocido",
            )

            current_count = source_counts.get(source, 0)

            if current_count >= max_per_source:
                continue

            selected.append(candidate)
            source_counts[source] = current_count + 1

            if len(selected) >= limit:
                break

        return selected 
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
    async def build_context(
        self,
        channel_id: int,
        query: str,
        limit: int = 5,
    ) -> tuple[str, list[dict]]:
        results = await self.search(
            channel_id=channel_id,
            query=query,
            limit=limit,
        )

        if not results:
            return "", []

        context_blocks = []
        sources = []
        seen_sources = set()

        for position, result in enumerate(results, start=1):
            metadata = result.get("metadata") or {}

            source = metadata.get(
                "source",
                "Documento desconocido",
            )

            chunk_index = int(
                metadata.get("chunk_index", 0)
            )

            context_blocks.append(
                f"[Fuente {position}: {source}, "
                f"fragmento {chunk_index + 1}]\n"
                f"{result['text']}"
            )

            source_key = (
                source,
                chunk_index,
            )

            if source_key not in seen_sources:
                seen_sources.add(source_key)

                sources.append(
                    {
                        "number": position,
                        "source": source,
                        "chunk_index": chunk_index,
                        "distance": result.get("distance"),
                    }
                )

        context = "\n\n---\n\n".join(context_blocks)

        return context, sources


    async def answer_question(
        self,
        channel_id: int,
        question: str,
        ollama_client,
        facts: list[str] | None = None,
        history: list[dict[str, str]] | None = None,
        limit: int = 5,
    ) -> str:
        context, sources = await self.build_context(
            channel_id=channel_id,
            query=question,
            limit=limit,
        )
        self.last_sources[channel_id] = sources

        facts = facts or []
        history = history or []

        facts_text = (
            "\n".join(
                f"- {fact}"
                for fact in facts
            )
            if facts
            else "No hay hechos permanentes guardados."
        )

        recent_history = history[-10:]

        history_text = (
            "\n".join(
                f"{item['role']}: {item['content']}"
                for item in recent_history
            )
            if recent_history
            else "No hay conversación reciente."
        )

        documents_text = (
            context
            if context
            else "No se recuperaron fragmentos relevantes."
        )

        system_prompt = """
    Eres ComputahMind, un asistente para proyectos.

    Dispones de tres tipos de información:

    1. Memoria permanente del proyecto.
    2. Historial reciente de la conversación.
    3. Fragmentos recuperados de documentos.

    Reglas:

    - Utiliza la memoria permanente para datos como el nombre del
    proyecto, tecnologías, objetivos y decisiones técnicas.
    - Utiliza el historial para mantener continuidad.
    - Utiliza los documentos cuando la pregunta trate sobre su contenido.
    - No afirmes que algo no existe solamente porque no aparece en los
    documentos; primero revisa la memoria y el historial.
    - Ignora los fragmentos que no estén relacionados con la pregunta.
    - Cuando uses un documento, cita [Fuente 1], [Fuente 2], etc.
    - No inventes información.
    - Responde siempre en español.
    """.strip()

        user_prompt = f"""
    MEMORIA PERMANENTE:

    {facts_text}

    HISTORIAL RECIENTE:

    {history_text}

    DOCUMENTOS RECUPERADOS:

    {documents_text}

    PREGUNTA ACTUAL:

    {question}
    """.strip()

        response = await ollama_client.generate(
            system=system_prompt,
            prompt=user_prompt,
        )

        if not sources:
            return response.strip()

        source_lines = []

        for source in sources:
            source_lines.append(
                f"- [Fuente {source['number']}] "
                f"`{source['source']}` "
                f"(fragmento {source['chunk_index'] + 1})"
            )

        return (
            f"{response.strip()}\n\n"
            "📚 **Fragmentos recuperados:**\n"
            + "\n".join(source_lines)
        )
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
                    "file_name": metadata.get(
                        "source",
                        "Fuente desconocida",
                    ),
                    "kind": metadata.get(
                        "kind",
                        "document",
                    ),
                    "chunks": 0,
                }

            documents[document_id]["chunks"] += 1

        return list(documents.values())

    def get_last_sources(
        self,
        channel_id: int,
    ) -> list[dict]:
        return self.last_sources.get(
            channel_id,
            [],
        )