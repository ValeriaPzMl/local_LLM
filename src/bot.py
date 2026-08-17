import asyncio
from pathlib import Path

import discord

from src.config import (
    DISCORD_TOKEN,
    EMBEDDING_MODEL,
    OLLAMA_MODEL,
    VISION_MODEL,
)
from src.loaders import DocumentLoader
from src.memory import ChannelMemory
from src.ollama_client import OllamaClient
from src.rag import RAGService


SYSTEM_PROMPT = """
Eres ComputahMind, un asistente para proyectos.

Cada canal de Discord representa un proyecto diferente.

Debes utilizar el historial reciente y la memoria permanente
del proyecto para responder.

No inventes recuerdos. Cuando una respuesta dependa de información
anterior, revisa la memoria y el historial proporcionados.

Responde siempre en español de forma clara y útil.
""".strip()

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}

class ComputahMindBot:
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True

        self.client = discord.Client(intents=intents)
        self.ollama = OllamaClient()
        self.memory = ChannelMemory("data/memory.db")
        self.rag = RAGService()

        # Impide procesar dos solicitudes simultáneamente
        # en el mismo canal.
        self.channel_locks: dict[int, asyncio.Lock] = {}

        @self.client.event
        async def on_ready():
            print(
                "ComputahMind está conectado como "
                f"{self.client.user}"
            )

        @self.client.event
        async def on_message(message):
            if message.author.bot:
                return

            channel_id = message.channel.id
            user_message = message.content.strip()

            if user_message.lower() == "!docs":
                documents = self.rag.list_documents(
                    channel_id=channel_id,
                )

                if not documents:
                    await message.channel.send(
                        "Este canal no tiene documentos "
                        "ni imágenes indexadas."
                    )
                    return

                lines = []

                for document in documents:
                    document_id = document["document_id"]
                    short_id = document_id[:12]
                    file_name = document["file_name"]
                    chunks = document["chunks"]
                    kind = document.get(
                        "kind",
                        "document",
                    )

                    icon = (
                        "🖼️"
                        if kind == "image"
                        else "📄"
                    )

                    lines.append(
                        f"{icon} `{short_id}` — "
                        f"`{file_name}` "
                        f"({chunks} fragmentos)"
                    )

                await self.send_long_message(
                    channel=message.channel,
                    text=(
                        "📚 **Contenido indexado en este canal:**\n"
                        + "\n".join(lines)
                        + "\n\nPara borrar uno usa:\n"
                        "`!borrar-doc ID`"
                    ),
                )
                return
            if user_message.lower() == "!fuentes":
                sources = self.rag.get_last_sources(
                    channel_id=channel_id,
                )

                if not sources:
                    await message.channel.send(
                        "Todavía no hay fuentes recuperadas en este canal."
                    )
                    return

                lines = []

                for source in sources:
                    distance = source.get("distance")
                    semantic = source.get("semantic_score")
                    lexical = source.get("lexical_score")
                    final_score = source.get("final_score")

                    distance_text = (
                        f"{distance:.4f}"
                        if isinstance(distance, (int, float))
                        else "N/D"
                    )

                    semantic_text = (
                        f"{semantic:.3f}"
                        if isinstance(semantic, (int, float))
                        else "N/D"
                    )

                    lexical_text = (
                        f"{lexical:.3f}"
                        if isinstance(lexical, (int, float))
                        else "N/D"
                    )

                    final_text = (
                        f"{final_score:.3f}"
                        if isinstance(final_score, (int, float))
                        else "N/D"
                    )

                    lines.append(
                        f"📄 [Fuente {source['number']}] "
                        f"`{source['source']}`\n"
                        f"   Fragmento: {source['chunk_index'] + 1}\n"
                        f"   Distancia: {distance_text}\n"
                        f"   Semántica: {semantic_text}\n"
                        f"   Léxica: {lexical_text}\n"
                        f"   Score final: {final_text}"
                    )

                await self.send_long_message(
                    channel=message.channel,
                    text="🔎 **Últimas fuentes recuperadas:**\n\n"
                    + "\n\n".join(lines),
                )
                return

            if user_message.lower().startswith("!borrar-doc"):
                parts = user_message.split(
                    maxsplit=1
                )

                if len(parts) < 2:
                    await message.channel.send(
                        "Uso correcto:\n"
                        "`!borrar-doc ID`"
                    )
                    return

                requested_id = parts[1].strip()

                documents = self.rag.list_documents(
                    channel_id=channel_id,
                )

                matches = [
                    document
                    for document in documents
                    if document["document_id"].startswith(
                        requested_id
                    )
                ]

                if not matches:
                    await message.channel.send(
                        "No encontré ningún documento "
                        "con ese ID."
                    )
                    return

                if len(matches) > 1:
                    await message.channel.send(
                        "Ese ID coincide con varios documentos. "
                        "Escribe más caracteres del ID."
                    )
                    return

                document = matches[0]

                self.rag.delete_document(
                    channel_id=channel_id,
                    document_id=document["document_id"],
                )

                await message.channel.send(
                    f"🗑️ Eliminé `{document['file_name']}` "
                    "del RAG de este canal."
                )
                return
            
            if user_message.lower() == "!estado":
                documents = self.rag.list_documents(
                    channel_id=channel_id,
                )

                message_count = self.memory.count_messages(
                    channel_id
                )

                facts = self.memory.get_facts(
                    channel_id=channel_id,
                    limit=100,
                )

                document_count = sum(
                    1
                    for document in documents
                    if document.get("kind") != "image"
                )

                image_count = sum(
                    1
                    for document in documents
                    if document.get("kind") == "image"
                )

                total_chunks = sum(
                    document.get("chunks", 0)
                    for document in documents
                )

                collection = self.rag.get_collection(
                    channel_id
                )

                vector_count = collection.count()

                status_text = (
                    "⚙️ **Estado de ComputahMind**\n\n"

                    "🤖 **Modelos**\n"
                    f"💬 Texto: `{OLLAMA_MODEL}`\n"
                    f"🖼️ Visión: `{VISION_MODEL}`\n"
                    f"🧬 Embeddings: `{EMBEDDING_MODEL}`\n\n"

                    "📚 **RAG del canal**\n"
                    f"📄 Documentos: `{document_count}`\n"
                    f"🖼️ Imágenes: `{image_count}`\n"
                    f"📦 Fuentes totales: `{len(documents)}`\n"
                    f"🧩 Fragmentos: `{total_chunks}`\n"
                    f"🔢 Vectores en Chroma: `{vector_count}`\n\n"

                    "🧠 **Memoria**\n"
                    f"💭 Mensajes guardados: `{message_count}`\n"
                    f"📌 Hechos permanentes: `{len(facts)}`\n\n"

                    "💾 **Servicios**\n"
                    "✅ SQLite conectado\n"
                    "✅ ChromaDB conectado\n"
                    "✅ Ollama configurado"
                )

                await self.send_long_message(
                    channel=message.channel,
                    text=status_text,
                )
                return
            if user_message.lower() in {
                "!formatos",
                "!documentos",
                "!ayuda-archivos",
            }:
                document_formats = ", ".join(
                    sorted(DocumentLoader.SUPPORTED_EXTENSIONS)
                )

                image_formats = (
                    ".png, .jpg, .jpeg, .webp, .bmp"
                )

                await message.channel.send(
                    "📂 **Formatos compatibles**\n\n"
                    "**Documentos para RAG:**\n"
                    f"`{document_formats}`\n\n"
                    "**Imágenes para análisis visual:**\n"
                    f"`{image_formats}`\n\n"
                    "Puedes adjuntar una imagen sola para obtener una "
                    "descripción, o acompañarla con una pregunta, por ejemplo:\n"
                    "`¿Qué información contiene esta gráfica?`"
                )
                return

            # Muestra cuántos mensajes hay guardados.
            if user_message.lower() == "!memoria":
                total = self.memory.count_messages(channel_id)

                await message.channel.send(
                    f"Este canal tiene {total} mensajes "
                    "guardados en su historial."
                )
                return

            # Muestra los hechos permanentes del proyecto.
            if user_message.lower() == "!hechos":
                facts = self.memory.get_facts(
                    channel_id=channel_id,
                    limit=50,
                )

                if not facts:
                    await message.channel.send(
                        "No hay hechos permanentes guardados "
                        "en este canal."
                    )
                    return

                facts_text = "\n".join(
                    f"- {fact}"
                    for fact in facts
                )

                await self.send_long_message(
                    channel=message.channel,
                    text=(
                        "🧠 **Memoria permanente del proyecto:**\n"
                        f"{facts_text}"
                    ),
                )
                return

            # Borra solamente el historial de conversación.
            if user_message.lower() == "!olvidar":
                self.memory.clear_channel(channel_id)

                await message.channel.send(
                    "He borrado el historial de conversación "
                    "de este canal."
                )
                return

            # Borra solamente los hechos permanentes.
            if user_message.lower() == "!olvidar-hechos":
                self.memory.clear_facts(channel_id)

                await message.channel.send(
                    "He borrado los hechos permanentes "
                    "de este canal."
                )
                return

            # Si no hay texto ni adjuntos, no hay nada
            # que procesar.
            if not user_message and not message.attachments:
                return

            lock = self.channel_locks.setdefault(
                channel_id,
                asyncio.Lock(),
            )

            async with lock:
                # Primero procesa los documentos.
                # Así se puede adjuntar un archivo y escribir una
                # pregunta en el mismo mensaje.
                image_prompt_used = False

                if message.attachments:
                    image_prompt_used = await self.process_attachments(
                        message
                    )

                # Si el texto fue usado como pregunta sobre una imagen,
                # no lo enviamos otra vez como mensaje normal.
                if user_message and not image_prompt_used:
                    await self.process_message(
                        message=message,
                        user_message=user_message,
                    )

    async def process_attachments(
        self,
        message: discord.Message,
    ) -> bool:
        channel_id = message.channel.id
        user_prompt = message.content.strip()

        upload_directory = (
            Path("data/uploads") / str(channel_id)
        )

        upload_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_prompt_used = False

        for attachment in message.attachments:
            extension = Path(
                attachment.filename
            ).suffix.lower()

            safe_name = Path(
                attachment.filename
            ).name

            file_path = upload_directory / (
                f"{attachment.id}_{safe_name}"
            )

            # Procesamiento de imágenes.
            if extension in IMAGE_EXTENSIONS:
                try:
                    await message.channel.send(
                        f"🖼️ Analizando `{safe_name}`..."
                    )

                    await attachment.save(file_path)

                    if user_prompt:
                        vision_prompt = f"""
                    Observa la imagen y responde únicamente a la petición del usuario.

                    Petición:
                    {user_prompt}

                    Sé preciso y no describas elementos que no sean necesarios.
                    """.strip()
                    else:
                        vision_prompt = """
                    Describe brevemente la imagen e identifica cualquier texto,
                    tabla, gráfica o dato importante visible.
                    """.strip()
                   

                    async with message.channel.typing():
                        visual_analysis = await self.ollama.analyze_image(
                            image_path=str(file_path),
                            prompt=vision_prompt,
                        )

                        image_hash = self.rag.calculate_file_hash(
                            str(file_path)
                        )

                        index_result = await self.rag.index_text(
                            channel_id=channel_id,
                            text=visual_analysis,
                            source_name=safe_name,
                            document_id=image_hash,
                            kind="image",
                        )

                    await self.send_long_message(
                        channel=message.channel,
                        text=visual_analysis,
                    )

                    if index_result["status"] == "already_exists":
                        await message.channel.send(
                            f"ℹ️ La imagen `{safe_name}` "
                            "ya estaba guardada en el RAG."
                        )
                    else:
                        await message.channel.send(
                            f"✅ Guardé el análisis de `{safe_name}` "
                            "en el RAG.\n"
                            f"Se crearon "
                            f"{index_result['chunks']} fragmentos."
                        )

                    image_prompt_used = bool(user_prompt)

                except Exception as error:
                    file_path.unlink(
                        missing_ok=True
                    )

                    print(
                        f"Error analizando {safe_name}: "
                        f"{type(error).__name__}: {error}"
                    )

                    await message.channel.send(
                        f"❌ No pude analizar `{safe_name}`.\n"
                        f"`{error}`"
                    )

                continue

            # Procesamiento de documentos para RAG.
            if extension not in DocumentLoader.SUPPORTED_EXTENSIONS:
                supported_documents = ", ".join(
                    sorted(
                        DocumentLoader.SUPPORTED_EXTENSIONS
                    )
                )

                supported_images = ", ".join(
                    sorted(IMAGE_EXTENSIONS)
                )

                await message.channel.send(
                    f"⚠️ `{safe_name}` no fue procesado.\n"
                    f"Documentos: `{supported_documents}`\n"
                    f"Imágenes: `{supported_images}`"
                )
                continue

            try:
                await message.channel.send(
                    f"📄 Descargando `{safe_name}`..."
                )

                await attachment.save(file_path)

                await message.channel.send(
                    f"🧠 Indexando `{safe_name}`..."
                )

                async with message.channel.typing():
                    result = await self.rag.index_document(
                        channel_id=channel_id,
                        file_path=str(file_path),
                    )

                if result["status"] == "already_exists":
                    file_path.unlink(missing_ok=True)

                    await message.channel.send(
                        f"ℹ️ `{safe_name}` ya estaba indexado "
                        "en este canal."
                    )
                    continue

                await message.channel.send(
                    f"✅ `{safe_name}` fue indexado correctamente.\n"
                    f"Se guardaron {result['chunks']} fragmentos."
                )

            except Exception as error:
                file_path.unlink(missing_ok=True)

                print(
                    f"Error indexando {safe_name}: "
                    f"{type(error).__name__}: {error}"
                )

                await message.channel.send(
                    f"❌ No pude indexar `{safe_name}`.\n"
                    f"`{error}`"
                )

        return image_prompt_used

    async def process_message(
        self,
        message: discord.Message,
        user_message: str,
    ) -> None:
        channel_id = message.channel.id
        author_name = message.author.display_name

        stored_user_message = (
            f"{author_name}: {user_message}"
        )

        # Guarda el mensaje del usuario en el historial.
        self.memory.add_message(
            channel_id=channel_id,
            role="user",
            content=stored_user_message,
        )

        # Intenta extraer información permanente del mensaje.
        try:
            extracted_facts = await self.ollama.extract_facts(
                user_message
            )

            for fact in extracted_facts:
                self.memory.add_fact(
                    channel_id=channel_id,
                    fact=fact,
                )

            if extracted_facts:
                print(
                    "Hechos guardados: "
                    f"{extracted_facts}"
                )

        except Exception as error:
            # La extracción de hechos no debe impedir que el bot
            # responda al mensaje.
            print(
                "No se pudieron extraer hechos: "
                f"{type(error).__name__}: {error}"
            )

        # Recupera el historial reciente.
        history = self.memory.get_messages(
            channel_id=channel_id,
            limit=30,
        )

        # Recupera la memoria permanente.
        facts = self.memory.get_facts(
            channel_id=channel_id,
            limit=50,
        )

        if facts:
            facts_text = "\n".join(
                f"- {fact}"
                for fact in facts
            )
        else:
            facts_text = (
                "No hay hechos permanentes guardados."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}\n\n"
                    "MEMORIA PERMANENTE DEL PROYECTO:\n"
                    f"{facts_text}"
                ),
            },
            *history,
        ]

        print(
            f"[Canal: {message.channel.name}] "
            f"{stored_user_message}"
        )

        try:
            async with message.channel.typing():
                documents = self.rag.list_documents(
                    channel_id=channel_id,
                )

                if documents:
                    print(
                        "Usando RAG con "
                        f"{len(documents)} documento(s)."
                    )

                    answer = await self.rag.answer_question(
                        channel_id=channel_id,
                        question=user_message,
                        ollama_client=self.ollama,
                        facts=facts,
                        history=history,
                        limit=5,
                    )

                else:
                    print(
                        "No hay documentos indexados. "
                        "Usando conversación normal."
                    )

                    answer = await self.ollama.chat(
                        messages
                    )

        except Exception as error:
            print(
                "Error generando respuesta: "
                f"{type(error).__name__}: {error}"
            )

            await message.channel.send(
                "❌ Ocurrió un error al generar la respuesta.\n"
                f"`{error}`"
            )
            return

        # Guarda la respuesta del asistente.
        self.memory.add_message(
            channel_id=channel_id,
            role="assistant",
            content=answer,
        )

        await self.send_long_message(
            channel=message.channel,
            text=answer,
        )

    async def send_long_message(
        self,
        channel,
        text: str,
        limit: int = 1900,
    ) -> None:
        if not text:
            await channel.send(
                "El modelo no generó ninguna respuesta."
            )
            return

        for start in range(0, len(text), limit):
            chunk = text[start:start + limit]
            await channel.send(chunk)

    def run(self):
        self.client.run(DISCORD_TOKEN)