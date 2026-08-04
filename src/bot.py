import asyncio
from pathlib import Path

import discord

from src.config import DISCORD_TOKEN
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
                if message.attachments:
                    await self.process_attachments(message)

                if user_message:
                    await self.process_message(
                        message=message,
                        user_message=user_message,
                    )

    async def process_attachments(
        self,
        message: discord.Message,
    ) -> None:
        channel_id = message.channel.id

        upload_directory = (
            Path("data/uploads") / str(channel_id)
        )

        upload_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for attachment in message.attachments:
            extension = Path(
                attachment.filename
            ).suffix.lower()

            if extension not in DocumentLoader.SUPPORTED_EXTENSIONS:
                supported = ", ".join(
                    sorted(
                        DocumentLoader.SUPPORTED_EXTENSIONS
                    )
                )

                await message.channel.send(
                    f"⚠️ `{attachment.filename}` no fue procesado.\n"
                    f"Formatos compatibles: `{supported}`"
                )
                continue

            # Evita nombres que intenten incluir rutas,
            # por ejemplo ../../archivo.pdf.
            safe_name = Path(
                attachment.filename
            ).name

            # El ID de Discord evita colisiones cuando dos archivos
            # tienen el mismo nombre.
            file_path = upload_directory / (
                f"{attachment.id}_{safe_name}"
            )

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
                    # No necesitamos conservar una copia duplicada.
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
                    f"Error: `{error}`"
                )

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