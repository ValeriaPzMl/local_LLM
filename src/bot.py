import asyncio
from pathlib import Path

import discord

from src.config import DISCORD_TOKEN
from src.memory import ChannelMemory
from src.ollama_client import OllamaClient
from src.rag import RAGService


SYSTEM_PROMPT = """
Eres ComputahMind, un asistente para proyectos.

Cada canal de Discord representa un proyecto diferente.
Debes usar el historial proporcionado para recordar información
mencionada anteriormente dentro de ese canal.

No inventes recuerdos. Cuando una respuesta dependa del historial,
revisa los mensajes anteriores antes de responder.

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
        # dentro del mismo canal.
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

            # Los comandos solo se procesan cuando hay texto.
            if user_message.lower() == "!memoria":
                total = self.memory.count_messages(channel_id)

                await message.channel.send(
                    f"Este canal tiene {total} mensajes guardados "
                    "en su memoria."
                )
                return

            if user_message.lower() == "!olvidar":
                self.memory.clear_channel(channel_id)

                await message.channel.send(
                    "He borrado la memoria de este canal."
                )
                return

            # Si no hay texto ni archivos adjuntos, no hay nada
            # que procesar.
            if not user_message and not message.attachments:
                return

            lock = self.channel_locks.setdefault(
                channel_id,
                asyncio.Lock(),
            )

            async with lock:
                # Primero procesamos los documentos adjuntos.
                # Así una persona puede subir un PDF y hacer una
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

            if extension != ".pdf":
                await message.channel.send(
                    f"⚠️ `{attachment.filename}` no fue procesado. "
                    "Por ahora solamente acepto archivos PDF."
                )
                continue

            # Path(...).name evita que un nombre de archivo incluya
            # rutas peligrosas como ../../archivo.pdf.
            safe_name = Path(
                attachment.filename
            ).name

            # El ID del archivo de Discord evita colisiones entre
            # archivos que tengan el mismo nombre.
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
                    # El contenido ya está en Chroma, por lo que
                    # no necesitamos conservar otra copia local.
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

        # Guardamos primero el mensaje del usuario.
        self.memory.add_message(
            channel_id=channel_id,
            role="user",
            content=stored_user_message,
        )

        history = self.memory.get_messages(
            channel_id=channel_id,
            limit=30,
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
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
                        f"Usando RAG con "
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

                    answer = await self.ollama.chat(messages)

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