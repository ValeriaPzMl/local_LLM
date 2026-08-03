import asyncio

import aiohttp

from src.config import EMBEDDING_MODEL, OLLAMA_URL


class OllamaEmbeddings:
    def __init__(self):
        self.base_url = OLLAMA_URL.rstrip("/")
        self.model = EMBEDDING_MODEL

    async def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        url = f"{self.base_url}/api/embed"

        payload = {
            "model": self.model,
            "input": texts,
        }

        timeout = aiohttp.ClientTimeout(total=300)

        try:
            async with aiohttp.ClientSession(
                timeout=timeout
            ) as session:
                async with session.post(
                    url,
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            embeddings = data.get("embeddings")

            if not isinstance(embeddings, list):
                raise ValueError(
                    "Ollama no devolvió una lista de embeddings."
                )

            return embeddings

        except aiohttp.ClientConnectorError as error:
            raise RuntimeError(
                "No se pudo conectar con Ollama. "
                "Comprueba que esté ejecutándose."
            ) from error

        except aiohttp.ClientResponseError as error:
            raise RuntimeError(
                "Ollama respondió con un error HTTP: "
                f"{error.status}"
            ) from error

        except asyncio.TimeoutError as error:
            raise RuntimeError(
                "Ollama tardó demasiado en generar los embeddings."
            ) from error

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:
        embeddings = await self.embed_texts([text])

        if not embeddings:
            raise ValueError(
                "No se generó ningún embedding."
            )

        return embeddings[0]