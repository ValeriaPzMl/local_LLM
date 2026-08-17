import asyncio

import aiohttp

from src.config import (
    OLLAMA_MODEL,
    OLLAMA_URL,
    VISION_MODEL,
)

import base64
from pathlib import Path


class OllamaClient:
    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_URL,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=300)

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
        }

        if system:
            payload["system"] = system

        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
            ) as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as response:
                    data = await response.json()

                    if response.status != 200:
                        error_message = data.get(
                            "error",
                            "Error desconocido",
                        )

                        raise RuntimeError(
                            f"Ollama respondió con HTTP "
                            f"{response.status}: {error_message}"
                        )

            content = data.get("response", "").strip()

            if not content:
                print("Respuesta completa de Ollama:")
                print(data)

                raise RuntimeError(
                    "Ollama no devolvió una respuesta final."
                )

            return content

        except aiohttp.ClientConnectorError as error:
            raise RuntimeError(
                "No se pudo conectar con Ollama."
            ) from error

        except asyncio.TimeoutError as error:
            raise RuntimeError(
                "Ollama tardó demasiado en responder."
            ) from error

    async def chat(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
        }

        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
            ) as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:
                    data = await response.json()

                    if response.status != 200:
                        error_message = data.get(
                            "error",
                            "Error desconocido",
                        )

                        raise RuntimeError(
                            f"Ollama respondió con HTTP "
                            f"{response.status}: {error_message}"
                        )

            message = data.get("message", {})
            content = message.get("content", "").strip()

            if not content:
                print("Respuesta completa de Ollama:")
                print(data)

                raise RuntimeError(
                    "Ollama no devolvió una respuesta final."
                )

            return content

        except aiohttp.ClientConnectorError as error:
            raise RuntimeError(
                "No se pudo conectar con Ollama."
            ) from error

        except asyncio.TimeoutError as error:
            raise RuntimeError(
                "Ollama tardó demasiado en responder."
            ) from error
    async def extract_facts(
        self,
        user_message: str,
    ) -> list[str]:
        system = """
    Analiza el mensaje del usuario y extrae únicamente hechos
    importantes y duraderos sobre su proyecto.

    Ejemplos de hechos útiles:
    - Nombre del proyecto.
    - Lenguajes o frameworks usados.
    - Objetivos del proyecto.
    - Decisiones técnicas.
    - Preferencias permanentes.
    - Requisitos importantes.

    No extraigas:
    - Saludos.
    - Preguntas.
    - Comentarios temporales.
    - Información incierta.
    - Datos inventados.

    Devuelve únicamente una lista, un hecho por línea.
    Cada línea debe comenzar con "- ".
    Si no hay hechos importantes, responde exactamente:
    NINGUNO
    """.strip()

        prompt = f"""
    MENSAJE:

    {user_message}
    """.strip()

        result = await self.generate(
            prompt=prompt,
            system=system,
        )

        if result.strip().upper() == "NINGUNO":
            return []

        facts = []

        for line in result.splitlines():
            line = line.strip()

            if line.startswith("-"):
                fact = line[1:].strip()

                if fact:
                    facts.append(fact)

        return facts
    async def analyze_image(
        self,
        image_path: str,
        prompt: str | None = None,
    ) -> str:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"No existe la imagen: {path}"
            )

        image_base64 = base64.b64encode(
            path.read_bytes()
        ).decode("utf-8")

        user_prompt = (
            prompt.strip()
            if prompt and prompt.strip()
            else (
                "Describe detalladamente esta imagen. "
                "Identifica texto, objetos, diagramas, tablas "
                "y cualquier información importante."
            )
        )

        payload = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [image_base64],
                }
            ],
            "stream": False,
            "think": False,
            "options": {
                "num_predict": 1200,
                "temperature": 0.2,
            },
        }

        try:
            async with aiohttp.ClientSession(
                timeout=self.timeout,
            ) as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                ) as response:
                    data = await response.json()

                    if response.status != 200:
                        error_message = data.get(
                            "error",
                            "Error desconocido",
                        )

                        raise RuntimeError(
                            f"Ollama respondió con HTTP "
                            f"{response.status}: {error_message}"
                        )

            message = data.get("message", {})
            content = message.get("content", "").strip()

            if not content:
                thinking = message.get("thinking", "").strip()
                done_reason = data.get("done_reason", "desconocido")

                print(
                    "El modelo visual no produjo respuesta final. "
                    f"Motivo: {done_reason}"
                )

                if thinking:
                    print(
                        "El modelo generó razonamiento, "
                        "pero agotó la salida antes de responder."
                    )

                raise RuntimeError(
                    "El modelo visual agotó su límite antes de producir "
                    "la respuesta final. Intenta una pregunta más directa."
                )

            return content

        except aiohttp.ClientConnectorError as error:
            raise RuntimeError(
                "No se pudo conectar con Ollama."
            ) from error

        except asyncio.TimeoutError as error:
            raise RuntimeError(
                "El modelo visual tardó demasiado."
            ) from error