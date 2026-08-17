import os

from dotenv import load_dotenv


load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3.5:4b",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text",
)

VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "qwen3-vl:4b",
)
CODING_MODEL = os.getenv(
    "CODING_MODEL",
    "qwen2.5-coder:7b",
)

if not DISCORD_TOKEN:
    raise ValueError(
        "No se encontró DISCORD_TOKEN en el archivo .env"
    )