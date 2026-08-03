import asyncio

from src.ollama_client import OllamaClient
from src.rag import RAGService


CHANNEL_ID = 123456


async def main():
    rag = RAGService()
    ollama = OllamaClient()

    answer = await rag.answer_question(
        channel_id=CHANNEL_ID,
        question="¿Cuál es el tema principal del documento?",
        ollama_client=ollama,
        limit=5,
    )

    print("\nRESPUESTA:\n")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
#data/uploads/Code.The.Hidden.Language.of.Computer.Hardware.and.Software.2nd.Edition.2022.7.pdf