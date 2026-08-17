import asyncio
from pathlib import Path

from src.agents.coding_agent import CodingAgent


async def main():
    agent = CodingAgent(
        workspace=Path("."),
    )

    answer = await agent.run(
    "Revisa los cambios exactos que hice en "
    "src/rag.py y explícame qué modifiqué."
)

    print("\n" + "=" * 70)
    print("RESPUESTA DEL AGENTE")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())