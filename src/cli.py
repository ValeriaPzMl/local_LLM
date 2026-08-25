import argparse
import asyncio
from pathlib import Path

from src.agents.coding_agent import CodingAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="ComputahMind Coding Agent"
    )

    parser.add_argument(
        "--project",
        "-p",
        type=Path,
        default=Path.cwd(),
        help="Ruta del proyecto donde trabajará el agente.",
    )

    return parser.parse_args()


async def cli_main():
    args = parse_args()

    workspace = args.project.expanduser().resolve()

    if not workspace.exists():
        print(
            f"❌ La ruta no existe: {workspace}"
        )
        return

    if not workspace.is_dir():
        print(
            f"❌ La ruta no es una carpeta: {workspace}"
        )
        return

    agent = CodingAgent(
        workspace=workspace
    )

    print("======================================")
    print("       ComputahMind Coding Agent")
    print("======================================")
    print()
    print(f"📁 Workspace: {workspace}")
    print()
    print("Escribe una tarea para el agente.")
    print("Escribe 'salir' para terminar.")
    print()

    while True:
        try:
            task = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not task:
            continue

        if task.lower() in {
            "salir",
            "exit",
            "quit",
        }:
            break

        try:
            answer = await agent.run(task)

            print()
            print("=" * 70)
            print("RESPUESTA DEL AGENTE")
            print("=" * 70)
            print(answer)
            print()

        except Exception as error:
            print()
            print("❌ Error:")
            print(error)
            print()


def main():
    asyncio.run(cli_main())


if __name__ == "__main__":
    main()