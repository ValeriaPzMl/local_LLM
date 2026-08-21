from pathlib import Path

from src.agents.coding_agent import CodingAgent


agent = CodingAgent(
    workspace=Path(".")
)

for change in agent.list_pending_changes():
    print(
        change.id,
        change.file_path,
        change.description,
    )