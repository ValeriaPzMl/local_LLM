from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4


@dataclass
class ChangeRequest:
    id: str
    workspace: Path
    file_path: str

    original_content: str
    new_content: str

    description: str

    created_at: datetime

    approved: bool = False
    rejected: bool = False
    applied: bool = False


def create_change_request(
    workspace: Path,
    file_path: str,
    original_content: str,
    new_content: str,
    description: str,
) -> ChangeRequest:
    return ChangeRequest(
        id=uuid4().hex[:8],
        workspace=workspace.resolve(),
        file_path=file_path,
        original_content=original_content,
        new_content=new_content,
        description=description,
        created_at=datetime.now(),
    )