from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4


@dataclass
class RunRequest:
    id: str
    command: str
    description: str
    created_at: datetime
    approved: bool = False
    rejected: bool = False
    executed: bool = False
    result: str | None = None


def create_run_request(
    command: str,
    description: str,
) -> RunRequest:
    return RunRequest(
        id=uuid4().hex[:8],
        command=command,
        description=description,
        created_at=datetime.now(),
    )