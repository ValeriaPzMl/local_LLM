import sqlite3
from datetime import datetime
from pathlib import Path

from src.agents.run_request import RunRequest


class RunStore:
    def __init__(
        self,
        database_path: str = "data/memory.db",
    ):
        path = Path(database_path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row

        self._create_tables()

    def _create_tables(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
                id TEXT PRIMARY KEY,
                workspace TEXT NOT NULL,
                command TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL,
                approved INTEGER NOT NULL DEFAULT 0,
                rejected INTEGER NOT NULL DEFAULT 0,
                executed INTEGER NOT NULL DEFAULT 0,
                result TEXT
            )
            """
        )

        self.connection.commit()

    def save(
        self,
        workspace: Path,
        run: RunRequest,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO agent_runs (
                id,
                workspace,
                command,
                description,
                created_at,
                approved,
                rejected,
                executed,
                result
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                str(workspace.resolve()),
                run.command,
                run.description,
                run.created_at.isoformat(),
                int(run.approved),
                int(run.rejected),
                int(run.executed),
                run.result,
            ),
        )

        self.connection.commit()

    def get(
        self,
        run_id: str,
    ) -> RunRequest | None:
        cursor = self.connection.execute(
            """
            SELECT *
            FROM agent_runs
            WHERE id = ?
            """,
            (run_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return RunRequest(
            id=row["id"],
            command=row["command"],
            description=row["description"],
            created_at=datetime.fromisoformat(
                row["created_at"]
            ),
            approved=bool(row["approved"]),
            rejected=bool(row["rejected"]),
            executed=bool(row["executed"]),
            result=row["result"],
        )
    def list_pending(
        self,
        workspace: Path | None = None,
    ) -> list[RunRequest]:
        if workspace is None:
            cursor = self.connection.execute(
                """
                SELECT *
                FROM agent_runs
                WHERE executed = 0
                AND rejected = 0
                ORDER BY created_at DESC
                """
            )
        else:
            cursor = self.connection.execute(
                """
                SELECT *
                FROM agent_runs
                WHERE workspace = ?
                AND executed = 0
                AND rejected = 0
                ORDER BY created_at DESC
                """,
                (
                    str(workspace.resolve()),
                ),
            )

        return [
            RunRequest(
                id=row["id"],
                command=row["command"],
                description=row["description"],
                created_at=datetime.fromisoformat(
                    row["created_at"]
                ),
                approved=bool(row["approved"]),
                rejected=bool(row["rejected"]),
                executed=bool(row["executed"]),
                result=row["result"],
            )
            for row in cursor.fetchall()
        ]