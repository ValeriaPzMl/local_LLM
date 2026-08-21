import sqlite3
from datetime import datetime
from pathlib import Path

from src.agents.change_request import ChangeRequest


class ChangeStore:
    def __init__(
        self,
        database_path: str = "data/memory.db",
    ):
        path = Path(database_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            path
        )

        self.connection.row_factory = sqlite3.Row

        self._create_tables()

    def _create_tables(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_changes (
                id TEXT PRIMARY KEY,
                workspace TEXT NOT NULL,
                file_path TEXT NOT NULL,
                original_content TEXT NOT NULL,
                new_content TEXT NOT NULL,
                description TEXT NOT NULL,
                created_at TEXT NOT NULL,
                approved INTEGER NOT NULL DEFAULT 0,
                rejected INTEGER NOT NULL DEFAULT 0,
                applied INTEGER NOT NULL DEFAULT 0
            )
            """
        )

        self.connection.commit()

    def save(
        self,
        change: ChangeRequest,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO agent_changes (
                id,
                workspace,
                file_path,
                original_content,
                new_content,
                description,
                created_at,
                approved,
                rejected,
                applied
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                change.id,
                str(change.workspace),
                change.file_path,
                change.original_content,
                change.new_content,
                change.description,
                change.created_at.isoformat(),
                int(change.approved),
                int(change.rejected),
                int(change.applied),
            ),
        )

        self.connection.commit()

    def get(
        self,
        change_id: str,
    ) -> ChangeRequest | None:
        cursor = self.connection.execute(
            """
            SELECT *
            FROM agent_changes
            WHERE id = ?
            """,
            (change_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_change(
            row
        )

    def list_pending(
        self,
        workspace: Path | None = None,
    ) -> list[ChangeRequest]:
        if workspace is None:
            cursor = self.connection.execute(
                """
                SELECT *
                FROM agent_changes
                WHERE applied = 0
                AND rejected = 0
                ORDER BY created_at DESC
                """
            )
        else:
            cursor = self.connection.execute(
                """
                SELECT *
                FROM agent_changes
                WHERE workspace = ?
                AND applied = 0
                AND rejected = 0
                ORDER BY created_at DESC
                """,
                (
                    str(workspace.resolve()),
                ),
            )

        return [
            self._row_to_change(row)
            for row in cursor.fetchall()
        ]

    def update_status(
        self,
        change: ChangeRequest,
    ) -> None:
        self.connection.execute(
            """
            UPDATE agent_changes
            SET approved = ?,
                rejected = ?,
                applied = ?
            WHERE id = ?
            """,
            (
                int(change.approved),
                int(change.rejected),
                int(change.applied),
                change.id,
            ),
        )

        self.connection.commit()

    @staticmethod
    def _row_to_change(
        row: sqlite3.Row,
    ) -> ChangeRequest:
        return ChangeRequest(
            id=row["id"],
            workspace=Path(
                row["workspace"]
            ),
            file_path=row["file_path"],
            original_content=row[
                "original_content"
            ],
            new_content=row[
                "new_content"
            ],
            description=row[
                "description"
            ],
            created_at=datetime.fromisoformat(
                row["created_at"]
            ),
            approved=bool(
                row["approved"]
            ),
            rejected=bool(
                row["rejected"]
            ),
            applied=bool(
                row["applied"]
            ),
        )