

import sqlite3
from pathlib import Path


class ChannelMemory:
    def __init__(self, database_path: str = "data/memory.db"):
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row

        self._create_tables()

    def _create_tables(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS project_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(channel_id, fact)
            )
            """
        )
        self.connection.commit()

    def add_message(
        self,
        channel_id: int,
        role: str,
        content: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO messages (channel_id, role, content)
            VALUES (?, ?, ?)
            """,
            (
                str(channel_id),
                role,
                content,
            ),
        )

        self.connection.commit()

    def get_messages(
        self,
        channel_id: int,
        limit: int = 30,
    ) -> list[dict[str, str]]:
        cursor = self.connection.execute(
            """
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM messages
                WHERE channel_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (
                str(channel_id),
                limit,
            ),
        )

        return [
            {
                "role": row["role"],
                "content": row["content"],
            }
            for row in cursor.fetchall()
        ]

    def count_messages(self, channel_id: int) -> int:
        cursor = self.connection.execute(
            """
            SELECT COUNT(*) AS total
            FROM messages
            WHERE channel_id = ?
            """,
            (str(channel_id),),
        )

        row = cursor.fetchone()
        return int(row["total"])

    def clear_channel(self, channel_id: int) -> None:
        self.connection.execute(
            """
            DELETE FROM messages
            WHERE channel_id = ?
            """,
            (str(channel_id),),
        )

        self.connection.commit()
    def add_fact(
        self,
        channel_id: int,
        fact: str,
    ) -> None:
        fact = fact.strip()

        if not fact:
            return

        self.connection.execute(
            """
            INSERT OR IGNORE INTO project_facts (
                channel_id,
                fact
            )
            VALUES (?, ?)
            """,
            (
                str(channel_id),
                fact,
            ),
        )

        self.connection.commit()


    def get_facts(
        self,
        channel_id: int,
        limit: int = 50,
    ) -> list[str]:
        cursor = self.connection.execute(
            """
            SELECT fact
            FROM project_facts
            WHERE channel_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                str(channel_id),
                limit,
            ),
        )

        return [
            row["fact"]
            for row in cursor.fetchall()
        ]


    def clear_facts(
        self,
        channel_id: int,
    ) -> None:
        self.connection.execute(
            """
            DELETE FROM project_facts
            WHERE channel_id = ?
            """,
            (str(channel_id),),
        )

        self.connection.commit()