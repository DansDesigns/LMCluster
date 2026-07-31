"""Saved conversations.

Chats are kept on disk so that a reply which took four minutes to produce
is not lost to a browser refresh or a locked phone screen. That is a real
concern here rather than a theoretical one: a large model spread over a
home network is slow enough that you will often start something and come
back to it.

The database lives in the same folder as the node's identity and cluster
key, so it survives reinstalling the code.
"""

import os
import sqlite3
import threading
import time
import uuid

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id      TEXT PRIMARY KEY,
    title   TEXT NOT NULL,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    chat_id TEXT NOT NULL,
    seq     INTEGER NOT NULL,
    ts      REAL NOT NULL,
    role    TEXT NOT NULL,
    content TEXT NOT NULL,
    PRIMARY KEY (chat_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updated);
"""


class Store:
    def __init__(self, state_dir: str):
        os.makedirs(state_dir, exist_ok=True)
        self.path = os.path.join(state_dir, "lmcluster.db")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    def new_chat(self, first_prompt: str) -> str:
        """Start a chat, titled with the opening question.

        Using the first prompt as the title means the list is browsable
        without asking the model to summarise anything, which would cost
        another slow round trip.
        """
        chat_id = uuid.uuid4().hex[:12]
        title = " ".join(first_prompt.split())[:120] or "Untitled"
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO chats (id, title, created, updated) "
                "VALUES (?, ?, ?, ?)", (chat_id, title, now, now))
        return chat_id

    def add_message(self, chat_id: str, role: str, content: str):
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages "
                "WHERE chat_id = ?", (chat_id,)).fetchone()
            self._conn.execute(
                "INSERT INTO messages (chat_id, seq, ts, role, content) "
                "VALUES (?, ?, ?, ?, ?)",
                (chat_id, row[0], time.time(), role, content))
            self._conn.execute("UPDATE chats SET updated = ? WHERE id = ?",
                               (time.time(), chat_id))

    def chats(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.id, c.title, c.created, c.updated, "
                "       (SELECT COUNT(*) FROM messages m WHERE m.chat_id=c.id) "
                "FROM chats c ORDER BY c.updated DESC LIMIT ?",
                (limit,)).fetchall()
        return [{"id": r[0], "title": r[1], "created": r[2],
                 "updated": r[3], "messages": r[4]} for r in rows]

    def messages(self, chat_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, ts FROM messages WHERE chat_id = ? "
                "ORDER BY seq", (chat_id,)).fetchall()
        return [{"role": r[0], "content": r[1], "ts": r[2]} for r in rows]

    def delete_chat(self, chat_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM chats WHERE id = ?",
                                     (chat_id,))
            self._conn.execute("DELETE FROM messages WHERE chat_id = ?",
                               (chat_id,))
        return cur.rowcount > 0
