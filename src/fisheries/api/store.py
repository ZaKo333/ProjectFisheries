"""会话数据的落盘实现：每个会话一套独立的读数 / 历史 / 上传记录。

为什么不再用内存字典
--------------------
接口层原来把 ``buffers`` / ``history`` / ``uploads`` 三个 dict 挂在 ``AppState`` 上，
键只有 ``pond_id``。这带来两个真实问题：

1. **同一个 ``pond_id`` 跨请求串数据。** ``pond_id`` 是调用方自己起的名字
   （默认值就是 ``"unknown"``），两个浏览器只要都用默认值，看到的就是同一份缓冲：
   A 灌进 24 小时读数，B 点「按缓冲评估」时评估的是 A 的塘。
2. **进程重启即清空。** 内存态没有任何持久化，重启后「按缓冲评估」和历史全部丢失；
   开发时用 ``--reload`` 更是每改一次代码就清一次。

现在改为 SQLite：主键是 ``(session_id, pond_id)``，写盘即持久。
会话标识由 ``app.py`` 的中间件给出（请求头 ``X-Session-Id`` 优先，其次 cookie，
都没有就新生成一个并 Set-Cookie）。

边界
----
``session_id`` 只做**隔离**用，不做鉴权：知道别人 session_id 的人仍然能读到那份数据。
要防越权得在上面叠认证，不在本模块范围内。
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from typing_extensions import Self

__all__ = [
    "SESSION_COOKIE",
    "SESSION_HEADER",
    "SessionStore",
    "is_valid_session_id",
    "new_session_id",
]

#: 客户端可显式指定的会话标识头。
SESSION_HEADER = "X-Session-Id"
#: 浏览器侧的会话 cookie 名。
SESSION_COOKIE = "fisheries_session"

_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    pond_id TEXT NOT NULL,
    timestamp TEXT,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_readings_key ON readings(session_id, pond_id, seq);

CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    pond_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_key ON history(session_id, pond_id, id);

CREATE TABLE IF NOT EXISTS uploads (
    upload_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_uploads_session ON uploads(session_id);
"""


def new_session_id() -> str:
    """生成一个新的会话标识（32 位十六进制，满足 :data:`_SESSION_RE`）。"""
    return uuid.uuid4().hex


def is_valid_session_id(value: str) -> bool:
    """校验客户端给的会话标识：8~64 位的字母/数字/下划线/短横线。

    限定字符集不是为了安全，而是为了让它不可能被当成 SQL、路径或日志分隔符使用；
    长度上限是为了不让人用超长头把库撑大。
    """
    return bool(_SESSION_RE.match(value))


class SessionStore:
    """SQLite 支撑的会话数据仓库。

    一个实例对应一个数据库文件。连接在首次访问时建立（读空库不值得建文件，
    这样只跑 ``GET /health`` 的进程不会在磁盘上留东西）。
    """

    def __init__(
        self,
        path: Path | str,
        *,
        buffer_max: int = 5000,
        history_max: int = 200,
    ) -> None:
        self._memory = str(path) == ":memory:"
        self.path: Path | None = None if self._memory else Path(path)
        self.buffer_max = max(1, int(buffer_max))
        self.history_max = max(1, int(history_max))
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------ #
    # 连接
    # ------------------------------------------------------------------ #

    @property
    def connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                if self.path is not None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                target = ":memory:" if self.path is None else str(self.path)
                conn = sqlite3.connect(
                    target,
                    check_same_thread=False,
                    isolation_level=None,
                    timeout=5.0,
                )
                conn.row_factory = sqlite3.Row
                # WAL：一个进程写、多个进程读时不会互相把对方锁死。
                # 内存库不支持 WAL，跳过。
                if not self._memory:
                    conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")
                conn.executescript(_SCHEMA)
                self._conn = conn
            return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # 读数缓冲
    # ------------------------------------------------------------------ #

    def append_readings(
        self,
        session_id: str,
        pond_id: str,
        readings: list[dict[str, Any]],
        *,
        replace: bool = False,
    ) -> tuple[int, str | None, str | None]:
        """写入读数，返回 ``(当前总数, 首条时间戳, 末条时间戳)``。

        ``replace=True`` 先清空这一对 ``(session, pond)`` 的旧数据——语义是
        「重新导入这个塘的历史」，而不是在旧数据后面接着灌。
        超过 ``buffer_max`` 时按时间顺序丢最旧的，避免演示进程挂久了把盘写满。
        """
        payloads = [
            (
                session_id,
                pond_id,
                item.get("timestamp") if isinstance(item, dict) else None,
                json.dumps(item, ensure_ascii=False, default=str),
            )
            for item in readings
        ]
        with self._lock:
            conn = self.connection
            conn.execute("BEGIN")
            try:
                if replace:
                    conn.execute(
                        "DELETE FROM readings WHERE session_id = ? AND pond_id = ?",
                        (session_id, pond_id),
                    )
                if payloads:
                    conn.executemany(
                        "INSERT INTO readings (session_id, pond_id, timestamp, payload)"
                        " VALUES (?, ?, ?, ?)",
                        payloads,
                    )
                conn.execute(
                    "DELETE FROM readings WHERE seq IN ("
                    " SELECT seq FROM readings WHERE session_id = ? AND pond_id = ?"
                    " ORDER BY seq DESC LIMIT -1 OFFSET ?)",
                    (session_id, pond_id, self.buffer_max),
                )
                row = conn.execute(
                    "SELECT COUNT(*) AS n,"
                    " (SELECT timestamp FROM readings WHERE session_id = ? AND pond_id = ?"
                    "  ORDER BY seq LIMIT 1) AS first,"
                    " (SELECT timestamp FROM readings WHERE session_id = ? AND pond_id = ?"
                    "  ORDER BY seq DESC LIMIT 1) AS last"
                    " FROM readings WHERE session_id = ? AND pond_id = ?",
                    (session_id, pond_id, session_id, pond_id, session_id, pond_id),
                ).fetchone()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return int(row["n"]), row["first"], row["last"]

    def get_readings(self, session_id: str, pond_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT payload FROM readings WHERE session_id = ? AND pond_id = ?"
                " ORDER BY seq",
                (session_id, pond_id),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    # ------------------------------------------------------------------ #
    # 评估历史
    # ------------------------------------------------------------------ #

    def record(self, session_id: str, pond_id: str, result: Any) -> None:
        """追加一条评估结果（精简版，轨迹与知识块原文不入库）。"""
        if hasattr(result, "model_dump"):
            result = result.model_dump(mode="json")
        payload = json.dumps(result, ensure_ascii=False, default=str)
        with self._lock:
            conn = self.connection
            conn.execute("BEGIN")
            try:
                conn.execute(
                    "INSERT INTO history (session_id, pond_id, created_at, payload)"
                    " VALUES (?, ?, ?, ?)",
                    (session_id, pond_id, time.time(), payload),
                )
                conn.execute(
                    "DELETE FROM history WHERE id IN ("
                    " SELECT id FROM history WHERE session_id = ? AND pond_id = ?"
                    " ORDER BY id DESC LIMIT -1 OFFSET ?)",
                    (session_id, pond_id, self.history_max),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def get_history(self, session_id: str, pond_id: str) -> list[Any]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT payload FROM history WHERE session_id = ? AND pond_id = ?"
                " ORDER BY id",
                (session_id, pond_id),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    # ------------------------------------------------------------------ #
    # 上传文件
    # ------------------------------------------------------------------ #

    def register_upload(self, session_id: str, upload_id: str, path: Path | str) -> None:
        with self._lock:
            self.connection.execute(
                "INSERT INTO uploads (upload_id, session_id, path, created_at)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(upload_id) DO UPDATE SET"
                "  session_id = excluded.session_id, path = excluded.path,"
                "  created_at = excluded.created_at",
                (upload_id, session_id, str(path), time.time()),
            )

    def lookup_upload(self, session_id: str, upload_id: str) -> Path | None:
        """按会话查上传文件：别人的 ``upload_id`` 查不到，等价于「不存在」。"""
        with self._lock:
            row = self.connection.execute(
                "SELECT path FROM uploads WHERE upload_id = ? AND session_id = ?",
                (upload_id, session_id),
            ).fetchone()
        return Path(row["path"]) if row else None

    # ------------------------------------------------------------------ #
    # 清理
    # ------------------------------------------------------------------ #

    def clear(self, session_id: str | None = None) -> int:
        """删掉一个会话（或全部）的数据，返回删掉的行数。"""
        with self._lock:
            conn = self.connection
            total = 0
            conn.execute("BEGIN")
            try:
                for table in ("readings", "history", "uploads"):
                    if session_id is None:
                        cur = conn.execute(f"DELETE FROM {table}")
                    else:
                        cur = conn.execute(
                            f"DELETE FROM {table} WHERE session_id = ?",
                            (session_id,),
                        )
                    total += cur.rowcount or 0
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return total

    def counts(self) -> dict[str, int]:
        """每个通道有多少行——排障用（比如「另一个会话的数据是不是写进来了」）。"""
        with self._lock:
            conn = self.connection
            rows = {
                table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                for table in ("readings", "history", "uploads")
            }
        return rows
