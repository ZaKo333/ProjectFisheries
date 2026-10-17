"""会话存储（SQLite）的单元测试。

为什么单独测这一层
------------------
``AppState`` 原来是四个内存字典，键只有 ``pond_id``：两个浏览器用默认塘号
就会互相看见对方的数据，进程一重启又全丢。改成 SQLite 之后，隔离靠
``session_id``、持久化靠文件，两件事都不再依赖「进程还活着」这个前提。

这一层是纯逻辑（不经过 HTTP），所以断言可以写得很直接：同一会话能读回、
不同会话看不见、换一个实例（模拟重启）数据还在、超上限丢最旧的。
"""

from __future__ import annotations

import json

import pytest

from fisheries.api.store import (
    SESSION_COOKIE,
    SESSION_HEADER,
    SessionStore,
    is_valid_session_id,
    new_session_id,
)

A = "session-aaaa-1111"
B = "session-bbbb-2222"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "api_state.sqlite3"


@pytest.fixture
def store(db_path):
    s = SessionStore(db_path, buffer_max=5, history_max=3)
    yield s
    s.close()


def readings(count: int, start: int = 0) -> list[dict]:
    return [
        {"timestamp": f"2026-01-01T00:{i:02d}:00", "dissolved_oxygen": 6.0 - i * 0.1}
        for i in range(start, start + count)
    ]


# --------------------------------------------------------------------------- #
# 会话标识
# --------------------------------------------------------------------------- #


class TestSessionId:
    def test_generated_ids_are_valid_and_unique(self):
        ids = {new_session_id() for _ in range(50)}
        assert len(ids) == 50
        assert all(is_valid_session_id(v) for v in ids)

    @pytest.mark.parametrize(
        "value",
        ["", "short", "has space in it", "带中文的会话标识", "x" * 65, "a/b/c/d/e"],
    )
    def test_rejects_obviously_bad_values(self, value):
        assert not is_valid_session_id(value)

    def test_header_and_cookie_names_are_stable(self):
        # 前端与对接文档都引用了这两个名字，改名就是破坏性变更。
        assert SESSION_HEADER == "X-Session-Id"
        assert SESSION_COOKIE == "fisheries_session"


# --------------------------------------------------------------------------- #
# 读数缓冲
# --------------------------------------------------------------------------- #


class TestReadings:
    def test_append_returns_total_and_time_bounds(self, store):
        total, first, last = store.append_readings(A, "P1", readings(3))
        assert total == 3
        assert first == "2026-01-01T00:00:00"
        assert last == "2026-01-01T00:02:00"

    def test_second_append_accumulates(self, store):
        store.append_readings(A, "P1", readings(2))
        total, _, _ = store.append_readings(A, "P1", readings(2, start=2))
        assert total == 4
        assert len(store.get_readings(A, "P1")) == 4

    def test_replace_drops_the_old_series(self, store):
        store.append_readings(A, "P1", readings(3))
        total, first, _ = store.append_readings(A, "P1", readings(1, start=7), replace=True)
        assert total == 1
        assert first == "2026-01-01T00:07:00"

    def test_sessions_do_not_see_each_other(self, store):
        store.append_readings(A, "P1", readings(3))
        assert store.get_readings(B, "P1") == []

    def test_ponds_do_not_see_each_other(self, store):
        store.append_readings(A, "P1", readings(3))
        assert store.get_readings(A, "P2") == []

    def test_oldest_rows_are_dropped_beyond_the_limit(self, store):
        store.append_readings(A, "P1", readings(8))  # buffer_max=5
        kept = store.get_readings(A, "P1")
        assert len(kept) == 5
        # 留下的是最新 5 条（时间戳 03..07），不是最旧的。
        assert [r["timestamp"] for r in kept] == [f"2026-01-01T00:0{i}:00" for i in range(3, 8)]

    def test_json_round_trip_keeps_types(self, store):
        payload = [{"timestamp": "t", "dissolved_oxygen": 5.25, "note": "中文"}]
        store.append_readings(A, "P1", payload)
        assert store.get_readings(A, "P1") == payload


# --------------------------------------------------------------------------- #
# 评估历史
# --------------------------------------------------------------------------- #


class TestHistory:
    def test_record_and_read_back_in_order(self, store):
        for i in range(2):
            store.record(A, "P1", {"risk_level": "正常", "round": i})
        items = store.get_history(A, "P1")
        assert [i["round"] for i in items] == [0, 1]

    def test_history_is_isolated_per_session(self, store):
        store.record(A, "P1", {"risk_level": "高危"})
        assert store.get_history(B, "P1") == []

    def test_history_is_truncated_to_the_limit(self, store):
        for i in range(6):  # history_max=3
            store.record(A, "P1", {"round": i})
        assert [i["round"] for i in store.get_history(A, "P1")] == [3, 4, 5]

    def test_pydantic_like_objects_are_dumped(self, store):
        class Fake:
            def model_dump(self, *, mode: str) -> dict:
                assert mode == "json"
                return {"risk_level": "预警", "score": 42.5}

        store.record(A, "P1", Fake())
        assert store.get_history(A, "P1")[0]["risk_level"] == "预警"

    def test_unserializable_values_do_not_crash(self, store):
        store.record(A, "P1", {"path": object()})  # default=str 兜住
        assert isinstance(store.get_history(A, "P1")[0]["path"], str)


# --------------------------------------------------------------------------- #
# 上传文件
# --------------------------------------------------------------------------- #


class TestUploads:
    def test_lookup_scoped_to_its_own_session(self, store, tmp_path):
        image = tmp_path / "a.jpg"
        image.write_bytes(b"x")
        store.register_upload(A, "upload-1", image)
        assert store.lookup_upload(A, "upload-1") == image
        assert store.lookup_upload(B, "upload-1") is None

    def test_registering_the_same_id_twice_updates_the_row(self, store, tmp_path):
        first, second = tmp_path / "one.jpg", tmp_path / "two.jpg"
        store.register_upload(A, "upload-2", first)
        store.register_upload(A, "upload-2", second)
        assert store.lookup_upload(A, "upload-2") == second
        assert store.counts()["uploads"] == 1

    def test_unknown_id_is_none(self, store):
        assert store.lookup_upload(A, "nope") is None


# --------------------------------------------------------------------------- #
# 持久化与清理
# --------------------------------------------------------------------------- #


class TestPersistence:
    def test_data_survives_a_new_instance(self, db_path):
        with SessionStore(db_path, buffer_max=10, history_max=10) as first:
            first.append_readings(A, "P1", readings(2))
            first.record(A, "P1", {"risk_level": "正常"})
            first.register_upload(A, "upload-3", db_path.parent / "x.jpg")
        # 换实例 = 模拟进程重启：数据必须还在，否则「持久化」名不副实。
        with SessionStore(db_path, buffer_max=10, history_max=10) as second:
            assert len(second.get_readings(A, "P1")) == 2
            assert len(second.get_history(A, "P1")) == 1
            assert second.lookup_upload(A, "upload-3") is not None

    def test_lazy_creation_leaves_no_file_until_used(self, db_path):
        s = SessionStore(db_path)
        try:
            assert not db_path.exists(), "只构造不该建库文件"
            s.counts()
            assert db_path.exists()
        finally:
            s.close()

    def test_clear_one_session_keeps_the_other(self, store):
        store.append_readings(A, "P1", readings(2))
        store.append_readings(B, "P1", readings(2))
        removed = store.clear(A)
        assert removed == 2
        assert store.get_readings(A, "P1") == []
        assert len(store.get_readings(B, "P1")) == 2

    def test_clear_without_argument_wipes_everything(self, store):
        store.append_readings(A, "P1", readings(2))
        store.append_readings(B, "P1", readings(2))
        store.record(A, "P1", {"x": 1})
        assert store.clear() == 5
        assert store.counts() == {"readings": 0, "history": 0, "uploads": 0}

    def test_counts_reports_each_channel(self, store):
        store.append_readings(A, "P1", readings(3))
        store.record(A, "P1", {"x": 1})
        assert store.counts() == {"readings": 3, "history": 1, "uploads": 0}

    def test_payload_is_stored_as_json_text(self, db_path):
        # 直接看一眼落盘内容：字段以 JSON 存，而不是 pickle 之类无法外部核对的东西。
        with SessionStore(db_path) as s:
            s.append_readings(A, "P1", [{"dissolved_oxygen": 6.1}])
            raw = s.connection.execute("SELECT payload FROM readings").fetchone()["payload"]
        assert json.loads(raw)["dissolved_oxygen"] == pytest.approx(6.1)
