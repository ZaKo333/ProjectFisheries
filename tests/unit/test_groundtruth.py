"""M8 第三组验证（人工标签）的机械部分测试。

真实标注必须由人来做，所以这里只验证：抽样可复现且分层、清单能写出再读回、
非法标注会被拒、不足条数时不出指标。**没有一条断言要求 human_level 有值。**
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fisheries.evaluation import groundtruth, ponds, runner

UTC_START = datetime(2021, 6, 19, tzinfo=UTC)


def make_sample(
    pond_id: str = "P1",
    *,
    minute: int = 0,
    step_minutes: int = 10,
    day: int = 19,
    water_temp: float = 25.0,
    ph: float = 7.5,
    dissolved_oxygen: float = 6.0,
    ammonia_n: float = 0.1,
    nitrite_n: float | None = None,
) -> ponds.PondSample:
    """与 `test_evaluation.py` 里同名辅助函数保持一致（10 分钟一个聚合点）。"""
    start = UTC_START + timedelta(days=day - 19)
    return ponds.PondSample(
        pond_id=pond_id,
        sampled_at=start + timedelta(minutes=step_minutes * minute),
        water_temp=water_temp,
        ph=ph,
        dissolved_oxygen=dissolved_oxygen,
        ammonia_n=ammonia_n,
        nitrite_n=nitrite_n,
        n_raw=1,
    )


class TestGroundtruth:
    """第三组验证：人工标注的真值集。

    这里只测「抽样—清单—评测」这条链路的机械部分；真实标注不是测试能造的，
    所以没有任何一条断言要求 `human_level` 有值。
    """

    @staticmethod
    def _long(
        pond_id: str = "P1",
        *,
        count: int = 145,
        dissolved_oxygen: float = 6.0,
        day: int = 19,
    ) -> list[ponds.PondSample]:
        return [
            make_sample(pond_id, minute=i, dissolved_oxygen=dissolved_oxygen, day=day)
            for i in range(count)
        ]

    @staticmethod
    def _labeled(window: ponds.PondWindow, level: str) -> groundtruth.LabeledWindow:
        return groundtruth.LabeledWindow(
            window_id="GT0001",
            pond_id=window.pond_id,
            started_at=window.started_at,
            ended_at=window.ended_at,
            values={},
            readings=window.readings,
            human_level=level,
            annotator="测试",
        )

    @staticmethod
    def _manual(tmp_path: Path, body: str) -> None:
        (tmp_path / "windows.json").write_text('{"examples": []}', encoding="utf-8")
        header = ",".join(groundtruth.LABEL_COLUMNS)
        (tmp_path / "labels.csv").write_text(f"{header}\n{body}\n", encoding="utf-8")

    # -- 抽样 --------------------------------------------------------------

    def test_sampling_is_deterministic(self) -> None:
        by_pond = {"P1": self._long("P1"), "P2": self._long("P2", day=20)}
        first = groundtruth.sample_windows(by_pond, n=4, seed=7, min_gap_hours=0.0)
        second = groundtruth.sample_windows(by_pond, n=4, seed=7, min_gap_hours=0.0)
        assert [row.window_id for row in first] == [row.window_id for row in second]
        assert [(row.pond_id, row.started_at) for row in first] == [
            (row.pond_id, row.started_at) for row in second
        ]

    def test_sampling_covers_oxygen_strata(self) -> None:
        by_pond = {"低": self._long("低", dissolved_oxygen=1.0), "高": self._long("高")}
        rows = groundtruth.sample_windows(by_pond, n=2, seed=3, min_gap_hours=0.0)
        levels = {row.values["dissolved_oxygen"] for row in rows}
        assert min(levels) < 3.0 and max(levels) >= 5.0

    def test_sampling_respects_the_per_pond_gap(self) -> None:
        by_pond = {"P1": self._long("P1", count=145)}  # 8 个窗口，跨度 24 小时
        rows = groundtruth.sample_windows(by_pond, n=8, seed=1, min_gap_hours=24.0)
        assert len(rows) == 1, "相隔 3 小时的窗口不该同时入选"

    def test_sampling_rejects_non_positive_n(self) -> None:
        with pytest.raises(ValueError):
            groundtruth.sample_windows({"P1": self._long()}, n=0)

    # -- 清单 --------------------------------------------------------------

    def test_write_read_roundtrip_leaves_human_columns_blank(self, tmp_path: Path) -> None:
        by_pond = {"P1": self._long("P1")}
        rows = groundtruth.sample_windows(by_pond, n=3, seed=1, min_gap_hours=0.0)
        paths = groundtruth.write_sample(rows, tmp_path)

        assert paths["labels"].exists() and paths["windows"].exists()
        assert paths["manifest"].exists()
        # BOM：标注人双击用 Excel 打开，中文列名与「正常/预警/高危」不能乱码
        assert paths["labels"].read_bytes().startswith(b"\xef\xbb\xbf")

        back = groundtruth.read_sample(tmp_path)
        assert [row.window_id for row in back] == [row.window_id for row in rows]
        assert all(not row.labeled for row in back)
        assert back[0].readings == rows[0].readings

    def test_read_rejects_an_unknown_level(self, tmp_path: Path) -> None:
        self._manual(
            tmp_path,
            "GT0001,P1,2021-06-19T00:00:00,2021-06-19T03:00:00,18,"
            "25,7.5,6.0,0.1,,0.001,严重,张三,",
        )
        with pytest.raises(ValueError, match="不在"):
            groundtruth.read_sample(tmp_path)

    def test_read_rejects_annotator_without_a_level(self, tmp_path: Path) -> None:
        self._manual(
            tmp_path,
            "GT0001,P1,2021-06-19T00:00:00,2021-06-19T03:00:00,18,"
            "25,7.5,6.0,0.1,,0.001,,张三,",
        )
        with pytest.raises(ValueError, match="没填 human_level"):
            groundtruth.read_sample(tmp_path)

    def test_read_requires_the_sample_files(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            groundtruth.read_sample(tmp_path)

    def test_isolation_rejects_training_directories(self, tmp_path: Path) -> None:
        groundtruth.check_isolation(tmp_path)  # 临时目录可以
        with pytest.raises(ValueError, match="不能放在"):
            groundtruth.check_isolation(groundtruth.PROJECT_ROOT / "data" / "derived" / "gt")
        with pytest.raises(ValueError, match="不能放在"):
            groundtruth.check_isolation(groundtruth.PROJECT_ROOT / "data" / "datasets" / "gt")

    # -- 非离子氨参考值 ----------------------------------------------------

    def test_free_ammonia_is_a_concentration_not_a_fraction(self) -> None:
        """回归：`unionized_ammonia()` 先返回占比、后返回浓度，别拿错位置。"""
        window = groundtruth.WindowSample(
            window_id="GT0001",
            pond_id="P1",
            started_at=datetime(2021, 6, 19, tzinfo=UTC),
            ended_at=datetime(2021, 6, 19, 3, tzinfo=UTC),
            readings=(),
            values={"water_temp": 24.5556, "ph": 8.1704, "ammonia_n": 0.0004},
        )
        free = window.free_ammonia()
        assert free is not None
        # 总氨氮 0.0004 mg/L、占比约 7.5% ⇒ 非离子氨约 3e-5 mg/L（不是 0.075）
        assert free < 0.0001

    # -- 评测 --------------------------------------------------------------

    def test_pending_labels_produce_no_metrics(self, tmp_path: Path) -> None:
        by_pond = {"P1": self._long("P1")}
        window = next(iter(ponds.iter_windows(by_pond)))
        rows = [
            groundtruth.LabeledWindow(
                window_id="GT0001",
                pond_id=window.pond_id,
                started_at=window.started_at,
                ended_at=window.ended_at,
                values={},
                readings=window.readings,
            )
        ]
        result = groundtruth.evaluate(rows, min_labels=20)
        assert result["status"] == "pending_labels"
        assert result["n_labeled"] == 0
        assert "accuracy" not in result, "没标注就不该有准确率这个键"
        assert result["metric_kind"] == "human_label_accuracy"

    def test_labels_matching_the_engine_score_full_marks(self) -> None:
        by_pond = {"P1": self._long("P1")}
        window = next(iter(ponds.iter_windows(by_pond)))
        samples, _ = runner.run_engine([window], species="鲤科", horizon_hours=72)
        engine_level = samples[0].risk_level
        assert engine_level, "引擎对这段平稳数据应给出一个等级"

        result = groundtruth.evaluate([self._labeled(window, engine_level)], min_labels=1)
        assert result["status"] == "ok"
        assert result["n_scored"] == 1
        assert result["accuracy"] == 1.0
        # 只有一类样本时 κ 的分母为 0，按约定回 0——不因此判失败
        assert isinstance(result["kappa"], float)
        assert result["metric_kind"] == "human_label_accuracy"
        assert "人工标签" in result["markdown"]

    def test_labeled_rows_filters_blank_levels(self) -> None:
        def build(level: str) -> groundtruth.LabeledWindow:
            return groundtruth.LabeledWindow(
                window_id="GT1",
                pond_id="P1",
                started_at=datetime(2021, 6, 19, tzinfo=UTC),
                ended_at=datetime(2021, 6, 19, 3, tzinfo=UTC),
                values={},
                readings=(),
                human_level=level,
            )

        assert [row.human_level for row in groundtruth.labeled_rows([build(""), build("高危")])] == [
            "高危"
        ]
