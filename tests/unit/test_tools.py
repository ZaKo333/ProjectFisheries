"""工具层单元测试：water_risk_predict / fish_disease_detect。

覆盖三类用例：
    1. 契约用例  —— schema 注册、返回体 JSON 可序列化、字段齐全
    2. 边界用例  —— 阈值表每个分档的临界值（安全/警戒/危险 三带的端点）
    3. 异常用例  —— 非法入参、缺失图片、伪图片文件

运行：
    cd D:\\设置\\项目\\Agent\\project
    .venv\\Scripts\\python.exe -m pytest tests/unit/test_tools.py -v
"""

from __future__ import annotations

import json
import struct
import zlib
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from fisheries.tools import TOOL_HANDLERS, TOOL_SCHEMAS, fish_disease_detect, water_risk_predict
from fisheries.tools.fish_disease import (
    DISEASE_LABELS,
    LABEL_META,
    RAW_CLASS_TO_LABEL,
    _mock_scores,
)
from fisheries.tools.water_risk import SPECIES_PROFILES, _severity

# --------------------------------------------------------------------------- #
# 测试夹具
# --------------------------------------------------------------------------- #

T0 = datetime(2026, 10, 14, 0, 0, 0)


def make_series(n: int = 24, **overrides) -> list[dict]:
    """构造一段平稳的基准水质序列（全部处于适宜区间）。"""
    base = {
        "water_temp": 26.0,
        "ph": 7.4,
        "dissolved_oxygen": 6.5,
        "ammonia_n": 0.10,
        "nitrite_n": 0.05,
    }
    base.update(overrides)
    return [
        {"timestamp": (T0 + timedelta(hours=i)).isoformat(), **base}
        for i in range(n)
    ]


def write_min_png(path: Path, width: int = 8, height: int = 8) -> Path:
    """用标准库生成一张合法的最小 PNG，避免测试依赖真实图片文件。"""
    def chunk(typ: bytes, data: bytes) -> bytes:
        body = typ + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes((20, 120, 200)) * width for _ in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
    return path


@pytest.fixture
def png_file(tmp_path: Path) -> Path:
    return write_min_png(tmp_path / "gill_01.png")


# --------------------------------------------------------------------------- #
# 1. 契约测试
# --------------------------------------------------------------------------- #

class TestToolContract:
    def test_both_tools_registered(self):
        assert set(TOOL_SCHEMAS) == {"water_risk_predict", "fish_disease_detect"}
        assert set(TOOL_HANDLERS) == set(TOOL_SCHEMAS)

    @pytest.mark.parametrize("name", ["water_risk_predict", "fish_disease_detect"])
    def test_schema_shape(self, name: str):
        fn = TOOL_SCHEMAS[name]["function"]
        assert TOOL_SCHEMAS[name]["type"] == "function"
        assert fn["name"] == name
        assert isinstance(fn["description"], str) and len(fn["description"]) > 30
        assert fn["parameters"]["type"] == "object"
        assert fn["parameters"]["required"], "每个工具至少要有一个必填参数"
        # 必填参数必须在 properties 中声明
        for req in fn["parameters"]["required"]:
            assert req in fn["parameters"]["properties"]

    def test_schema_is_json_serializable(self):
        """schema 要能直接塞进 HTTP 请求体。"""
        assert json.loads(json.dumps(TOOL_SCHEMAS, ensure_ascii=False))

    def test_water_output_fields(self):
        out = water_risk_predict(make_series())
        for field in ("tool", "status", "risk_level", "risk_score",
                      "degradation_summary", "trends", "projections",
                      "data_quality", "rule_hits", "redline_hits"):
            assert field in out, f"返回体缺少字段 {field}"
        assert isinstance(out["degradation_summary"], str)
        assert out["degradation_summary"], "恶化描述不能为空"
        json.dumps(out, ensure_ascii=False)  # 必须可序列化

    def test_fish_output_fields(self, png_file: Path):
        out = fish_disease_detect(str(png_file), force_mock=True)
        for field in ("tool", "status", "disease_type", "confidence",
                      "all_scores", "model_version", "degraded_mode", "note"):
            assert field in out, f"返回体缺少字段 {field}"
        assert out["degraded_mode"] is True
        assert out["model_version"].startswith("mock")
        json.dumps(out, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 2. 水质工具：阈值边界
# --------------------------------------------------------------------------- #

class TestWaterThresholdBoundaries:
    """逐一钉住 5 个指标的档位端点，防止以后改阈值时静默走偏。"""

    P = SPECIES_PROFILES["鲤科"]

    @pytest.mark.parametrize("value,expected", [
        (14.9, 1.0), (15.0, 0.5), (19.9, 0.5), (20.0, 0.0),
        (30.0, 0.0), (30.1, 0.5), (33.0, 0.5), (33.1, 1.0),
    ])
    def test_water_temp_bands(self, value: float, expected: float):
        assert _severity("water_temp", value, self.P)[0] == expected

    @pytest.mark.parametrize("value,expected", [
        (5.9, 1.0), (6.0, 0.5), (6.49, 0.5), (6.5, 0.0),
        (8.5, 0.0), (8.51, 0.5), (9.0, 0.5), (9.01, 1.0),
    ])
    def test_ph_bands(self, value: float, expected: float):
        assert _severity("ph", value, self.P)[0] == expected

    @pytest.mark.parametrize("value,expected", [
        (2.99, 1.0), (3.0, 0.5), (4.99, 0.5), (5.0, 0.0), (9.0, 0.0),
    ])
    def test_do_bands(self, value: float, expected: float):
        assert _severity("dissolved_oxygen", value, self.P)[0] == expected

    @pytest.mark.parametrize("value,expected", [
        (0.0, 0.0), (0.2, 0.0), (0.201, 0.5), (0.5, 0.5), (0.501, 1.0),
    ])
    def test_ammonia_bands(self, value: float, expected: float):
        assert _severity("ammonia_n", value, self.P)[0] == expected

    @pytest.mark.parametrize("value,expected", [
        (0.0, 0.0), (0.1, 0.0), (0.101, 0.5), (0.3, 0.5), (0.301, 1.0),
    ])
    def test_nitrite_bands(self, value: float, expected: float):
        assert _severity("nitrite_n", value, self.P)[0] == expected

    def test_species_changes_thresholds(self):
        """同样 26℃ 对鲤科适宜，对虾蟹则偏凉——证明品种档案确实生效。"""
        assert _severity("water_temp", 18.0, SPECIES_PROFILES["鲤科"])[0] == 0.5
        assert _severity("water_temp", 18.0, SPECIES_PROFILES["虾蟹"])[0] == 1.0


# --------------------------------------------------------------------------- #
# 3. 水质工具：等级判定
# --------------------------------------------------------------------------- #

class TestWaterRiskLevels:
    def test_all_optimal_is_normal(self):
        out = water_risk_predict(make_series(24))
        assert out["status"] == "ok"
        assert out["risk_level"] == "正常"
        assert out["risk_score"] == 0.0
        assert out["exceeded_indicators"] == []
        assert out["redline_hits"] == []
        assert out["limiting_factor"] is None

    def test_warning_band_gives_warning(self):
        """溶氧 4.0 落在警戒区间 → 预警，但不能升到高危。"""
        out = water_risk_predict(make_series(24, dissolved_oxygen=4.0))
        assert out["risk_level"] == "预警"
        assert out["limiting_factor"] == "dissolved_oxygen"
        assert "DO_BELOW_OPTIMAL" in out["rule_hits"]

    def test_single_danger_indicator_is_critical(self):
        """只有一项越界（溶氧 2.8）也必须判高危——不能被平均分稀释。"""
        out = water_risk_predict(make_series(24, dissolved_oxygen=2.8))
        assert out["risk_level"] == "高危"
        assert out["severity_map"]["dissolved_oxygen"] == 1.0
        # 其余四项仍适宜
        assert sum(1 for v in out["severity_map"].values() if v > 0) == 1

    def test_do_redline_hit(self):
        out = water_risk_predict(make_series(24, dissolved_oxygen=1.2))
        assert "DO_REDLINE" in out["redline_hits"]
        assert out["risk_level"] == "高危"

    def test_ammonia_redline_hit(self):
        out = water_risk_predict(make_series(24, ammonia_n=1.5))
        assert "NH3_REDLINE" in out["redline_hits"]
        assert out["risk_level"] == "高危"

    def test_do_crash_trend_detected(self):
        """最近 3 小时内溶氧从 6.5 掉到 3.5，跌幅 3.0 > 2.0，应命中骤降规则。"""
        series = make_series(24)
        for i, p in enumerate(series):
            p["dissolved_oxygen"] = 6.5 - max(0, i - 18) * 1.0
        out = water_risk_predict(series)
        assert "DO_CRASH_TREND" in out["redline_hits"]

    def test_persistence_amplifies_score(self):
        """连续超阈越久，persistence 越大，分数越高。"""
        short = water_risk_predict(make_series(3, ammonia_n=0.4))
        long = water_risk_predict(make_series(48, ammonia_n=0.4))
        assert long["risk_score"] > short["risk_score"]
        assert long["risk_score"] <= 100.0

    def test_historical_spike_does_not_stick(self):
        """3 天前氨氮尖峰、当前已恢复 → 不应继续命中红线。"""
        series = make_series(48, ammonia_n=0.1)
        series[0]["ammonia_n"] = 2.0
        out = water_risk_predict(series)
        assert "NH3_REDLINE" not in out["redline_hits"]
        assert out["risk_level"] == "正常"

    def test_unionized_ammonia_synergy_rule(self):
        """pH>8.5 且水温>28℃ 时氨氮超标，应额外命中分子氨风险规则。"""
        out = water_risk_predict(make_series(24, ph=8.8, water_temp=29.0, ammonia_n=0.35))
        assert "UNIONIZED_AMMONIA_RISK" in out["rule_hits"]


# --------------------------------------------------------------------------- #
# 4. 水质工具：权重重归一化与异常入参
# --------------------------------------------------------------------------- #

class TestWaterRiskRobustness:
    def test_missing_nitrite_renormalizes_weights(self):
        """亚硝酸盐探头缺失时，其余指标权重必须重新归一化到 1.0。"""
        series = make_series(24)
        for p in series:
            p.pop("nitrite_n")
        out = water_risk_predict(series)
        assert out["status"] == "ok"
        assert abs(sum(out["effective_weights"].values()) - 1.0) < 1e-5
        assert "亚硝酸盐" in out["data_quality"]["missing_indicators"]
        assert "nitrite_n" not in out["severity_map"]

    def test_explicit_null_nitrite(self):
        series = make_series(24, nitrite_n=None)
        out = water_risk_predict(series)
        assert out["status"] == "ok"

    def test_too_few_points_returns_error(self):
        out = water_risk_predict(make_series(2))
        assert out["status"] == "error"
        assert out["error_code"] == "INVALID_INPUT"
        assert "readings" in out["error_message"]

    def test_nan_rejected(self):
        series = make_series(24)
        series[5]["dissolved_oxygen"] = float("nan")
        out = water_risk_predict(series)
        assert out["status"] == "error"
        assert out["error_code"] == "INVALID_INPUT"

    def test_out_of_range_ph_rejected(self):
        out = water_risk_predict(make_series(24, ph=15.0))
        assert out["status"] == "error"
        assert "pH" in out["error_message"]

    def test_short_series_is_flagged(self):
        """样本不足时必须在 data_quality 里留下置信度提示。"""
        out = water_risk_predict(make_series(6))
        assert any("置信度" in w for w in out["data_quality"]["warnings"])
        assert "置信度提示" in out["degradation_summary"]

    def test_interval_inferred_from_timestamp(self):
        """带 timestamp 且间隔 2 小时 → 应自动推断而非用默认 60 分钟。"""
        series = make_series(24)
        for i, p in enumerate(series):
            p["timestamp"] = (T0 + timedelta(hours=2 * i)).isoformat()
        out = water_risk_predict(series, interval_minutes=60)
        assert out["data_quality"]["interval_minutes"] == 120
        assert out["data_quality"]["interval_source"] == "timestamp"

    def test_pond_id_passthrough(self):
        out = water_risk_predict(make_series(24), pond_id="POND-07")
        assert out["pond_id"] == "POND-07"


# --------------------------------------------------------------------------- #
# 5. 图像工具
# --------------------------------------------------------------------------- #

class TestFishDiseaseDetect:
    def test_mock_is_deterministic(self, png_file: Path):
        a = fish_disease_detect(str(png_file), force_mock=True)
        b = fish_disease_detect(str(png_file), force_mock=True)
        assert a["disease_type"] == b["disease_type"]
        assert a["confidence"] == b["confidence"]

    def test_label_always_in_enum(self, png_file: Path):
        out = fish_disease_detect(str(png_file), force_mock=True)
        assert out["disease_type"] in DISEASE_LABELS
        assert out["disease_group"] in {"细菌性", "真菌性", "其他", "健康"}
        assert 0.0 <= out["confidence"] <= 1.0

    def test_scores_sum_to_one(self, png_file: Path):
        out = fish_disease_detect(str(png_file), force_mock=True, top_k=5)
        assert abs(sum(out["all_scores"].values()) - 1.0) < 1e-3

    def test_top_class_matches_disease_type(self, png_file: Path):
        out = fish_disease_detect(str(png_file), force_mock=True, top_k=5)
        assert max(out["all_scores"], key=out["all_scores"].get) == out["disease_type"]

    def test_severity_weight_matches_registry(self, png_file: Path):
        out = fish_disease_detect(str(png_file), force_mock=True)
        assert out["severity_weight"] == LABEL_META[out["disease_type"]]["severity_weight"]
        assert out["is_healthy"] == (out["disease_type"] == "健康")

    def test_mock_scores_are_distribution(self, png_file: Path):
        winner, scores = _mock_scores(str(png_file))
        assert set(scores) == set(DISEASE_LABELS)
        assert abs(sum(scores.values()) - 1.0) < 1e-9
        assert max(scores, key=scores.get) == winner

    def test_missing_file_returns_error(self, tmp_path: Path):
        out = fish_disease_detect(str(tmp_path / "nope.jpg"))
        assert out["status"] == "error"
        assert out["error_code"] == "IMAGE_NOT_FOUND"
        assert out["disease_type"] is None

    def test_non_image_file_returns_error(self, tmp_path: Path):
        fake = tmp_path / "actually_a_text.jpg"
        fake.write_text("这不是图片", encoding="utf-8")
        out = fish_disease_detect(str(fake))
        assert out["status"] == "error"
        assert out["error_code"] == "UNSUPPORTED_IMAGE"

    def test_directory_returns_error(self, tmp_path: Path):
        out = fish_disease_detect(str(tmp_path))
        assert out["status"] == "error"
        assert out["error_code"] == "NOT_A_FILE"

    def test_image_meta_recorded(self, png_file: Path):
        out = fish_disease_detect(str(png_file), force_mock=True)
        assert out["image_meta"]["format"] == "png"
        assert out["image_meta"]["size_bytes"] > 0
        assert out["image_meta"]["is_remote"] is False

    def test_low_confidence_flagged(self, png_file: Path):
        """把阈值抬到 1.0，任何结果都应被标记为低置信度。"""
        out = fish_disease_detect(str(png_file), conf_threshold=1.0, force_mock=True)
        assert out["is_low_confidence"] is True
        assert "人工复核" in out["note"]

    def test_default_mock_source_is_hash(self, tmp_path: Path):
        """mock 路径默认绝不能从文件名读标签——否则无法自证没在偷看答案。

        这里显式 ``force_mock=True``：本条测的是 **mock 兜底行为**，
        不该因为本机恰好训练出了权重就改变或跳过。
        """
        folder = tmp_path / "Bacterial gill disease"
        folder.mkdir()
        img = write_min_png(folder / "sample.png")
        out = fish_disease_detect(str(img), force_mock=True)
        assert out["model_version"] == "mock-v0.1"
        assert "+folder_hint" not in out["note"]
        assert "文件名提示" not in out["note"]

    def test_real_model_ignores_folder_name(self, tmp_path: Path):
        """真实权重路径同样不许从文件名偷看标签。

        这是「没在偷看答案」的**正面证据**：同一张图放进类名完全不同的两个目录，
        真实模型必须给出逐字段一致的输出。没有训练权重时跳过
        （工程默认不带权重，见 README §12.1）。
        """
        probe = write_min_png(tmp_path / "probe.png")
        if fish_disease_detect(str(probe))["degraded_mode"]:
            pytest.skip("本机没有可用的 YOLO 权重（先跑 scripts/train_yolo_fish.py）")

        png_bytes = probe.read_bytes()
        results = []
        for dirname in ("Bacterial gill disease", "Healthy Fish"):
            folder = tmp_path / dirname
            folder.mkdir()
            img = folder / "same.png"
            img.write_bytes(png_bytes)
            results.append(fish_disease_detect(str(img)))

        for key in ("disease_type", "raw_class", "confidence", "model_version", "is_healthy"):
            assert results[0][key] == results[1][key], f"{key} 随目录名变了"
        assert results[0]["degraded_mode"] is False

    def test_folder_hint_opt_in_works(self, tmp_path: Path):
        """显式开启 folder_hint 后，标签来自路径，且必须被大声标注出来。"""
        folder = tmp_path / "Bacterial gill disease"
        folder.mkdir()
        img = write_min_png(folder / "sample.png")
        out = fish_disease_detect(str(img), force_mock=True, mock_source="folder_hint")
        assert out["disease_type"] == "烂鳃"
        assert out["raw_class"] == "Bacterial gill disease"
        assert out["model_version"].endswith("+folder_hint")
        assert "不是图像推理结果" in out["note"]

    def test_folder_hint_maps_each_family(self, tmp_path: Path):
        """7 分类目录名都要能正确落到对外标签上（多对一映射不能接错）。"""
        from fisheries.tools.fish_disease import RAW_CLASS_TO_LABEL

        for raw, expected in RAW_CLASS_TO_LABEL.items():
            folder = tmp_path / raw
            folder.mkdir()
            img = write_min_png(folder / "x.png")
            out = fish_disease_detect(str(img), force_mock=True, mock_source="folder_hint")
            assert out["disease_type"] == expected, f"{raw} 应映射到 {expected}"
            assert out["raw_class"] == raw

    def test_folder_hint_falls_back_when_path_has_no_label(self, tmp_path: Path):
        """路径里没有类别信息时，退回哈希模式，不能崩。"""
        img = write_min_png(tmp_path / "random_name.png")
        out = fish_disease_detect(str(img), force_mock=True, mock_source="folder_hint")
        assert out["status"] == "ok"
        assert out["raw_class"] is None


# --------------------------------------------------------------------------- #
# 外推可信度：这三个测试对应演示中真实暴露过的缺陷，防止回归
# --------------------------------------------------------------------------- #

def crashing_series(n: int = 24) -> list[dict]:
    """构造一段快速恶化的序列：溶氧从 6.2 掉到 2.6，氨氮爬升到 0.62。"""
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append({
            "timestamp": (T0 + timedelta(hours=i)).isoformat(),
            "water_temp": 28.0 + 1.5 * t,
            "ph": 8.05 + 0.20 * t,
            "dissolved_oxygen": 6.2 - 3.6 * t,
            "ammonia_n": 0.15 + 0.47 * t,
            "nitrite_n": 0.05 + 0.10 * t,
        })
    return out


class TestExtrapolationSanity:
    """长程线性外推必须被截断，否则会算出"72 小时后溶氧 0.00"这种自毁可信度的结论。"""

    def test_horizon_is_capped(self):
        out = water_risk_predict(crashing_series(), horizon_hours=72)
        assert out["horizon_hours"] == 72
        assert out["extrapolation_horizon_hours"] == 12.0
        assert out["projections"]["dissolved_oxygen"]["horizon_used_hours"] == 12.0

    def test_no_absurd_projection(self):
        """外推值必须落在物理量程内，且不能因为截断而仍然是 0。"""
        out = water_risk_predict(crashing_series(), horizon_hours=72)
        do = out["projections"]["dissolved_oxygen"]
        assert do["projected_extreme"] > 0.0
        assert do["projected_extreme"] < 6.2

    def test_short_horizon_not_padded(self):
        """horizon_hours 小于上限时，应按用户要求的窗口外推，不能被强行拉长到 12h。"""
        out = water_risk_predict(crashing_series(), horizon_hours=6)
        assert out["extrapolation_horizon_hours"] == 6.0
        assert out["projections"]["dissolved_oxygen"]["horizon_used_hours"] == 6.0

    def test_do_alert_countdown_reported(self):
        """溶氧跌破下一条管控线的倒计时必须给出，这是最可执行的一个数。"""
        out = water_risk_predict(crashing_series(), horizon_hours=72)
        assert out["projected_do_alert_label"] == "泛塘临界值"
        assert out["projected_do_alert_threshold"] == 1.5
        eta = out["projected_do_alert_eta_hours"]
        # 溶氧当前 2.60、斜率约 -0.16/h → (2.60-1.50)/0.16 ≈ 6.9h
        assert 5.0 < eta < 9.0

    def test_alert_targets_upper_line_when_safe(self):
        """溶氧尚在 3.0 以上时，倒计时应对准 3.0 危险线（该开增氧机），而不是 1.5 泛塘线。"""
        series = [
            {"timestamp": (T0 + timedelta(hours=i)).isoformat(),
             "water_temp": 26.0, "ph": 7.4,
             "dissolved_oxygen": 6.4 - 0.05 * i,
             "ammonia_n": 0.10, "nitrite_n": 0.05}
            for i in range(24)
        ]
        out = water_risk_predict(series, horizon_hours=72)
        assert out["projected_do_alert_threshold"] == 3.0
        assert out["projected_do_alert_label"] == "危险线"

    def test_summary_states_extrapolation_caveat(self):
        out = water_risk_predict(crashing_series())
        assert "仅前 12 小时" in out["degradation_summary"]


class TestSummaryConsistency:
    """告警文案不能与数据自相矛盾——这是给养殖户看的东西。"""

    def test_no_falling_wording_when_flat(self):
        """趋势平稳时不得出现"溶氧下降"的措辞。"""
        out = water_risk_predict(make_series(24, dissolved_oxygen=4.1, ammonia_n=0.35))
        assert out["trends"]["dissolved_oxygen"]["direction"] == "平稳"
        assert "溶氧持续下降" not in out["degradation_summary"]

    def test_missing_indicator_not_called_exceeded(self):
        """探头缺失的指标不得在文案里被说成"超标"（场景三真实踩过的坑）。"""
        readings = [
            {"timestamp": (T0 + timedelta(hours=i)).isoformat(),
             "water_temp": 27.0, "ph": 7.8,
             "dissolved_oxygen": 4.1, "ammonia_n": 0.35}
            for i in range(12)
        ]
        out = water_risk_predict(readings)
        assert "亚硝酸盐" in out["data_quality"]["missing_indicators"]
        assert "亚硝酸盐超标" not in out["degradation_summary"]
        assert "氨氮超标" in out["degradation_summary"]

    def test_summary_numbers_match_structured_fields(self):
        """自然语言里的数字必须与结构化字段一致，不能各说各话。"""
        out = water_risk_predict(crashing_series())
        eta = out["projected_do_alert_eta_hours"]
        assert f"{eta:.1f} 小时后溶氧将跌破" in out["degradation_summary"]
        assert f"{out['risk_score']:.1f}" in out["degradation_summary"]


# --------------------------------------------------------------------------- #
# 训练集类别 ↔ 推理侧类别映射 的一致性
# --------------------------------------------------------------------------- #


def _load_build_yolo_dataset():
    """按路径加载 ``scripts/build_yolo_dataset.py``。

    它不是安装包的一部分（``scripts/`` 不在 ``src/`` 里），只能这样加载。
    这个守卫之所以值得存在，是因为两处不一致的后果**是静默的**：
    模型会输出一个 ``RAW_CLASS_TO_LABEL`` 里查不到的类别名，
    真实推理走到 ``fish_disease.py`` 那句 ValueError 才炸，
    而且要等到跑真实图片才暴露——训练完不验收根本发现不了。
    """
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[2] / "scripts" / "build_yolo_dataset.py"
    spec = importlib.util.spec_from_file_location("_build_yolo_dataset", path)
    module = importlib.util.module_from_spec(spec)
    # 必须先登记到 sys.modules 再 exec：脚本里用了 @dataclass，
    # 而 dataclasses 解析注解时要靠 cls.__module__ 反查模块字典，
    # 未登记时 __module__ 查到 None，会以 "NoneType has no attribute __dict__" 报错。
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


class TestTrainingInferenceClassContract:
    """训练脚本产出的类别名必须能被推理侧翻译成对外标签。"""

    def test_every_trained_class_is_registered(self):
        trained = {c.key for c in _load_build_yolo_dataset().CLASSES}
        missing = trained - set(RAW_CLASS_TO_LABEL)
        assert not missing, f"训练类别未登记到 RAW_CLASS_TO_LABEL，真实推理会直接报错：{missing}"

    def test_trained_classes_cover_the_four_target_labels(self):
        """四分类训练集必须恰好覆盖 健康/烂鳃/出血/水霉 四个对外标签。"""
        labels = {RAW_CLASS_TO_LABEL[c.key] for c in _load_build_yolo_dataset().CLASSES}
        assert labels == {"健康", "烂鳃", "出血", "水霉"}

    def test_every_trained_label_has_severity_meta(self):
        """融合层要靠 LABEL_META 取 severity_weight，缺一个就是 KeyError。"""
        for cls in _load_build_yolo_dataset().CLASSES:
            label = RAW_CLASS_TO_LABEL[cls.key]
            assert label in LABEL_META, f"{cls.key} → {label} 在 LABEL_META 里没有条目"
            assert "severity_weight" in LABEL_META[label]
