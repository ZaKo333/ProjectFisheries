"""病灶检测器（两类实例分割）与「是不是鱼」门控的单元测试。

背景（真实缺陷）：四类分类器 ``models/yolov8-fish.pt`` 的 softmax 必须归一，
因此**任何一张图都会得到一个答案**。实测把一张设计海报喂进去，它返回
「健康，置信度 0.9997」—— 界面上就是「已识别：健康（健康）置信度 100%」。

之后补进了 201 张真实鱼的多边形标注，训练出一个两类分割模型
（``models/yolov8-fish-lesion-seg.pt``，val 38 张：fish AP50 0.990 /
lesion AP50 0.948）。它带来两件事，本文件把两者的**不同可信度**都钉住：

1. **病灶定位可信**：鱼身上有几处病灶、占鱼体多大面积，落在返回体里；
2. **「有没有鱼」不可信**：训练集全是鱼、没有任何负样本，那张海报同样得到
   ``conf=0.830`` 的鱼框（真实鱼 val 集最低 0.815，分不开）。所以门控
   ``enable_gate`` **默认关闭**，``is_fish`` 默认是 None；只有显式打开时才
   会出现 True/False，并且一定会附带免责声明。

这些测试全部用假的检测结果，不依赖本机是否训练过权重。

运行：
    cd <工程>
    .venv\\Scripts\\python.exe -m pytest tests/unit/test_fish_lesion.py -v
"""

from __future__ import annotations

import struct
import sys
import types
import zlib
from pathlib import Path

import pytest

from fisheries.agent.fusion import _vision_score
from fisheries.tools import fish_disease
from fisheries.tools.fish_disease import (
    CLOSED_SET_CAVEAT,
    NOT_A_FISH_NOTE,
    detect_fish_lesions,
    fish_disease_detect,
)

# --------------------------------------------------------------------------- #
# 测试夹具
# --------------------------------------------------------------------------- #

def write_min_png(path: Path, width: int = 8, height: int = 8) -> Path:
    """生成一张合法的最小 PNG（与 test_tools.py 同款，避免共用私有函数）。"""
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
    return write_min_png(tmp_path / "frame.png")


def fake_detection(fish: list[dict] | None = None, lesion: list[dict] | None = None) -> dict:
    return {"fish": fish or [], "lesion": lesion or [], "image_size": [200, 100]}


def box(conf: float, area_ratio: float) -> dict:
    side = 10.0
    return {
        "confidence": conf,
        "box": [0.0, 0.0, side, side],
        "area_ratio": area_ratio,
    }


@pytest.fixture
def real_backend(monkeypatch: pytest.MonkeyPatch):
    """把「选了真实后端」这件事固定下来，用于测门本身的行为。"""
    monkeypatch.setattr(
        fish_disease, "_resolve_backend", lambda _p: ("yolo", "yolov8-fish.pt")
    )
    monkeypatch.setattr(fish_disease, "_resolve_lesion_weights", lambda _p=None: "seg.pt")


def forbid_cls(monkeypatch: pytest.MonkeyPatch) -> None:
    """断言「被门挡住了，所以分类器根本没跑」。"""
    def boom(*_a, **_kw):
        raise AssertionError("不应该调用四类分类器：门已经判定图里没有鱼")

    monkeypatch.setattr(fish_disease, "_yolo_predict", boom)


# --------------------------------------------------------------------------- #
# 一、没有病灶权重：行为与旧版一致
# --------------------------------------------------------------------------- #

class TestNoDetector:
    def test_without_lesion_weights_no_verdict_about_fish(self, png_file: Path) -> None:
        """本机没训练过分割模型时，不能假装自己判断过「是不是鱼」。"""
        out = fish_disease_detect(str(png_file), force_mock=True)

        assert out["status"] == "ok"
        assert out["is_fish"] is None
        assert out["detector_version"] is None
        assert out["lesion_count"] == 0
        assert out["lesion_area_ratio"] is None
        assert out["lesions"] == []
        assert not set(out) - {
            "tool", "status", "image_path", "disease_type", "disease_group", "confidence",
            "is_healthy", "severity_weight", "is_low_confidence", "all_scores", "raw_class",
            "model_version", "degraded_mode", "image_meta", "note", "is_fish",
            "lesion_count", "lesion_area_ratio", "lesions", "detector_version",
            "error_code", "error_message",
        }

    def test_missing_weights_path_falls_through(self, png_file: Path) -> None:
        """显式给一个不存在的权重路径，等于关掉检测器，而不是报错。"""
        out = fish_disease_detect(
            str(png_file), weights_path="./nope.pt", lesion_weights_path="./nope-seg.pt"
        )
        assert out["status"] == "ok"
        assert out["is_fish"] is None

    def test_gate_disabled_restores_old_behaviour(
        self, png_file: Path, real_backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """enable_gate=False 必须回到旧行为：分类器直接给答案。"""
        monkeypatch.setattr(
            fish_disease,
            "_yolo_predict",
            lambda *_a, **_kw: ("健康", {"健康": 1.0, "水霉": 0.0, "出血": 0.0, "烂鳃": 0.0}, "healthy"),
        )
        monkeypatch.setattr(
            fish_disease, "detect_fish_lesions", lambda *_a, **_kw: fake_detection()
        )

        out = fish_disease_detect(str(png_file), enable_gate=False)

        assert out["disease_type"] == "健康"
        assert out["is_fish"] is None
        assert CLOSED_SET_CAVEAT in out["note"]


# --------------------------------------------------------------------------- #
# 二、门判定「不是鱼」
# --------------------------------------------------------------------------- #

class TestNotAFish:
    """门控开启（``enable_gate=True``）时的拒绝路径。

    注意这是**显式选择**的行为：默认关闭，因为当前训练集判不准非鱼类图片。
    """

    def test_zero_fish_boxes_yields_no_disease_conclusion(
        self, png_file: Path, real_backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """海报那个 bug 在「有负样本之后」应有的表现。"""
        monkeypatch.setattr(
            fish_disease, "detect_fish_lesions", lambda *_a, **_kw: fake_detection()
        )
        forbid_cls(monkeypatch)

        out = fish_disease_detect(str(png_file), enable_gate=True)

        assert out["status"] == "ok"
        assert out["is_fish"] is False
        # 四个「结论字段」必须全部为空，不能留一个 0.9997 在那里
        assert out["disease_type"] is None
        assert out["confidence"] is None
        assert out["is_healthy"] is None
        assert out["severity_weight"] is None
        assert out["all_scores"] == {}
        assert out["degraded_mode"] is False
        assert out["note"] == NOT_A_FISH_NOTE
        assert out["detector_version"] == "seg.pt"
        assert "健康" not in out["note"]

    def test_lesion_area_ratio_is_null_without_fish_area(
        self, png_file: Path, real_backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """没有鱼框时算不出「病灶占鱼体比例」，只能是 None 而不是 0。"""
        monkeypatch.setattr(
            fish_disease,
            "detect_fish_lesions",
            lambda *_a, **_kw: fake_detection(lesion=[box(0.9, 0.4)]),
        )
        forbid_cls(monkeypatch)

        out = fish_disease_detect(str(png_file), enable_gate=True)
        assert out["lesion_area_ratio"] is None
        assert out["lesion_count"] == 1

    def test_detector_crash_falls_back_to_old_path(
        self, png_file: Path, real_backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """检测器自己挂了不该让整条链路断掉：退回旧行为并保留免责声明。"""
        def boom(*_a, **_kw):
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(fish_disease, "detect_fish_lesions", boom)
        monkeypatch.setattr(
            fish_disease,
            "_yolo_predict",
            lambda *_a, **_kw: ("出血", {"出血": 0.9, "健康": 0.1, "水霉": 0.0, "烂鳃": 0.0}, "bleeding"),
        )

        out = fish_disease_detect(str(png_file), enable_gate=True)
        assert out["disease_type"] == "出血"
        assert out["is_fish"] is None
        assert CLOSED_SET_CAVEAT in out["note"]


# --------------------------------------------------------------------------- #
# 三、默认行为：只补病灶定位，不下「是不是鱼」的结论
# --------------------------------------------------------------------------- #

class TestDefaultIsLesionsOnly:
    def test_default_runs_detector_but_keeps_is_fish_null(
        self, png_file: Path, real_backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """默认值下检测器照样跑（病灶是真收益），但 ``is_fish`` 保持 None。

        这是刻意设计：拿一个分不开海报与鱼的门去做拒绝，比不做还危险。
        """
        monkeypatch.setattr(
            fish_disease,
            "detect_fish_lesions",
            # 连一个鱼框都没有，默认也不应该拒绝
            lambda *_a, **_kw: fake_detection(),
        )
        monkeypatch.setattr(
            fish_disease,
            "_yolo_predict",
            lambda *_a, **_kw: ("健康", {"健康": 1.0, "水霉": 0.0, "出血": 0.0, "烂鳃": 0.0}, "healthy"),
        )

        out = fish_disease_detect(str(png_file))

        assert out["is_fish"] is None
        assert out["disease_type"] == "健康"
        assert out["detector_version"] == "seg.pt"

    def test_lesion_stats_attached_even_without_gate(
        self, png_file: Path, real_backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """门关着也要有病灶定位——这是这次重训唯一的真收益。"""
        monkeypatch.setattr(
            fish_disease,
            "detect_fish_lesions",
            lambda *_a, **_kw: fake_detection(fish=[box(0.9, 0.5)], lesion=[box(0.8, 0.05)]),
        )
        monkeypatch.setattr(
            fish_disease,
            "_yolo_predict",
            lambda *_a, **_kw: ("水霉", {"水霉": 0.9, "健康": 0.1, "出血": 0.0, "烂鳃": 0.0}, "fungus"),
        )

        out = fish_disease_detect(str(png_file))

        assert out["is_fish"] is None
        assert out["lesion_count"] == 1
        assert out["lesion_area_ratio"] == pytest.approx(0.10, abs=1e-4)
        assert "10.0%" in out["note"]
        assert CLOSED_SET_CAVEAT in out["note"]


# --------------------------------------------------------------------------- #
# 四、门通过：鱼 + 病灶
# --------------------------------------------------------------------------- #

class TestFishWithLesion:
    def test_lesion_stats_land_in_output(
        self, png_file: Path, real_backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            fish_disease,
            "detect_fish_lesions",
            lambda *_a, **_kw: fake_detection(
                fish=[box(0.95, 0.50)], lesion=[box(0.8, 0.10), box(0.6, 0.05)]
            ),
        )
        monkeypatch.setattr(
            fish_disease,
            "_yolo_predict",
            lambda *_a, **_kw: ("烂鳃", {"烂鳃": 0.9, "健康": 0.1, "水霉": 0.0, "出血": 0.0}, "gill"),
        )

        out = fish_disease_detect(str(png_file), enable_gate=True)

        assert out["is_fish"] is True
        assert out["disease_type"] == "烂鳃"
        assert out["lesion_count"] == 2
        # 0.10 + 0.05 占鱼体 0.50 = 0.30
        assert out["lesion_area_ratio"] == pytest.approx(0.30, abs=1e-4)
        assert out["detector_version"] == "seg.pt"
        assert out["model_version"] == "yolov8-fish.pt+seg.pt"
        assert "病灶" in out["note"]
        assert "30.0%" in out["note"]
        # 即使门放行了，也不能声称「确认是鱼」—— 检测器现在分不开海报和鱼，
        # 这句免责必须一直在，否则就是把一个未经验证的判断当成事实。
        assert CLOSED_SET_CAVEAT in out["note"]

    def test_health_verdict_with_lesion_reported_as_conflict(
        self, png_file: Path, real_backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            fish_disease,
            "detect_fish_lesions",
            lambda *_a, **_kw: fake_detection(fish=[box(0.9, 0.6)], lesion=[box(0.7, 0.12)]),
        )
        monkeypatch.setattr(
            fish_disease,
            "_yolo_predict",
            lambda *_a, **_kw: ("健康", {"健康": 0.99, "水霉": 0.01, "出血": 0.0, "烂鳃": 0.0}, "healthy"),
        )

        out = fish_disease_detect(str(png_file))

        assert out["disease_type"] == "健康"
        assert out["lesion_count"] == 1
        assert "矛盾" in out["note"]

    def test_disease_without_located_lesion_asks_for_review(
        self, png_file: Path, real_backend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """分类器说有病、检测器却找不到病灶 —— 这种不一致要说出来。"""
        monkeypatch.setattr(
            fish_disease,
            "detect_fish_lesions",
            lambda *_a, **_kw: fake_detection(fish=[box(0.9, 0.6)]),
        )
        monkeypatch.setattr(
            fish_disease,
            "_yolo_predict",
            lambda *_a, **_kw: ("水霉", {"水霉": 0.8, "健康": 0.2, "出血": 0.0, "烂鳃": 0.0}, "fungus"),
        )

        out = fish_disease_detect(str(png_file))
        assert out["lesion_count"] == 0
        assert "未定位到明确病灶区域" in out["note"]


# --------------------------------------------------------------------------- #
# 四、检测器本身的解析逻辑（用假的 ultralytics）
# --------------------------------------------------------------------------- #

class _Tensor:
    def __init__(self, values: list) -> None:
        self._values = values

    def tolist(self) -> list:
        return self._values


class _Boxes:
    def __init__(self, xyxy: list, conf: list, cls: list) -> None:
        self.xyxy = _Tensor(xyxy)
        self.conf = _Tensor(conf)
        self.cls = _Tensor(cls)

    def __len__(self) -> int:
        return len(self.xyxy.tolist())


class _Result:
    def __init__(self, xyxy: list, conf: list, cls: list) -> None:
        self.names = {0: "fish", 1: "lesion"}
        self.orig_shape = (100, 200)  # h, w
        self.boxes = _Boxes(xyxy, conf, cls)


@pytest.fixture
def stub_ultralytics(monkeypatch: pytest.MonkeyPatch):
    """装一个假的 ultralytics，用来测我们自己的解析代码。"""
    captured: dict = {}

    class _Model:
        def __init__(self, weights: str) -> None:
            captured["weights"] = weights

        def predict(self, path: str, conf=None, verbose=False):  # noqa: ARG002
            captured["path"] = path
            captured["conf"] = conf
            return [captured["result"]]

    module = types.ModuleType("ultralytics")
    module.YOLO = _Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return captured


class TestDetectorParsing:
    def test_boxes_are_split_and_sorted_by_area(self, stub_ultralytics) -> None:
        stub_ultralytics["result"] = _Result(
            xyxy=[(0, 0, 10, 10), (0, 0, 100, 50), (5, 5, 15, 25)],
            conf=[0.40, 0.92, 0.70],
            cls=[1, 0, 1],
        )

        out = detect_fish_lesions("frame.png", "seg.pt", conf=0.35)

        assert out["image_size"] == [200, 100]
        assert len(out["fish"]) == 1 and len(out["lesion"]) == 2
        # 面积降序：病灶 (10x20) 大于 (10x10)
        assert out["lesion"][0]["box"] == [5.0, 5.0, 15.0, 25.0]
        # 100x50 / (200x100) = 0.25
        assert out["fish"][0]["area_ratio"] == pytest.approx(0.25, abs=1e-4)
        assert stub_ultralytics["conf"] == 0.35

    def test_no_boxes_gives_empty_both(self, stub_ultralytics) -> None:
        stub_ultralytics["result"] = _Result(xyxy=[], conf=[], cls=[])
        out = detect_fish_lesions("frame.png", "seg.pt")
        assert out["fish"] == [] and out["lesion"] == []


# --------------------------------------------------------------------------- #
# 五、下游融合：「没检测到鱼」不能当成「有病害」
# --------------------------------------------------------------------------- #

class TestFusionTreatsItAsNoEvidence:
    def test_vision_score_is_zero_with_explicit_evidence(self) -> None:
        score, ev = _vision_score({
            "status": "ok",
            "is_fish": False,
            "disease_type": None,
            "is_healthy": None,
            "severity_weight": None,
            "confidence": None,
        })

        assert score == 0.0
        assert [e.code for e in ev] == ["VISION_NOT_A_FISH"]
        # 关键：不能出现「检出None」这种字符串
        assert all("None" not in e.label for e in ev)

    def test_real_disease_still_scores(self) -> None:
        """门引入的 early-return 不能把正常的病害评分也一并跳过。"""
        score, ev = _vision_score({
            "status": "ok",
            "is_fish": True,
            "disease_type": "烂鳃",
            "disease_group": "细菌性",
            "severity_weight": 0.8,
            "confidence": 0.9,
            "is_healthy": False,
        })

        assert score == pytest.approx(0.72, abs=1e-6)
        assert [e.code for e in ev] == ["VISION_DISEASE"]
