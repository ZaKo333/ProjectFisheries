r"""探针：可信度说明的「规则算事实 + 模板措辞」在几种输入下的实际输出。

不是交付物，只为肉眼核对措辞。运行：
    python scripts\probes\_probe_confidence.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console

_console.install()

from fisheries.agent.prompts import (  # noqa: E402
    build_confidence_facts,
    format_confidence_block,
    render_confidence_note,
)


class _Chunk:
    def __init__(self, verified: bool) -> None:
        self.verified = verified

class _Ctx:
    chunks: ClassVar[list[_Chunk]] = [_Chunk(True), _Chunk(True), _Chunk(False)]

def show(title: str, facts: dict) -> None:
    print(f"\n===== {title} =====")
    print("[facts] " + json.dumps(facts, ensure_ascii=False))
    print("[zh] " + render_confidence_note(facts))
    print("[en] " + render_confidence_note(facts, "en"))

def main() -> int:
    water = {
        "status": "ok", "risk_level": "预警", "risk_score": 57.07,
        "data_quality": {"n_points": 24, "interval_minutes": 60, "span_hours": 23.0,
                         "missing_indicators": [], "warnings": []},
        "extrapolation_horizon_hours": 12.0, "degraded_mode": False,
    }
    vision = {"status": "ok", "is_healthy": False, "disease_type": "烂鳃",
              "confidence": 0.86, "severity_weight": 0.85, "is_low_confidence": False}
    fusion = {"risk_level": "预警", "modalities": ["water", "vision"],
              "failed_modalities": {}}

    facts = build_confidence_facts(water, vision, fusion, _Ctx())
    show("水质预警 + 烂鳃 0.86 + 一条未核对资料", facts)
    print("\n[zh block]\n" + format_confidence_block(facts))
    print("\n[en block]\n" + format_confidence_block(facts, "en"))

    show(
        "水质正常、点数充足、无病害输入",
        build_confidence_facts(
            {"status": "ok", "risk_level": "正常", "risk_score": 0.0,
             "data_quality": {"n_points": 48, "interval_minutes": 60},
             "extrapolation_horizon_hours": 0.0, "degraded_mode": False},
            None,
            {"risk_level": "正常", "modalities": ["water"], "failed_modalities": {}},
        ),
    )

    show(
        "水质只有 12 点、工具降级、图像分析失败",
        build_confidence_facts(
            {"status": "ok", "risk_level": "预警", "risk_score": 43.0,
             "data_quality": {"n_points": 12, "interval_minutes": 30},
             "extrapolation_horizon_hours": 4.0, "degraded_mode": True},
            None,
            {"risk_level": "预警", "modalities": ["water"],
             "failed_modalities": {"vision": "IMAGE_NOT_FOUND"}},
        ),
    )

    show(
        "只有图片、病害置信度偏低",
        build_confidence_facts(
            None,
            {"status": "ok", "is_healthy": False, "disease_type": "水霉",
             "confidence": 0.45, "severity_weight": 0.70, "is_low_confidence": True},
            {"risk_level": "预警", "modalities": ["vision"], "failed_modalities": {}},
        ),
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
