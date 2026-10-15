"""从 examples/pond_demo_water.csv 生成「可直接粘贴到 /docs 里」的请求体，并实测其真实结果。

用途：让 docs/前端对接说明.md 里的示例请求与「预期输出」都是实测过的，不是手写的。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries.webui import WebRequest, WebSession  # noqa: E402

CSV_PATH = ROOT / "examples" / "pond_demo_water.csv"

FIELD_MAP = {
    "water_temp_c": "water_temp",
    "ph": "ph",
    "dissolved_oxygen_mg_l": "dissolved_oxygen",
    "ammonia_n_mg_l": "ammonia_n",
    "nitrite_n_mg_l": "nitrite_n",
}

def build_readings() -> list[dict]:
    rows: list[dict] = []
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            item = {"timestamp": row["timestamp"].replace(" ", "T")}
            for src, dst in FIELD_MAP.items():
                item[dst] = float(row[src])
            rows.append(item)
    return rows

def main() -> int:
    readings = build_readings()
    body = {
        "pond_id": "demo-pond-1",
        "species": "鲤科",
        "readings": readings,
        "image_path": None,
        "upload_id": None,
        "mock_source": "hash",
        "use_knowledge": None,
        "question": "",
        "include_trace": False,
        "include_knowledge_block": False,
    }

    print("=" * 78)
    print("请求体（压缩成一行，便于粘贴）")
    print("=" * 78)
    print(json.dumps(body, ensure_ascii=False, separators=(",", ":")))
    print()

    session = WebSession(use_knowledge=False)
    result = session.run(WebRequest(readings=readings, pond_id="demo-pond-1", species="鲤科", use_knowledge=False))

    print("=" * 78)
    print("实测结果")
    print("=" * 78)
    print(f"ok           = {result.ok}")
    print(f"risk_level   = {result.risk_level}")
    print(f"fused_score  = {result.fused_score}")
    print(f"headline     = {result.headline}")
    print(f"reason       = {result.reason}")
    print(f"n_alerts     = {len(result.alerts)}")
    for a in result.alerts:
        print(f"  - [{a.get('level')}] {a.get('code')} :: {a.get('title')}")
    print(f"n_water_actions   = {len(result.water_actions)}")
    for a in result.water_actions:
        print(f"  - P{a.get('priority')} [{a.get('target')}] {a.get('action')} @{a.get('timing')}")
    print(f"observation_window = {result.observation_window}")
    print(f"confidence_note    = {result.confidence_note}")
    fusion = result.fusion or {}
    print(
        "fusion: "
        f"weighted={fusion.get('weighted_score')} fused={fusion.get('fused_score')} "
        f"floor={fusion.get('score_floor')} floors={fusion.get('level_floors')} "
        f"weights={fusion.get('effective_weights')}"
    )
    water = result.water or {}
    print(f"water.limiting_factor = {water.get('limiting_factor')} / {water.get('limiting_factor_label')}")
    print(f"water.projected_do_alert_eta_hours = {water.get('projected_do_alert_eta_hours')}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
