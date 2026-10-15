"""从 examples/pond_demo_water.csv 生成「可直接粘贴到 /docs 里」的请求体，并实测其真实结果。

用途：让 docs/ 里的示例请求与「预期输出」都是实测过的，不是手写的。

走的是**真实 HTTP 链路**（``create_app()`` + ``TestClient``），而不是直接调内部对象 ——
示例要能贴进 /docs，就必须先证明它经得起一次真实的请求校验。
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

from fastapi.testclient import TestClient  # noqa: E402

from fisheries.api.app import create_app  # noqa: E402

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
        "mock_source": "hash",
        "include_trace": False,
        "include_knowledge_block": False,
    }

    print("=" * 78)
    print("请求体（压缩成一行，便于粘贴到 /docs 的 POST /api/v1/assess 里）")
    print("=" * 78)
    print(json.dumps(body, ensure_ascii=False, separators=(",", ":")))
    print()

    with TestClient(create_app()) as client:
        resp = client.post("/api/v1/assess", json=body)

    print("=" * 78)
    print(f"实测结果（HTTP {resp.status_code}）")
    print("=" * 78)
    if resp.status_code != 200:
        print(resp.text[:2000])
        return 1

    data = resp.json()
    print(f"risk_level   = {data.get('risk_level')}")
    print(f"fused_score  = {data.get('fused_score')}")
    print(f"headline     = {data.get('headline')}")
    print(f"reason       = {data.get('reason')}")

    alerts = data.get("alerts") or []
    print(f"n_alerts     = {len(alerts)}")
    for a in alerts:
        print(f"  - [{a.get('level')}] {a.get('code')} :: {a.get('title')}")

    report = data.get("report") or {}
    actions = report.get("water_actions") or []
    print(f"n_water_actions = {len(actions)}")
    for a in actions:
        print(f"  - P{a.get('priority')} [{a.get('target')}] {a.get('action')} @{a.get('timing')}")

    print(f"observation_window = {data.get('observation_window')}")
    print(f"confidence_note    = {data.get('confidence_note')}")

    fusion = data.get("fusion") or {}
    print(
        "fusion: "
        f"weighted={fusion.get('weighted_score')} fused={fusion.get('fused_score')} "
        f"floor={fusion.get('score_floor')} floors={fusion.get('level_floors')} "
        f"weights={fusion.get('effective_weights')}"
    )

    water = data.get("water") or {}
    print(f"water.limiting_factor = {water.get('limiting_factor')} / {water.get('limiting_factor_label')}")
    print(f"water.projected_do_alert_eta_hours = {water.get('projected_do_alert_eta_hours')}")

    refs = data.get("references") or []
    print(f"n_references = {len(refs)}")
    for r in refs[:3]:
        print(f"  - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
