r"""生成一份真实的 /assess 响应样例，供前端同学对着做界面。

不是交付物，只为产出一个可当 mock 用的 JSON。运行：
    python scripts\probes\_make_sample_response.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console

_console.install()

from starlette.testclient import TestClient  # noqa: E402

from fisheries.api.app import create_app  # noqa: E402

READINGS = [
    {
        "timestamp": f"2026-10-13T{h:02d}:00:00",
        "water_temp": 27.4 - h * 0.05,
        "ph": 7.85 + h * 0.004,
        "dissolved_oxygen": round(6.42 - h * 0.093, 3),
        "ammonia_n": round(0.100 + h * 0.0104, 4),
        "nitrite_n": round(0.050 + h * 0.004, 4),
    }
    for h in range(24)
]

#: 高危样例：溶氧末点 1.9（跌破泛塘线）、氨氮 0.72、亚硝酸盐 0.35 三项都 severity=1.0。
CRASH_READINGS = [
    {
        "timestamp": f"2026-10-13T{h:02d}:00:00",
        "water_temp": 29.6,
        "ph": 7.10,
        "dissolved_oxygen": round(6.80 - (6.80 - 1.90) * h / 23, 3),
        "ammonia_n": round(0.180 + (0.720 - 0.180) * h / 23, 4),
        "nitrite_n": round(0.080 + (0.350 - 0.080) * h / 23, 4),
    }
    for h in range(24)
]

def main() -> int:
    out: dict = {}
    with TestClient(create_app()) as client:
        assess = client.post(
            "/api/v1/assess",
            json={"pond_id": "demo-pond-1", "species": "鲤科", "readings": READINGS,
                  "question": "明天早上要不要提前开增氧机？"},
        )
        out["assess_status"] = assess.status_code
        out["assess"] = assess.json()

        crash = client.post(
            "/api/v1/assess",
            json={"pond_id": "demo-pond-1", "species": "鲤科", "readings": CRASH_READINGS,
                  "mock_source": "folder_hint"},
        )
        out["assess_critical_status"] = crash.status_code
        out["assess_critical"] = crash.json()

        health = client.get("/api/v1/health")
        out["health_status"] = health.status_code
        out["health"] = health.json()
        out["thresholds"] = client.get("/api/v1/config/thresholds").json()
        out["tools"] = client.get("/api/v1/config/tools").json()
        chat = client.post("/api/v1/chat", json={"query": "溶氧低于多少要开增氧机", "k": 3})
        out["chat_status"] = chat.status_code
        out["chat"] = chat.json()
        out["tool_call"] = client.post(
            "/api/v1/tools/call",
            json={"tool": "water_risk_predict",
                  "arguments": {"readings": CRASH_READINGS, "pond_id": "demo-pond-1"}},
        ).json()

    sys.stdout.write(json.dumps(out, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
