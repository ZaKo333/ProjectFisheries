"""接口层冒烟探针：真实地把 /assess、/ingest/*、/tools/call、/chat 打一遍。

这不是单测（正式断言在 tests/unit/test_api.py），目的是在写测试前先看一遍
真实响应，避免照着「以为的响应」去写断言。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from starlette.testclient import TestClient  # noqa: E402

from fisheries import datasets as _datasets  # noqa: E402
from fisheries.api.app import create_app  # noqa: E402

IMAGE = str(_datasets.resolve(
    "fish_disease", "Test", "Bacterial gill disease", "Bacterial gill disease (100).jpg"
))

def crashing_water(n: int = 24) -> list[dict]:
    out = []
    for i in range(n):
        t = i / (n - 1)
        out.append(
            {
                "water_temp": round(28.0 + 1.5 * t, 2),
                "ph": round(8.05 + 0.20 * t, 3),
                "dissolved_oxygen": round(6.2 - 3.6 * t, 3),
                "ammonia_n": round(0.15 + 0.47 * t, 4),
                "nitrite_n": round(0.05 + 0.10 * t, 4),
            }
        )
    return out

def good_water(n: int = 24) -> list[dict]:
    out = []
    for i in range(n):
        w = 0.15 * (1 if i % 2 else -1)
        out.append(
            {
                "water_temp": round(26.0 + w, 2),
                "ph": round(7.6 + w * 0.5, 3),
                "dissolved_oxygen": round(6.4 + w, 3),
                "ammonia_n": round(0.12 + w * 0.05, 4),
                "nitrite_n": round(0.05 + w * 0.02, 4),
            }
        )
    return out

def show(title: str, resp) -> None:
    print(f"\n{'=' * 78}\n{title}  ->  HTTP {resp.status_code}\n{'=' * 78}")
    body = resp.json()
    print(json.dumps(body, ensure_ascii=False, indent=2)[:2600])

def main() -> int:
    app = create_app()
    with TestClient(app) as c:
        show("1) 高危 + 真实烂鳃图片", c.post("/api/v1/assess", json={
            "pond_id": "P-01",
            "readings": crashing_water(),
            "image_path": IMAGE,
            "mock_source": "folder_hint",
            "include_knowledge_block": False,
        }))

        show("2) 仅正常水质", c.post("/api/v1/assess", json={
            "pond_id": "P-02", "readings": good_water(),
        }))

        show("3) 仅图片（单模态权重归一化）", c.post("/api/v1/assess", json={
            "pond_id": "P-03", "image_path": IMAGE, "mock_source": "folder_hint",
        }))

        show("4) 两模态都不给 -> 应当 422", c.post("/api/v1/assess", json={"pond_id": "P-04"}))

        show("5) readings 传空列表 -> 应当 422", c.post("/api/v1/assess", json={
            "pond_id": "P-05", "readings": [],
        }))

        show("6) 未知字段 dissolved_oxyen -> 应当 422", c.post("/api/v1/assess", json={
            "readings": [{"water_temp": 26, "ph": 7.5, "dissolved_oxyen": 6.0}] * 3,
        }))

        show("7) 温度传 NaN 字符串 -> 应当 422", c.post("/api/v1/assess", json={
            "readings": [{"water_temp": "abc", "ph": 7.5, "dissolved_oxygen": 6.0}] * 3,
        }))

        show("8) 直调工具 water_risk_predict（只 3 条，应提示样本不足）",
             c.post("/api/v1/tools/call", json={
                 "tool": "water_risk_predict",
                 "arguments": {"pond_id": "P-08", "readings": crashing_water(3)},
             }))

        show("9) 直调工具 fish_disease_detect（不存在的路径）",
             c.post("/api/v1/tools/call", json={
                 "tool": "fish_disease_detect",
                 "arguments": {"image_path": r"D:\nope\none.jpg"},
             }))

        show("10) 传感器写入缓冲", c.post("/api/v1/ingest/sensor", json={
            "pond_id": "P-10", "readings": crashing_water(),
        }))

        show("11) 用缓冲评估（不带 readings）", c.post("/api/v1/assess", json={
            "pond_id": "P-10", "use_buffer": True,
        }))

        show("12) 空缓冲评估 -> 应当 409", c.post("/api/v1/assess", json={
            "pond_id": "P-EMPTY", "use_buffer": True,
        }))

        show("13) 读回缓冲", c.get("/api/v1/ponds/P-10/readings?limit=2"))

        show("14) 历史", c.get("/api/v1/ponds/P-01/history?limit=1"))

        show("15) 阈值配置", c.get("/api/v1/config/thresholds"))

        show("16) 工具契约", c.get("/api/v1/config/tools"))

        with open(IMAGE, "rb") as fh:
            show("17) 上传图片", c.post(
                "/api/v1/ingest/image",
                files={"file": ("fish.jpg", fh, "image/jpeg")},
            ))

        show("18) 上传非法扩展名 -> 应当 415", c.post(
            "/api/v1/ingest/image",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        ))

        show("19) 上传伪装成 jpg 的文本 -> 应当 422（魔数校验）", c.post(
            "/api/v1/ingest/image",
            files={"file": ("fake.jpg", b"not an image at all", "image/jpeg")},
        ))

        show("20) 知识库问答（本进程已禁用知识库 -> 503）",
             c.post("/api/v1/chat", json={"query": "溶氧低怎么办"}))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
