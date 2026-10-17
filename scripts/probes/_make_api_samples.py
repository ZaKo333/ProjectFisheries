"""导出前端对接契约：OpenAPI 规格 + 全部端点的真实响应样例。

输出两份文件，**它们是「前端拿到的接口长什么样」的唯一真相来源**：

* ``docs/openapi.json``        —— FastAPI 自动生成的机器可读规格（端点、模型、错误码）
* ``docs/_api_samples.json``   —— 20 个真实响应的样例（含 4xx 错误响应）

两个文件都做了**可重现化**：时间、耗时、随机 id、落盘路径这类每次都会变的字段
被换成固定占位符，所以重复运行产物逐字节一致，能进版本库、能 diff。业务字段
一律不碰——它们变了说明 API 真的变了，那正是想被看见的。

交付给前端时把这两份连同 ``mock_server.py`` 一起拷进交付包；
交付包的对应副本由 ``scripts/probes/_make_api_samples.py`` 之后手工同步，
或直接覆盖（见 ``docs/数据集说明.md`` 同一处的说明）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "_api_samples.json"
OPENAPI_OUT = ROOT / "docs" / "openapi.json"

# 允许直接运行——这些探针原来是靠调用方先设 PYTHONPATH 才能跑。
sys.path.insert(0, str(ROOT / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries import datasets as _datasets  # noqa: E402
from fisheries import paths as _paths  # noqa: E402

GILL_IMAGE = _datasets.resolve(
    "fish_disease", "Test", "Bacterial gill disease", "Bacterial gill disease (100).jpg"
)

def readings(n: int = 24) -> list[dict]:
    out = []
    for i in range(n):
        step = i / max(n - 1, 1)
        out.append(
            {
                "timestamp": f"2026-10-13T{i:02d}:00:00",
                "water_temp": round(27.5 - 0.4 * step, 2),
                "ph": round(7.9 - 0.25 * step, 2),
                "dissolved_oxygen": round(6.4 - 2.2 * step, 2),
                "ammonia_n": round(0.10 + 0.24 * step, 3),
                "nitrite_n": round(0.05 + 0.09 * step, 3),
            }
        )
    return out

#: 上传样例的 vision 段无法固化：落盘路径含随机 id，而 mock 分值由路径派生。
_UPLOAD_SAMPLE = "POST /api/v1/ingest/image"
_VISION_POINTER = (
    "<mock 识别结果。字段结构与 POST /api/v1/assess (图像单模态) 的 vision 字段完全一致，"
    "请看那份样例——那份走的是固定图片路径，数值是稳定的真值。"
    "这里不固化的原因：上传后的落盘路径含随机 id，而 mock 分值由路径派生，每次不同。>"
)


def _scrub(value: object) -> object:
    """把每次跑都会变的字段换成稳定占位符。

    交付给前端的样例文件如果每生成一次就变一次，就没法进版本库做 diff 了。
    这里只处理**明确与时间、耗时、随机 id 有关**的字段；业务字段一个都不动——
    它们的变化说明 API 真的变了，那正是我们想看见的。

    本机绝对路径也在此列，而且要一路扫到**字符串值内部**：错误响应的
    ``detail`` 是字符串化的 JSON，按 key 替换够不到它。
    """
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k == "server_time":
                out[k] = "<服务器当前时间，形如 2026-10-15T13:27:13+0800>"
            elif k == "uptime_seconds":
                out[k] = "0"
            elif k == "elapsed_seconds":
                out[k] = "<本次调用耗时，秒>"
            elif k == "upload_id":
                out[k] = "<每次上传生成的 id>"
            elif k in ("stored_path", "image_path", "resolved_path"):
                out[k] = "<服务端落盘路径，前缀是 data/uploads/>"
            else:
                out[k] = _scrub(v)
        return out
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    if isinstance(value, str):
        # 占位符定义在 fisheries.paths —— 和 evaluate_water_rules.py 共用一份，
        # 否则「哪些算本机路径」会分叉成两套说法。
        return _paths.stabilize(value)
    return value

def main() -> int:
    from fastapi.testclient import TestClient

    from fisheries.api.app import create_app

    samples: dict[str, object] = {}
    with TestClient(create_app(enable_cors=False)) as client:
        # OpenAPI 规格：前端的类型定义可以直接从它生成。
        openapi = client.get("/openapi.json").json()
        OPENAPI_OUT.write_text(
            json.dumps(openapi, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        samples["GET /api/v1/health"] = client.get("/api/v1/health").json()
        samples["GET /api/v1/config/thresholds"] = client.get("/api/v1/config/thresholds").json()
        samples["GET /api/v1/config/tools"] = client.get("/api/v1/config/tools").json()
        samples["GET /api/v1/config/templates"] = client.get("/api/v1/config/templates").json()

        # 模板 CSV 不是 JSON，但它就是前端「下载后直接塞进上传框」的那个字节流，
        # 给原样的文本比给一段描述有用。
        _template_csv = client.get("/api/v1/config/templates/dawn_hypoxia/csv")
        samples["GET /api/v1/config/templates/{key}/csv"] = {
            "content_type": _template_csv.headers["content-type"],
            "content_disposition": _template_csv.headers["content-disposition"],
            "body": _template_csv.text,
        }

        body = {
            "pond_id": "POND-03",
            "species": "鲤科",
            "readings": readings(),
            "mock_source": "hash",
        }
        samples["POST /api/v1/assess (水质单模态)"] = client.post("/api/v1/assess", json=body).json()

        body2 = dict(body)
        body2["readings"] = None
        body2["image_path"] = str(GILL_IMAGE)
        samples["POST /api/v1/assess (图像单模态)"] = client.post("/api/v1/assess", json=body2).json()

        # 双模态：水质 + 图像一起传，才是融合分和协同规则真正生效的路径。
        body3 = dict(body)
        body3["pond_id"] = "demo-pond-1"
        body3["image_path"] = str(GILL_IMAGE)
        samples["POST /api/v1/assess (双模态)"] = client.post("/api/v1/assess", json=body3).json()

        samples["POST /api/v1/tools/call"] = client.post(
            "/api/v1/tools/call",
            json={"tool": "water_risk_predict", "arguments": {"readings": readings()}},
        ).json()

        samples["POST /api/v1/ingest/sensor"] = client.post(
            "/api/v1/ingest/sensor",
            json={"pond_id": "POND-03", "readings": readings(4), "replace": False},
        ).json()
        samples["GET /api/v1/ponds/{pond_id}/readings"] = client.get(
            "/api/v1/ponds/POND-03/readings"
        ).json()
        samples["GET /api/v1/ponds/{pond_id}/history"] = client.get(
            "/api/v1/ponds/POND-03/history"
        ).json()

        # 图片上传：前端要对着做文件选择 + 预览，这条必须给。
        with GILL_IMAGE.open("rb") as fh:
            samples["POST /api/v1/ingest/image"] = client.post(
                "/api/v1/ingest/image",
                files={"file": (GILL_IMAGE.name, fh, "image/jpeg")},
            ).json()

        # 鱼种识别：可用性探测永远 200；识别则取决于本机有没有 ylov8-species.pt。
        # 没有权重时这一条给的是 503 的说明体（"鱼种识别不可用：…"），
        # 这也正是前端「隐藏鱼种块」时要读的那句话，比编一个假响应有用。
        samples["GET /api/v1/species"] = client.get("/api/v1/species").json()
        with GILL_IMAGE.open("rb") as fh:
            _species = client.post(
                "/api/v1/species/detect",
                files={"file": (GILL_IMAGE.name, fh, "image/jpeg")},
            )
        samples["POST /api/v1/species/detect"] = _species.json()

        samples["POST /api/v1/chat"] = client.post(
            "/api/v1/chat", json={"query": "溶氧低怎么办", "k": 3}
        ).json()

        samples["POST /api/v1/assess (错误：什么都没传，422)"] = client.post(
            "/api/v1/assess", json={}
        ).json()
        samples["POST /api/v1/chat (错误：query 太短，422)"] = client.post(
            "/api/v1/chat", json={"query": "x"}
        ).json()
        samples["GET /api/v1/ponds/{pond_id}/history (不存在的塘口)"] = client.get(
            "/api/v1/ponds/NOPE/history"
        ).json()
        samples["GET /api/v1/config/templates/{key}/csv (不存在的模板，404)"] = client.get(
            "/api/v1/config/templates/nope/csv"
        ).json()

    scrubbed = _scrub(samples)
    upload = scrubbed.get(_UPLOAD_SAMPLE) if isinstance(scrubbed, dict) else None
    if isinstance(upload, dict) and "vision" in upload:
        upload["vision"] = _VISION_POINTER
    OUT.write_text(json.dumps(scrubbed, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OPENAPI_OUT} ({OPENAPI_OUT.stat().st_size} bytes)")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(scrubbed)} samples)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
