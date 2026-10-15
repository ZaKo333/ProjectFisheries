"""ProjectFisheries —— 后端接口（FastAPI）演示。

用 ``fastapi.testclient.TestClient`` 在**进程内**跑完整请求链路，
不需要先起 uvicorn，也不占用端口。

    python scripts/demo_api.py                 # 全部场景
    python scripts/demo_api.py --only assess   # 只跑某一个场景
    python scripts/demo_api.py --show-trace    # /assess 额外打印推理轨迹

覆盖场景：
    1. overview  —— /health + /config/*（看看服务到底降级在哪、阈值表长什么样）
    2. assess    —— 核心链路：24 条恶化水质 + 真实烂鳃图 → 等级/告警/建议/依据
    3. ingest    —— 传感器缓冲：/ingest/sensor → /readings → 不带 readings 直接 /assess
    4. upload    —— 图片上传：真图 200 / 假图 422 / 超限 413 / 不落地脏文件
    5. chat      —— 知识库问答 + /tools/call 直调工具
    6. errors    —— 契约护栏：字段拼错 422、读数太少 422、未知 upload_id 404
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from pathlib import Path
from typing import Any

# FastAPI/Starlette 1.8 起把 422/413 常量改了名，会在 stderr 刷一堆
# StarletteDeprecationWarning，被 PowerShell 渲染成红色错误块，盖住演示内容。
# 这不是本项目的问题（代码里用的是 `status.HTTP_422_...` 的推荐写法），
# 演示脚本里屏蔽掉即可。
warnings.filterwarnings("ignore", category=DeprecationWarning)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries import datasets as _datasets  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fisheries.api.app import create_app  # noqa: E402
from fisheries.api.deps import ApiSettings  # noqa: E402

BAR = "=" * 78
DATASET = _datasets.resolve("fish_disease", "Test")
GILL_IMAGE = DATASET / "Bacterial gill disease" / "Bacterial gill disease (100).jpg"

# --------------------------------------------------------------------------- #
# 构造演示数据
# --------------------------------------------------------------------------- #

def crashing_water(n: int = 24) -> list[dict[str, Any]]:
    """24 条逐小时读数：溶氧 6.2 → 2.6、氨氮 0.15 → 0.62、亚硝酸盐 0.05 → 0.15。"""
    out: list[dict[str, Any]] = []
    for i in range(n):
        r = i / (n - 1)
        out.append({
            "timestamp": f"2026-05-11T{i:02d}:00:00",
            "water_temp": round(28.0 + 1.5 * r, 2),
            "ph": round(8.05 + 0.20 * r, 3),
            "dissolved_oxygen": round(6.2 - 3.6 * r, 3),
            "ammonia_n": round(0.15 + 0.47 * r, 3),
            "nitrite_n": round(0.05 + 0.10 * r, 3),
        })
    return out

def good_water(n: int = 24) -> list[dict[str, Any]]:
    """24 条平稳读数，用于对比。"""
    out: list[dict[str, Any]] = []
    for i in range(n):
        out.append({
            "timestamp": f"2026-05-11T{i:02d}:00:00",
            "water_temp": 26.0,
            "ph": 7.6,
            "dissolved_oxygen": 6.4,
            "ammonia_n": 0.10,
            "nitrite_n": 0.04,
        })
    return out

def _show(resp: Any, *, limit: int = 12) -> None:
    print(f"  HTTP {resp.status_code}")
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001 - 非 JSON 响应
        print(f"  {resp.text[:200]}")
        return
    text = json.dumps(body, ensure_ascii=False, indent=2)
    lines = text.splitlines()
    for line in lines[:limit]:
        print(f"  {line}")
    if len(lines) > limit:
        print(f"  ...（共 {len(lines)} 行，此处只展示前 {limit} 行）")

# --------------------------------------------------------------------------- #
# 场景
# --------------------------------------------------------------------------- #

def case_overview(client: TestClient) -> None:
    print(BAR)
    print("场景一  服务自检与配置（/health + /config/*）")
    print(BAR)

    print("\n[GET /api/v1/health]")
    health = client.get("/api/v1/health").json()
    print(f"  status      = {health['status']}")
    print(f"  version     = {health.get('version')}")
    print(f"  tools       = ")
    for tname, tdesc in (health.get("tools") or {}).items():
        print(f"      · {tname}: {tdesc}")
    kb = health.get("knowledge") or {}
    print(f"  knowledge   = status={kb.get('status')} "
          f"chunks={kb.get('n_chunks')} backend={kb.get('backend')}")
    for note in health.get("notes") or []:
        print(f"  note        · {note}")
    for reason in health.get("degraded_reasons") or []:
        print(f"  degraded    · {reason}")
    print("\n  ⚠ status=degraded 是**如实上报**：图像模型尚未训练，工具走 mock 兜底。")

    print("\n[GET /api/v1/config/thresholds]")
    th = client.get("/api/v1/config/thresholds").json()
    print(f"  species     = {list(th['profiles'])}")
    w = th["weights"]
    print(f"  weights     = { {k: round(v, 3) for k, v in w.items()} }")
    print(f"  redlines    = {th['redlines']}")
    print(f"  溶氧倒计时线 = {th['action_thresholds'].get('dissolved_oxygen')}")
    print(f"  外推上限     = {th['extrapolation_horizon_hours']} 小时"
          f"（线性外推只报这么远）")

    print("\n[GET /api/v1/config/tools]  —— 导出的 function-calling schema")
    tools = client.get("/api/v1/config/tools").json()["tools"]
    for t in tools:
        fn = t["function"]
        print(f"  · {fn['name']}  必填={fn['parameters'].get('required')}")

def case_assess(client: TestClient, show_trace: bool) -> None:
    print(BAR)
    print("场景二  核心链路 /api/v1/assess（水质时序 + 鱼体图片）")
    print(BAR)

    payload: dict[str, Any] = {
        "pond_id": "P01",
        "species": "鲤科",
        "readings": crashing_water(),
        "image_path": str(GILL_IMAGE),
        "mock_source": "folder_hint",
        "include_trace": show_trace,
        "include_knowledge_block": False,
    }
    print(f"\n  请求：24 条恶化读数（溶氧 6.2→2.6）+ 真实烂鳃图片")
    resp = client.post("/api/v1/assess", json=payload)
    if resp.status_code != 200:
        _show(resp)
        return
    body = resp.json()

    print(f"\n  HTTP {resp.status_code}")
    print(f"  risk_level  = {body['risk_level']}    fused_score = {body['fused_score']}")
    print(f"  headline    = {body['report']['headline']}")
    print(f"  reason      = {body['reason'][:110]}…")

    print(f"\n  告警 {len(body['alerts'])} 条：")
    for a in body["alerts"]:
        print(f"    [{a['level']:8s}] {a['title']}：{a['detail'][:60]}")

    print(f"\n  建议（水质 {len(body['report']['water_actions'])} 条 / "
          f"病害 {len(body['report']['disease_actions'])} 条）：")
    for a in body["report"]["water_actions"]:
        print(f"    {a['priority']}. [{a['target']}] {a['action']}（{a['timing']}）")
    for a in body["report"]["disease_actions"]:
        print(f"    {a['priority']}. [{a['target']}] {a['action']}（{a['timing']}）")
    print(f"  复查窗口    = {body['report']['observation_window']}")

    print(f"\n  引用的规范依据（RAG 检索 {len(body['references'])} 条）：")
    for ref in body["references"]:
        print(f"    · {ref}")

    ks = body.get("knowledge_summary") or {}
    print(f"\n  知识库：backend={ks.get('backend')} chunks={ks.get('n_chunks')} "
          f"purposes={ks.get('purposes')}")
    for wmsg in ks.get("warnings") or []:
        print(f"    ⚠ {wmsg}")

    if show_trace:
        print(f"\n  推理轨迹：")
        for s in body["trace"]:
            print(f"    {s['step']:2d}. [{s['kind']:11s}] {s['title']}")

    print(f"\n  → 结论：一次 HTTP 请求就拿到「等级 + 告警 + 处置 + 依据 + 轨迹」，"
          f"\n     且依据可追溯到具体小节，不是模型凭空说的。")

def case_ingest(client: TestClient) -> None:
    print(BAR)
    print("场景三  传感器缓冲（/ingest/sensor → /readings → /assess 不带 readings）")
    print(BAR)

    # 用独立塘口，避免和场景二的状态互相干扰
    print("\n[POST /api/v1/ingest/sensor]  分批上报 24 条读数")
    r1 = client.post("/api/v1/ingest/sensor", json={
        "pond_id": "P02", "readings": good_water()[:12]})
    r2 = client.post("/api/v1/ingest/sensor", json={
        "pond_id": "P02", "readings": good_water()[12:]})
    print(f"  第 1 批 HTTP {r1.status_code} → {r1.json()}")
    print(f"  第 2 批 HTTP {r2.status_code} → {r2.json()}")

    print("\n[GET /api/v1/ponds/P02/readings?limit=3]")
    got = client.get("/api/v1/ponds/P02/readings", params={"limit": 3}).json()
    print(f"  total       = {got['total']}")
    print(f"  最新 3 条    = {got['readings']}")

    print("\n[POST /api/v1/assess]  不传 readings，让服务从缓冲里取")
    resp = client.post("/api/v1/assess", json={
        "pond_id": "P02", "use_buffer": True, "include_trace": False,
        "include_knowledge_block": False})
    body = resp.json()
    print(f"  HTTP {resp.status_code} → risk_level={body['risk_level']} "
          f"fused={body['fused_score']}")
    print(f"  → 水质平稳，等级应为「正常」，说明缓冲里的 24 条真的被用上了。")

    print("\n[POST /api/v1/assess]  缓冲里没有的塘口")
    miss = client.post("/api/v1/assess", json={
        "pond_id": "P-NOT-EXIST", "use_buffer": True})
    print(f"  HTTP {miss.status_code} → {miss.json().get('detail')}")

def case_upload(client: TestClient) -> None:
    print(BAR)
    print("场景四  图片上传（/ingest/image）")
    print(BAR)

    print(f"\n[POST /api/v1/ingest/image]  真图（{GILL_IMAGE.name}）")
    if GILL_IMAGE.exists():
        with GILL_IMAGE.open("rb") as fh:
            up = client.post("/api/v1/ingest/image",
                             files={"file": (GILL_IMAGE.name, fh, "image/jpeg")})
        u = up.json()
        print(f"  HTTP {up.status_code}")
        print(f"  upload_id   = {u['upload_id']}")
        print(f"  stored_path = {u['stored_path']}")
        print(f"  识别结果     = {u['vision']['disease_type']} "
              f"（置信度 {u['vision']['confidence']}，"
              f"degraded={u['vision']['degraded_mode']}）")
        print(f"  → 图像模型未训练，此处走 mock（默认 hash 播种，与文件夹名无关），"
              f"\n     所以这张烂鳃图被随机判成了「{u['vision']['disease_type']}」。"
              f"\n     模型到位后同一接口返回真实推理结果，接口契约不变。")
        print(f"  → 文件名由服务端 uuid 生成，客户端传的 {GILL_IMAGE.name} 不落盘，"
              f"\n     路径穿越在源头就不成立。")

        print(f"\n[POST /api/v1/assess]  用 upload_id（推荐姿势：不暴露服务器路径）")
        a = client.post("/api/v1/assess", json={
            "pond_id": "P01", "readings": crashing_water(),
            "upload_id": u["upload_id"], "include_trace": False,
            "include_knowledge_block": False})
        ab = a.json()
        print(f"  HTTP {a.status_code} → {ab['risk_level']} / {ab['fused_score']}")
    else:
        print(f"  ⚠ 找不到测试图 {GILL_IMAGE}，跳过")

    print("\n[POST /api/v1/ingest/image]  假图（扩展名合法、内容不是图片）")
    fake = client.post("/api/v1/ingest/image",
                       files={"file": ("trick.jpg", io.BytesIO(b"not an image"),
                                       "image/jpeg")})
    print(f"  HTTP {fake.status_code} → {fake.json().get('detail')}")
    print(f"  → 扩展名只做粗筛，真格式由工具的魔数校验判定。")

    print("\n[POST /api/v1/ingest/image]  扩展名不在白名单")
    bad_ext = client.post("/api/v1/ingest/image",
                          files={"file": ("notes.txt", io.BytesIO(b"hello"),
                                          "text/plain")})
    print(f"  HTTP {bad_ext.status_code} → {bad_ext.json().get('detail')}")

    print("\n[POST /api/v1/ingest/image]  超过 10 MB 上限")
    big = client.post("/api/v1/ingest/image",
                      files={"file": ("big.jpg", io.BytesIO(b"\xff" * (11 * 1024 * 1024)),
                                      "image/jpeg")})
    print(f"  HTTP {big.status_code} → {big.json().get('detail')}")
    print(f"  → 失败的上传不会在磁盘留下垃圾文件（见 tests/unit/test_api.py）。")

def case_chat(client: TestClient) -> None:
    print(BAR)
    print("场景五  知识库问答（/chat）与工具直调（/tools/call）")
    print(BAR)

    for q, extra in (
        ("溶氧低于多少要开增氧机", {}),
        ("水霉病怎么处理", {"disease": "水霉", "category": "病害"}),
    ):
        print(f"\n[POST /api/v1/chat]  {q}  {extra or ''}")
        resp = client.post("/api/v1/chat", json={
            "query": q, "k": 3, "answer_with_llm": False, **extra})
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code} → {resp.json().get('detail')}")
            continue
        body = resp.json()
        print(f"  HTTP {resp.status_code}  "
              f"answer_is_raw_retrieval={body['answer_is_raw_retrieval']}")
        for c in body["chunks"]:
            print(f"    · [{c['score']:.3f}] 《{c['title']}》> {c['heading_text']}"
                  f"{'  ⚠未核对' if not c['verified'] else ''}")
        for wmsg in body.get("warnings") or []:
            print(f"    ⚠ {wmsg}")
    print("\n  → 未配大模型时接口**不编回答**，直接把检索原文交出来并置"
          "\n     answer_is_raw_retrieval=True：让调用方分得清「模型说的」和「系统查到的」。")

    print("\n[POST /api/v1/tools/call]  跳过 LLM 直调工具")
    resp = client.post("/api/v1/tools/call", json={
        "tool": "water_risk_predict",
        "arguments": {"readings": crashing_water(), "pond_id": "P01"},
    })
    body = resp.json()
    print(f"  HTTP {resp.status_code} → {body['tool']} / "
          f"{body['result']['risk_level']} / {body['result']['risk_score']}")

    print("\n[POST /api/v1/tools/call]  参数写错 —— 传了不存在的参数名")
    bad = client.post("/api/v1/tools/call", json={
        "tool": "water_risk_predict",
        "arguments": {"readings": crashing_water(), "pond": "P01"}})
    print(f"  HTTP {bad.status_code} → {str(bad.json().get('detail'))[:100]}…")
    print(f"  → 工具签名不认识 `pond`，直接 422 把真实原因回给调用方。")

    print("\n[POST /api/v1/tools/call]  参数类型不对 —— readings 不是列表")
    bad2 = client.post("/api/v1/tools/call", json={
        "tool": "water_risk_predict", "arguments": {"readings": "不是列表"}})
    b2 = bad2.json()
    print(f"  HTTP {bad2.status_code}")
    print(f"  tool   = {b2.get('tool')}")
    print(f"  result = status={b2.get('result', {}).get('status')} "
          f"error_code={b2.get('result', {}).get('error_code')}")
    print(f"  → 这里**故意**返回 200：工具的契约是「出错也不抛异常」，"
          f"\n     而是返回 status=error + 人话错误信息，"
          f"\n     这样 Agent 循环不会因为一次脏输入断链。")

def case_errors(client: TestClient) -> None:
    print(BAR)
    print("场景六  契约护栏（这些才是把 demo 变成系统的东西）")
    print(BAR)

    print("\n[字段拼错] dissolved_oxygen 写成 dissolved_oxyen")
    r = client.post("/api/v1/assess", json={
        "pond_id": "P01",
        "readings": [{**crashing_water()[0], "dissolved_oxyen": 5.0}],
    })
    detail = r.json().get("detail") or []
    msg = detail[0].get("msg") if isinstance(detail, list) and detail else detail
    print(f"  HTTP {r.status_code} → {msg}")
    print(f"  → 契约是 extra=\"forbid\"：拼错**报错**，而不是静默当成"
          f"\n     「没传溶氧」再算出一个没有溶氧的风险分。")

    print("\n[读数太少] 只传 2 条")
    r2 = client.post("/api/v1/assess", json={
        "pond_id": "P01", "readings": crashing_water()[:2]})
    d2 = r2.json().get("detail") or []
    m2 = d2[0].get("msg") if isinstance(d2, list) and d2 else d2
    print(f"  HTTP {r2.status_code} → {m2}")
    print(f"  → 少于 3 条直接拒绝：否则工具会在样本不足时降级，"
          f"\n     调用方看到的是一个基于空水质的「正常」——假结论比报错更危险。")

    print("\n[未知 upload_id]")
    r3 = client.post("/api/v1/assess", json={"pond_id": "P01", "upload_id": "deadbeef"})
    print(f"  HTTP {r3.status_code} → {r3.json().get('detail')}")

    print("\n[两模态都没给]")
    r4 = client.post("/api/v1/assess", json={"pond_id": "P01"})
    d4 = r4.json().get("detail") or []
    m4 = d4[0].get("msg") if isinstance(d4, list) and d4 else d4
    print(f"  HTTP {r4.status_code} → {m4}")

# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="ProjectFisheries 后端接口演示")
    ap.add_argument("--only", default="all",
                    choices=["all", "overview", "assess", "ingest",
                             "upload", "chat", "errors"])
    ap.add_argument("--show-trace", action="store_true",
                    help="/assess 额外打印推理轨迹")
    args = ap.parse_args()

    print(BAR)
    print("ProjectFisheries —— 后端接口（FastAPI）演示")
    print(BAR)
    print("  说明：用 TestClient 在进程内跑完整 HTTP 链路，不需要先起 uvicorn。")
    print("        上传目录指向系统临时目录，不会往仓库里堆文件。")

    import tempfile

    with tempfile.TemporaryDirectory(prefix="fisheries_demo_") as td:
        settings = ApiSettings(upload_dir=Path(td) / "uploads",
                               allow_local_path=True)
        app = create_app(settings)
        with TestClient(app) as client:
            jobs = {
                "overview": lambda: case_overview(client),
                "assess": lambda: case_assess(client, args.show_trace),
                "ingest": lambda: case_ingest(client),
                "upload": lambda: case_upload(client),
                "chat": lambda: case_chat(client),
                "errors": lambda: case_errors(client),
            }
            chosen = list(jobs) if args.only == "all" else [args.only]
            for name in chosen:
                print()
                jobs[name]()

    print(f"\n{BAR}")
    print("  结论：接口层不重新实现业务逻辑——它只做四件事：")
    print("        校验入参、调用已实现的工具/Agent、把结果转成稳定契约、")
    print("        把异常转成明确的 HTTP 状态码。")
    print(BAR)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
