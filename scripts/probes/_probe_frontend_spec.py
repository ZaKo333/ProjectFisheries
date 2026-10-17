"""导出前端对接所需的全部机器可验证事实。

输出：docs/fixtures/_frontend_spec.txt
- OpenAPI 各端点的请求/响应模型字段
- 工具层与融合层的全部枚举值（规则码 / 告警码 / 错误码 / 风险等级 / 病害标签）
- 关键常量阈值
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "fixtures" / "_frontend_spec.txt"

sys.path.insert(0, str(ROOT / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

buf = io.StringIO()

def w(*parts: object) -> None:
    buf.write(" ".join(str(p) for p in parts) + "\n")

def schema_fields(model: object) -> str:
    """把 pydantic 模型摊成 'name: type' 列表。"""
    out = []
    for name, field in model.model_fields.items():  # type: ignore[attr-defined]
        ann = getattr(field.annotation, "__name__", None) or str(field.annotation)
        req = "必填" if field.is_required() else "可选"
        out.append(f"    {name}: {ann}  [{req}]")
    return "\n".join(out)

def main() -> int:
    # ---- 1. OpenAPI ----
    from fisheries.api.app import create_app

    with __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(
        create_app(enable_cors=False)
    ) as client:
        spec = client.get("/openapi.json").json()

    w("=" * 78)
    w("一、OpenAPI 端点清单（GET /openapi.json）")
    w("=" * 78)
    for path, ops in sorted(spec["paths"].items()):
        for method, op in sorted(ops.items()):
            tags = ",".join(op.get("tags", []))
            w(f"\n{method.upper():6} {path}    [{tags}]")
            w(f"       摘要：{op.get('summary', '')}")
            # 请求体
            rb = op.get("requestBody", {})
            for ctype, media in rb.get("content", {}).items():
                ref = media.get("schema", {}).get("$ref", "")
                if ref:
                    w(f"       请求体({ctype})：{ref.rsplit('/', 1)[-1]}")
                else:
                    w(f"       请求体({ctype})：多部件表单")
                    for name, sub in (media.get("schema", {}).get("properties", {}) or {}).items():
                        ann = sub.get("type") or sub.get("anyOf") or sub.get("$ref", "")
                        w(f"           - {name}: {ann}")
            # 参数
            for param in op.get("parameters", []):
                w(f"       路径/查询参数：{param['name']} ({param['in']}, {'必填' if param.get('required') else '可选'})")
            # 响应
            for code, resp in sorted(op.get("responses", {}).items()):
                refs = []
                for media in (resp.get("content") or {}).values():
                    s = media.get("schema", {})
                    if "$ref" in s:
                        refs.append(s["$ref"].rsplit("/", 1)[-1])
                    elif s.get("type") == "array":
                        refs.append("array<" + s.get("items", {}).get("$ref", "?").rsplit("/", 1)[-1] + ">")
                    else:
                        refs.append(str(s.get("type", "?")))
                w(f"       响应 {code}：{resp.get('description', '')} -> {', '.join(refs) or '(无内容)'}")

    # ---- 2. 各响应模型的字段 ----
    w("\n" + "=" * 78)
    w("二、关键响应模型的字段（名字必须逐字对上）")
    w("=" * 78)
    from fisheries.api import schemas as api_schemas

    for model_name in (
        "AssessRequest", "AssessResponse", "ToolCallRequest", "ToolResultResponse",
        "SensorReadingIn", "IngestSensorRequest", "IngestSensorResponse",
        "IngestImageResponse", "ChatRequest", "ChatResponse", "ChatChunkOut",
        "HistoryResponse", "HealthResponse", "ThresholdResponse", "ToolSchemasResponse",
        "AdvisoryOut", "AlertOut", "EvidenceOut", "TraceStepOut", "WaterQualityPoint",
        "AdvisorReport",
    ):
        model = getattr(api_schemas, model_name, None)
        if model is None:
            w(f"\n{model_name}: 未找到")
            continue
        w(f"\n{model_name}:")
        w(schema_fields(model))

    # ---- 3. 工具层枚举 ----
    w("\n" + "=" * 78)
    w("三、工具层枚举与常量")
    w("=" * 78)
    from fisheries.tools import fish_disease as fd
    from fisheries.tools import water_risk as wr
    from fisheries.tools.definitions import TOOL_SCHEMAS

    w("\n[water_risk] 工具名: " + wr.water_risk_predict.__name__)
    w("  RISK_LEVELS: " + repr(["正常", "预警", "高危"]))
    w("  ACTION_THRESHOLDS:")
    for k, v in wr.ACTION_THRESHOLDS.items():
        w(f"    {k}: {v}")
    w("  REDLINE: " + json.dumps(wr.REDLINE, ensure_ascii=False))
    w("  MIN_RELIABLE_POINTS = " + str(wr.MIN_RELIABLE_POINTS))
    w("  MAX_EXTRAPOLATION_HOURS = " + str(wr.MAX_EXTRAPOLATION_HOURS))
    w("  SCORE_CRITICAL = " + str(wr.SCORE_CRITICAL) + " ; SCORE_WARNING = " + str(wr.SCORE_WARNING))
    w("  SPECIES_PROFILES keys: " + repr(list(wr.SPECIES_PROFILES.keys())))
    w("  BASE_WEIGHTS: " + json.dumps(wr.BASE_WEIGHTS, ensure_ascii=False))
    w("\n  WaterRiskOutput 字段:")
    w(schema_fields(wr.WaterRiskOutput))
    w("\n  DataQuality 字段:")
    w(schema_fields(wr.DataQuality))

    # 规则码
    import re

    src = (ROOT / "src" / "fisheries" / "tools" / "water_risk.py").read_text(encoding="utf-8")
    codes = sorted(set(re.findall(r'"([A-Z][A-Z0-9_]{3,})"', src)))
    w("\n  water_risk.py 里出现的全大写字符串字面量（规则码候选）：")
    for c in codes:
        w("    " + c)

    w("\n[fish_disease] 工具名: " + fd.fish_disease_detect.__name__)
    w("  DISEASE_LABELS: " + repr(list(fd.DISEASE_LABELS)))
    w("  LABEL_META:")
    for k, v in fd.LABEL_META.items():
        w(f"    {k}: {v}")
    w("  LOW_CONFIDENCE_THRESHOLD = " + str(getattr(fd, "LOW_CONFIDENCE_THRESHOLD", "?")))
    src2 = (ROOT / "src" / "fisheries" / "tools" / "fish_disease.py").read_text(encoding="utf-8")
    w("  fish_disease.py 错误码：")
    for c in sorted(set(re.findall(r'"(IMAGE_NOT_FOUND|NOT_A_FILE|UNSUPPORTED_IMAGE|MODEL_[A-Z_]+)"', src2))):
        w("    " + c)

    w("\n[definitions] TOOL_SCHEMAS:")
    items = TOOL_SCHEMAS.values() if isinstance(TOOL_SCHEMAS, dict) else TOOL_SCHEMAS
    for s in items:
        if isinstance(s, str):
            w("  " + s)
            continue
        fn = s.get("function", s)
        w(f"  {fn.get('name')}: required={fn.get('parameters', {}).get('required')}")
        for pname, pspec in fn.get("parameters", {}).get("properties", {}).items():
            w(f"      {pname}: {pspec.get('type')} — {pspec.get('description', '')[:60]}")

    # ---- 4. 融合层枚举 ----
    w("\n" + "=" * 78)
    w("四、融合层枚举")
    w("=" * 78)
    from fisheries.agent import fusion as fu

    w("W_WATER = " + str(fu.W_WATER) + " ; W_VISION = " + str(fu.W_VISION))
    w("SCORE_CRITICAL = " + str(fu.SCORE_CRITICAL) + " ; SCORE_WARNING = " + str(fu.SCORE_WARNING))
    w("SEVERE_DISEASE_WEIGHT = " + str(fu.SEVERE_DISEASE_WEIGHT) + " ; SEVERE_DISEASE_MIN_CONF = " + str(fu.SEVERE_DISEASE_MIN_CONF))
    w("DISEASE_MIN_SEVERITY_FOR_WARNING = " + str(fu.DISEASE_MIN_SEVERITY_FOR_WARNING))
    w("DISEASE_MIN_CONF_FOR_WARNING = " + str(fu.DISEASE_MIN_CONF_FOR_WARNING))
    w("SYNERGY_RULES:")
    for rule in fu.SYNERGY_RULES:
        w(f"    {rule}")
    w("\nFusionResult 字段:")
    w(schema_fields(fu.FusionResult))
    w("\nEvidence 字段:")
    w(schema_fields(fu.Evidence))
    w("\nAlert 字段:")
    w(schema_fields(fu.Alert))

    # ---- 5. Agent 层 ----
    w("\n" + "=" * 78)
    w("五、Agent 层输出结构")
    w("=" * 78)
    from fisheries.agent import orchestrator as orch

    w("Advisory 字段:")
    w(schema_fields(orch.Advisory))
    w("\nAdvisorReport 字段:")
    w(schema_fields(orch.AdvisorReport))
    w("\nAgentInput 字段:")
    w(schema_fields(orch.AgentInput))
    w("\nTraceStep 字段:")
    w(schema_fields(orch.TraceStep))
    w("\nAgentResult 字段:")
    w(schema_fields(orch.AgentResult))
    w("\nMAX_ROUNDS = " + str(orch.MAX_ROUNDS))

    # ---- 6. 环境变量 ----
    w("\n" + "=" * 78)
    w("六、环境变量")
    w("=" * 78)
    for fname in ("api/deps.py", "api/routes.py", "agent/llm.py", "agent/translate.py"):
        path = ROOT / "src" / "fisheries" / fname
        if not path.is_file():
            continue
        found = sorted(set(re.findall(r'"(FISHERIES_[A-Z0-9_]+)"', path.read_text(encoding="utf-8"))))
        if found:
            w(f"\n{fname}:")
            for f in found:
                w("    " + f)

    OUT.write_text(buf.getvalue(), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
