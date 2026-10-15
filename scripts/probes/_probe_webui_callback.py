"""探针：直接调用 Gradio 界面真正的回调函数 ``app._run``。

为什么不用假装点页面：``scripts/demo_webui.py`` 走的是 ``WebSession.run``，
而页面上「运行 Agent」按钮绑的是 ``app._run``——它会先经 ``_load_table``
处理「上传文件 / 粘贴文本」两条输入路径，再拼 ``WebRequest``，最后按
``outputs`` 的顺序返回一个**定长元组**。这个元组的长度必须和组件数严格一致，
差一个 Gradio 会抛 "The number of outputs does not match"。

探针要确认的就是这些接线问题：
    1. 上传文件的路径能跑通；2. 粘贴文本的路径能跑通；
    3. 两条路都故意给错数据时，返回的仍是完整长度的元组，而不是抛异常；
    4. 三个输入源全空时报错信息是人话。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fisheries import datasets as _datasets  # noqa: E402
# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

from fisheries.webui.app import _load_table, _run, build_app  # noqa: E402
from fisheries.webui.runner import WebSession, example_table_path  # noqa: E402
from fisheries.webui.table import TableError  # noqa: E402

DEMO_IMAGE = _datasets.resolve(
    "fish_disease", "Test", "Bacterial gill disease", "Bacterial gill disease (100).jpg"
)

ok = True


def check(label: str, cond: bool, extra: str = "") -> None:
    global ok
    mark = "✅" if cond else "❌"
    if not cond:
        ok = False
    print(f"{mark} {label}{('  → ' + extra) if extra else ''}")


session = WebSession(use_knowledge=True)
table_path = example_table_path()

print("=" * 78)
print("  1. 上传文件路径")
print("=" * 78)
out = _run(
    str(table_path), "", None, "probe-pond", "鲤科", "hash", True, 288, session,
)
check("返回定长元组（8 项）", len(out) == 8, f"len={len(out)}")
check("大结论非空", bool(out[0].strip()))
check("建议里出现「处置建议」", "处置建议" in out[2])
check("接入质量面板有内容", "列映射" in out[3] or "可用" in out[3])

print()
print("=" * 78)
print("  2. 粘贴文本路径（与上传文件互斥，粘贴优先）")
print("=" * 78)
pasted = table_path.read_text(encoding="utf-8")
out_paste = _run(
    str(table_path), pasted, None, "probe-pond", "鲤科", "hash", True, 288, session,
)
check("返回定长元组", len(out_paste) == 8)
check("接入质量注明来源是粘贴", "粘贴" in out_paste[3])

print()
print("=" * 78)
print("  3. 双模态（表格 + 真实病害图片）")
print("=" * 78)
if DEMO_IMAGE.is_file():
    out2 = _run(
        str(table_path), "", str(DEMO_IMAGE), "probe-pond", "鲤科", "hash",
        True, 288, session,
    )
    check("返回定长元组", len(out2) == 8)
    check("融合依据有内容", "融合" in out2[5])
    check("图片识别面板有内容", "标签" in out2[4] or "置信" in out2[4])
    check("推理链路含工具调用", "调用工具" in out2[7])
else:
    print("（跳过：演示图片不存在）")

print()
print("=" * 78)
print("  4. 错误路径必须返回完整元组，而不是抛异常")
print("=" * 78)
for label, kwargs in (
    ("三个输入源全空", dict(file_value=None, pasted="")),
    ("粘贴内容只有一行", dict(file_value=None, pasted="只有一行")),
    ("上传的是图片而不是表格", dict(file_value=str(DEMO_IMAGE), pasted="")),
):
    try:
        o = _run(kwargs["file_value"], kwargs["pasted"], None, "p", "鲤科",
                 "hash", True, 288, session)
        # render_error 输出的是「运行失败」红块；其余 7 栏必须是空串，
        # 否则上一次的结果会残留在页面上，用户会以为新数据也报了同样的警。
        check(
            label,
            len(o) == 8 and "运行失败" in o[0] and all(x == "" for x in o[1:]),
            f"len={len(o)}",
        )
    except Exception as exc:  # noqa: BLE001
        check(label, False, f"抛异常了：{type(exc).__name__}: {exc}")

print()
print("=" * 78)
print("  5. _load_table 的独立行为")
print("=" * 78)
try:
    _load_table(None, "", 288)
    check("空输入应报错", False)
except TableError as exc:
    check("空输入给出人话提示", "上传" in str(exc) or "粘贴" in str(exc), str(exc)[:40])

ing = _load_table(None, pasted, 12)
check("粘贴路径尊重 max_rows", len(ing.readings) == 12, f"n={len(ing.readings)}")

print()
print("=" * 78)
print("  6. 组件树")
print("=" * 78)
demo = build_app(session)
check("界面可构建", demo is not None)
check("组件数 > 0", len(demo.blocks) > 0, f"n={len(demo.blocks)}")

print()
print("=" * 78)
print(f"  结果：{'全部通过' if ok else '有失败项'}")
print("=" * 78)
raise SystemExit(0 if ok else 1)
