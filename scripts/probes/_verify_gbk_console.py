"""复验：脚本在 GBK 控制台下是否**整段**输出可读中文（不再混编码）。

为什么需要这个探针
------------------
``import ultralytics`` 会把 ``sys.stdout`` 的编码从 gbk 改成 utf-8。脚本若
只在开头调一次 ``console.install()``，输出就会前半段 GBK、后半段 UTF-8：
终端上加载模型之后整段变乱码，而且**退出码是 0**，没有任何报错。

所以判定标准不是"脚本跑通了"，而是"整份 stdout 能被 **一种** 编码解出来"。
本探针按行统计 gbk-only / utf8-only，任何 utf8-only 都算失败。

跑法（不需要 PYTHONPATH）::

    .venv\\Scripts\\python.exe scripts\\probes\\_verify_gbk_console.py

用 ``PYTHONIOENCODING=gbk`` 起子进程模拟中文 Windows 的默认控制台。
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Windows 控制台默认是 GBK 编码页，结论行里的 → 编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console  # noqa: E402

_console.install()

PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

#: (说明, argv)。挑的都是"会打印中文、且覆盖不同输出路径"的脚本。
CASES: list[tuple[str, list[str]]] = [
    ("知识库体检", [PY, "scripts/build_knowledge_base.py", "--check"]),
    ("数据集体检", [PY, "scripts/fetch_datasets.py", "--status"]),
    ("工具层演示", [PY, "scripts/demo_tools.py"]),
    ("边界案例", [PY, "scripts/demo_cases.py"]),
    ("Agent 演示", [PY, "scripts/demo_agent.py"]),
    ("训练体检（导入 ultralytics）", [PY, "scripts/train_yolo_fish.py", "--report"]),
    ("AST 体检", [PY, "scripts/probes/_check_eval_syntax.py"]),
]


def classify(raw: bytes) -> tuple[int, int, int, list[str]]:
    """返回 (可解为GBK行数, 仅UTF-8行数, 两者都可行数, 仅UTF-8样例)."""
    n_gbk = n_utf8 = n_both = 0
    samples: list[str] = []
    for line in raw.split(b"\r\n"):
        if not line or not any(b > 127 for b in line):
            continue
        try:
            line.decode("gbk")
            gbk_ok = True
        except UnicodeDecodeError:
            gbk_ok = False
        try:
            line.decode("utf-8")
            utf8_ok = True
        except UnicodeDecodeError:
            utf8_ok = False
        if gbk_ok and utf8_ok:
            n_both += 1
        elif utf8_ok:
            n_utf8 += 1
            if len(samples) < 2:
                samples.append(repr(line[:70]))
        elif gbk_ok:
            n_gbk += 1
    return n_gbk, n_utf8, n_both, samples


def main() -> int:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "gbk"  # 中文 Windows 的默认控制台
    env.pop("PYTHONUTF8", None)
    failures: list[str] = []

    for label, argv in CASES:
        proc = subprocess.run(argv, capture_output=True, env=env, cwd=str(ROOT), check=False)
        n_gbk, n_utf8, n_both, samples = classify(proc.stdout)
        verdict = "OK" if (n_utf8 == 0 and proc.returncode == 0) else "FAIL"
        print(
            f"[{verdict}] {label}: exit={proc.returncode} "
            f"gbk-only={n_gbk} utf8-only={n_utf8} both={n_both} bytes={len(proc.stdout)}"
        )
        if n_utf8:
            for s in samples:
                print(f"       utf8-only 行: {s}")
        if proc.returncode != 0:
            tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
            for line in tail:
                print(f"       stderr: {line}")
        if verdict == "FAIL":
            failures.append(label)

    print()
    if failures:
        print(f"结论：{len(failures)} 个脚本输出混编码或非零退出 → {failures}")
        return 1
    print(f"结论：{len(CASES)} 个脚本全部在 GBK 控制台下输出可读中文，零混编码")
    return 0


if __name__ == "__main__":
    sys.exit(main())
