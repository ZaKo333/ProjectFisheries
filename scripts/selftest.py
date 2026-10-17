"""环境自检：跑单元测试 + 端到端集成测试 + 全部演示脚本 + 检查知识库索引。

这是「运行自检.bat」真正的实现。之所以从 .bat 搬进 Python：

1. `.bat` 里不能有非 ASCII 字节（cmd 会解析错位，见 `fisheries.launcher`
   的模块说明），自检要输出的中文全是提示文本，留在 .bat 里只能变成乱码噪声；
2. 演示脚本列表原来是写死在 .bat 里的 8 个名字，新增脚本就会漏跑，这里改成
   扫描 `scripts/demo_*.py`；
3. 逻辑进了 Python 才能被单元测试覆盖（`tests/unit/test_selftest.py`）。

用法：

    python scripts/selftest.py

退出码：0 全部通过；1 有失败项。
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fisheries import console  # noqa: E402  (必须在 sys.path 处理之后)

PY = sys.executable
_BAR = "  " + "-" * 60
_RULE = "  " + "=" * 60

_OPTIONAL_DEPS = (
    ("fastapi", "接口服务（含托管的前端页面）不可用", 'pip install -e ".[api]"'),
    ("chromadb", "知识库不可用", 'pip install -e ".[rag]"'),
    ("ultralytics", "病害识别走模拟兜底（功能正常，只是不调用真实模型）", None),
)


def _run(cmd: list[str], *, quiet: bool = False) -> int:
    kwargs: dict[str, object] = {"cwd": str(PROJECT_ROOT)}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.STDOUT
    return subprocess.run(cmd, **kwargs).returncode  # type: ignore[call-overload]


def step_python_env() -> bool:
    print("  [1/5] 检查 Python 环境")
    print(f"        Python {sys.version.split()[0]}  ({PY})")
    try:
        import fisheries  # noqa: F401
    except Exception as exc:  # pragma: no cover - 环境损坏才会走到
        print(f"        [失败] 项目包 fisheries 无法导入：{exc!r}")
        print("        在本目录执行： .venv\\Scripts\\python.exe -m pip install -e .")
        return False
    print("        项目包 fisheries 可导入")
    return True


def step_optional_deps() -> None:
    print("  [2/5] 检查可选依赖")
    import importlib.util

    for name, missing_note, fix in _OPTIONAL_DEPS:
        installed = importlib.util.find_spec(name) is not None
        status = "已安装" if installed else "未安装"
        line = f"        {name:<12} {status}"
        if not installed:
            line += f"  -> {missing_note}"
        print(line)
        if not installed and fix:
            print(f"                     执行: {fix}")

    _report_weights()


def _report_weights() -> None:
    """把模型权重的有无说清楚——缺了会走 mock，绝不允许静默降级。"""
    from fisheries import weights as wt

    print("        模型权重：")
    for row in wt.status():
        item = wt.spec(row["name"])
        if not row["exists"]:
            print(f"          [缺失] {row['name']:<14} {row['file']}")
            if item.pinned:
                print(f"                 {wt.MOCK_NOTICE}（{item.label}）")
                print(f"                 执行: python scripts/fetch_datasets.py --fetch-weights {row['name']}")
            continue
        size = f"{row['bytes'] / 1048576:.1f} MB"
        note = "" if item.pinned else "（仅重训需要）"
        print(f"          [就绪] {row['name']:<14} {size:>8}  {row['file']}{note}")


def step_knowledge_base() -> bool:
    print("  [3/5] 检查知识库索引")
    # 向量库不入库（.gitignore 里），全新克隆第一次跑自检时它本来就不存在。
    # 这是「还没构建」而不是「坏了」，不该让整个自检失败；但也不能闷声跳过，
    # 得把下一步该跑什么说清楚。库存在却和语料对不上，才是真失败。
    if not (PROJECT_ROOT / "knowledge" / "chroma" / "fisheries_manifest.json").exists():
        print("        [跳过] 向量库尚未构建（构建前知识问答不可用，其余功能正常）")
        print("        构建: python scripts/build_knowledge_base.py")
        return True
    rc = _run([PY, "scripts/build_knowledge_base.py", "--check"])
    if rc != 0:
        print(f"        [失败] 知识库检查退出码 {rc}")
        return False
    return True


def step_unit_tests() -> bool:
    print("  [4/5] 运行测试：单元测试 + 端到端集成测试")
    print(_BAR)
    # 两层分开跑、分开报：单元测试是「代码对不对」，集成测试是「串起来还成不成立」。
    # 集成测试缺权重/缺索引会自己 skip（reason 里写补法），不会误判成失败。
    rc_unit = _run([PY, "-m", "pytest", "tests/unit", "-o", "addopts=", "-q"])
    print(_BAR)
    rc_it = _run([PY, "-m", "pytest", "tests/integration", "-o", "addopts=", "-q"])
    print(_BAR)
    ok = True
    if rc_unit != 0:
        print(f"  单元测试未全部通过，退出码 {rc_unit}")
        ok = False
    else:
        print("  单元测试全部通过。")
    if rc_it != 0:
        print(f"  集成测试未全部通过，退出码 {rc_it}")
        ok = False
    else:
        print("  集成测试全部通过（缺权重/缺索引的用例会自动跳过并说明补法）。")
    return ok


def demo_scripts() -> list[pathlib.Path]:
    return sorted((PROJECT_ROOT / "scripts").glob("demo_*.py"))


def step_demos() -> bool:
    scripts = demo_scripts()
    print("  [5/5] 跑一遍全部演示脚本（每个都应正常结束）")
    print(_BAR)
    failed: list[str] = []
    for path in scripts:
        rc = _run([PY, str(path.relative_to(PROJECT_ROOT))], quiet=True)
        if rc != 0:
            print(f"     [失败] {path.name}  (退出码 {rc})")
            failed.append(path.name)
        else:
            print(f"     [通过] {path.name}")
    print(_BAR)
    if failed:
        print(f"  {len(failed)} 个演示脚本失败：{', '.join(failed)}")
        return False
    print(f"  {len(scripts)} 个演示脚本全部正常。")
    return True


def main() -> int:
    console.install()
    print()
    print(_RULE)
    print("    ProjectFisheries 环境自检")
    print(_RULE)
    print()

    ok = step_python_env()
    print()
    if ok:
        step_optional_deps()
        print()
        ok = step_knowledge_base()
        print()
    if ok:
        ok = step_unit_tests()
        print()
    if ok:
        ok = step_demos()

    print()
    print(_RULE)
    if ok:
        print("    自检通过：测试全绿、演示脚本全部正常。")
        print("    接口服务 + 前端页面   双击 启动接口服务.bat")
        print("    （启动后打开横幅里给出的地址，页面与接口是同一个服务）")
    else:
        print("    自检未通过，请把上面的 [失败] 行连同报错贴给我。")
    print(_RULE)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
