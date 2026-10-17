"""守卫「单元测试 / 集成测试」这条分层线本身。

用户要求两层分开统计（`pytest -m "not integration"` 只跑单元层）。这条约定靠的是
每个集成模块顶上那句 `pytestmark = pytest.mark.integration`——**漏写一句，它就会被
当成单元测试跑**：在新克隆的机器上要等模型加载、还可能因为缺权重而变红。
所以这里把约定本身钉住。
"""

from __future__ import annotations

from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_DIR = PROJECT_ROOT / "tests" / "integration"
MARKER = "pytest.mark.integration"


def test_pyproject_declares_the_integration_marker() -> None:
    """`-m integration` 要能用，marker 必须先注册，否则 pytest 会只给一条警告。"""
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("integration:") for m in markers), markers
    # 两层都得在 testpaths 覆盖范围内，否则 `pytest` 一把梭会漏跑集成层
    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]


def test_every_integration_module_is_marked() -> None:
    """集成目录下每个 `test_*.py` 都必须带 marker，且带 `conftest.py`。"""
    assert (INTEGRATION_DIR / "conftest.py").is_file(), "集成测试缺 conftest.py"
    modules = sorted(INTEGRATION_DIR.glob("test_*.py"))
    assert len(modules) >= 5, [p.name for p in modules]
    missing = [p.name for p in modules if MARKER not in p.read_text(encoding="utf-8")]
    assert not missing, f"这些集成模块漏了 `pytestmark = pytest.mark.integration`：{missing}"
