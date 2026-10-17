"""一键配置环境脚本的契约测试。

被测的是 ``scripts/setup_env.py`` 里那些「不碰网络、不碰磁盘」的判断逻辑：
版本门槛、extras 解析、pip 参数拼接、镜像映射，以及它和 ``pyproject.toml``
的一致性——改了 extras 名字却忘了改脚本，这里会红。

真正跑 venv / pip / 建向量库的那几步需要网络和几分钟，属于手工验收，
不放在单测里（``--skip-install --skip-kb`` 就能单独把它们摘掉）。

另一个重点是两个入口文件：``setup.bat`` 必须纯 ASCII + CRLF（cmd.exe 的限制，
见 :mod:`fisheries.launcher`），``一键配置环境.bat`` 必须转发到它而不是各写一份。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETUP_PATH = PROJECT_ROOT / "scripts" / "setup_env.py"
SETUP_BAT = PROJECT_ROOT / "setup.bat"
FORWARD_BAT = PROJECT_ROOT / "一键配置环境.bat"


def _load_setup_module():
    """按文件路径载入脚本——它不是包的一部分，不能用 import。"""
    spec = importlib.util.spec_from_file_location("setup_env_under_test", SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


setup_env = _load_setup_module()


# --------------------------------------------------------------------------- #
# 1. Python 版本门槛
# --------------------------------------------------------------------------- #

class TestPythonGate:
    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ((2, 7), False),
            ((3, 9), False),
            ((3, 10), True),
            ((3, 11), True),
            ((3, 13), True),
            ((4, 0), True),
        ],
    )
    def test_version_gate(self, version: tuple[int, int], expected: bool) -> None:
        assert setup_env.python_version_ok(version) is expected

    def test_min_python_matches_pyproject(self) -> None:
        """门槛必须和 pyproject 的 requires-python 一致，否则脚本会放行装不上的版本。"""
        tomllib = pytest.importorskip("tomllib", reason="需要 Python 3.11+ 解析 toml")
        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        requires = data["project"]["requires-python"]
        # 形如 ">=3.10"
        major, minor = requires.lstrip(">=").split(".")
        assert setup_env.MIN_PYTHON == (int(major), int(minor))


# --------------------------------------------------------------------------- #
# 2. extras 解析与 pip 目标
# --------------------------------------------------------------------------- #

class TestExtras:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("dev,rag,api", ("dev", "rag", "api")),
            ("api", ("api",)),
            ("", ()),
            ("   ", ()),
            (" api , rag ", ("api", "rag")),
            ("api,,rag", ("api", "rag")),
        ],
    )
    def test_parse_extras(self, text: str, expected: tuple[str, ...]) -> None:
        assert setup_env.parse_extras(text) == expected

    @pytest.mark.parametrize(
        ("extras", "expected"),
        [
            ((), "."),
            (("api",), ".[api]"),
            (("dev", "rag", "api"), ".[dev,rag,api]"),
        ],
    )
    def test_pip_spec(self, extras: tuple[str, ...], expected: str) -> None:
        assert setup_env.pip_spec(extras) == expected

    def test_every_named_extra_exists_in_pyproject(self) -> None:
        """脚本里写死的 extras 名字必须在 pyproject 里真的存在。

        少了这个断言，改名 extras 只会在用户机器上炸 ``pip install -e .[x]``，
        而 CI 全绿。
        """
        tomllib = pytest.importorskip("tomllib", reason="需要 Python 3.11+ 解析 toml")
        data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        declared = set(data["project"]["optional-dependencies"])
        named = set(setup_env.DEFAULT_EXTRAS) | set(setup_env.EXTENDED_EXTRAS)
        assert named <= declared, f"pyproject 里没有这些 extras：{sorted(named - declared)}"


# --------------------------------------------------------------------------- #
# 3. 镜像
# --------------------------------------------------------------------------- #

class TestMirror:
    def test_known_mirrors_return_urls(self) -> None:
        for name in ("tsinghua", "aliyun", "ustc"):
            url = setup_env.mirror_url(name)
            assert url is not None
            assert url.startswith("https://")
            assert url.endswith("/simple")

    def test_official_is_empty_string(self) -> None:
        """official 用空串表示「不加 -i」，不能用 None 混在一起。"""
        assert setup_env.mirror_url("official") == ""

    def test_unknown_mirror_returns_none(self) -> None:
        assert setup_env.mirror_url("no-such-mirror") is None

    def test_every_mirror_is_https_or_default(self) -> None:
        for name, url in setup_env.MIRRORS.items():
            assert url == "" or url.startswith("https://"), name


# --------------------------------------------------------------------------- #
# 4. pip 命令行
# --------------------------------------------------------------------------- #

class TestPipCommand:
    def test_editable_install_with_extras(self) -> None:
        cmd = setup_env.pip_install_command(Path("py"), ("dev", "rag", "api"), None)
        assert cmd[:4] == ["py", "-m", "pip", "install"]
        assert "-e" in cmd
        assert cmd[cmd.index("-e") + 1] == ".[dev,rag,api]"
        assert "-i" not in cmd

    def test_mirror_is_appended_as_index_url(self) -> None:
        url = setup_env.mirror_url("tsinghua")
        cmd = setup_env.pip_install_command(Path("py"), ("api",), url)
        assert cmd[-2:] == ["-i", url]

    def test_uses_a_list_not_a_shell_string(self) -> None:
        """必须用列表传参：路径里有中文和方括号，拼成字符串会被 shell 拆坏。"""
        cmd = setup_env.pip_install_command(Path("py"), ("dev",), None)
        assert isinstance(cmd, list)
        assert all(isinstance(part, str) for part in cmd)


# --------------------------------------------------------------------------- #
# 5. Windows 商店版 python
# --------------------------------------------------------------------------- #

class TestWindowsStorePython:
    def test_store_stub_is_detected(self) -> None:
        stub = Path("C:/Users/me/AppData/Local/Microsoft/WindowsApps/python.exe")
        assert setup_env.is_windows_store_python(stub) is True

    def test_normal_install_is_not_detected(self) -> None:
        assert setup_env.is_windows_store_python(Path("C:/Python313/python.exe")) is False

    def test_case_insensitive(self) -> None:
        assert setup_env.is_windows_store_python(Path("C:/x/WINDOWSAPPS/python.exe")) is True


# --------------------------------------------------------------------------- #
# 6. 目录约定
# --------------------------------------------------------------------------- #

class TestLayout:
    def test_venv_lives_inside_the_project(self) -> None:
        assert setup_env.VENV_DIR == PROJECT_ROOT / ".venv"
        assert setup_env.VENV_PY.parent.parent == setup_env.VENV_DIR

    def test_venv_python_name_matches_platform(self) -> None:
        if os.name == "nt":
            assert setup_env.VENV_PY.name == "python.exe"
            assert setup_env.VENV_PY.parent.name == "Scripts"
        else:
            assert setup_env.VENV_PY.name == "python"
            assert setup_env.VENV_PY.parent.name == "bin"

    def test_kb_manifest_points_at_the_build_output(self) -> None:
        assert setup_env.KB_MANIFEST == PROJECT_ROOT / "knowledge" / "chroma" / "fisheries_manifest.json"


# --------------------------------------------------------------------------- #
# 7. 命令行参数
# --------------------------------------------------------------------------- #

class TestArgParser:
    def test_defaults_install_the_documented_extras(self) -> None:
        args = setup_env.build_parser().parse_args([])
        assert args.extras == ",".join(setup_env.DEFAULT_EXTRAS)
        assert args.mirror == "official"
        assert not any((args.recreate, args.rebuild_kb, args.skip_install,
                        args.skip_kb, args.skip_verify))
        assert args.python is None

    def test_flags_are_wired(self) -> None:
        args = setup_env.build_parser().parse_args(
            ["--skip-kb", "--skip-install", "--recreate", "--mirror", "tsinghua"]
        )
        assert args.skip_kb is True
        assert args.skip_install is True
        assert args.recreate is True
        assert args.mirror == "tsinghua"

    def test_mirror_argument_is_constrained(self) -> None:
        with pytest.raises(SystemExit):
            setup_env.build_parser().parse_args(["--mirror", "no-such-mirror"])


# --------------------------------------------------------------------------- #
# 8. 两个入口文件
# --------------------------------------------------------------------------- #

class TestLaunchers:
    @pytest.mark.parametrize("path", [SETUP_BAT, FORWARD_BAT], ids=["setup.bat", "forward.bat"])
    def test_entry_files_exist(self, path: Path) -> None:
        assert path.is_file(), f"缺少入口文件：{path.name}"

    def test_setup_bat_is_pure_ascii_and_crlf(self) -> None:
        """cmd.exe 会按代码页分块解码，非 ASCII 字节会吞掉后面的行。"""
        data = SETUP_BAT.read_bytes()
        assert all(byte < 128 for byte in data), "setup.bat 不能含非 ASCII 字节"
        assert data.count(b"\r\n") == data.count(b"\n"), "setup.bat 必须全部是 CRLF"
        assert data.count(b"\r\n") > 0

    def test_setup_bat_calls_the_python_script(self) -> None:
        text = SETUP_BAT.read_text(encoding="ascii")
        assert "scripts\\setup_env.py" in text
        assert "chcp 65001" in text, "需要 UTF-8 代码页，否则中文提示是乱码"

    def test_setup_bat_finds_a_base_interpreter(self) -> None:
        """此时 .venv 还不存在，入口只能靠系统解释器启动。"""
        text = SETUP_BAT.read_text(encoding="ascii")
        assert "py -3" in text
        assert "python3" in text
        text.encode("ascii")  # 再确认一次：读成 ascii 不抛异常

    def test_forward_bat_delegates_to_setup(self) -> None:
        text = FORWARD_BAT.read_text(encoding="ascii")
        assert 'call "%~dp0setup.bat" %*' in text
