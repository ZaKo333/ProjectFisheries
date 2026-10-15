"""守启动器：`.bat` 必须保持纯 ASCII，中文一律由 Python 输出。

## 这条不变量是怎么来的（不是拍脑袋定的规矩）

cmd.exe 按**当前控制台代码页**分块解码批处理文件。文件里只要出现非 ASCII 字节，
cmd 的行/字节偏移就会与实际内容错位，于是它会把某一行中文的**尾部片段**当成命令
去执行，stderr 里冒出：

    '动演示页.bat' is not recognized as an internal or external command,
    operable program or batch file.

排查时逐项试过、**都不是**原因：换行符（EOF/LF 混用）、UTF-8 带 BOM、把
``chcp 65001`` 提到第一行、拿掉 ``chcp``、去掉 ``title``、去掉中文 ``rem`` 块。
决定性判据是把原 `启动演示页.bat` 的每个非 ASCII 字符替换成**相同字节数**的 ASCII
（改后仍是 2036 字节，与原文件完全一致）——stderr 立刻变空。所以是**内容**导致的，
不是长度、不是代码页本身。

文件名可以是中文（那是命令行参数，走 Unicode 路径），**文件内容**不行。

真实验证方式（人工，跑一次约 1 分钟）::

    cmd /c "启动演示页.bat 7930 > out.txt 2> err.txt"

然后检查 `err.txt` 是空文件（uvicorn 那类 INFO 日志属正常输出，本测试不管）。
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO_PARENT = PROJECT_ROOT.parent


def _all_bat_files() -> list[pathlib.Path]:
    """工程内 + 工程上一级的全部 .bat（启动器分布在两层）。"""
    return sorted(set(PROJECT_ROOT.glob("*.bat")) | set(REPO_PARENT.glob("*.bat")))


def test_there_is_at_least_one_launcher() -> None:
    assert _all_bat_files(), "一个 .bat 启动器都没找到，路径推断可能错了"


@pytest.mark.parametrize("path", _all_bat_files(), ids=lambda p: p.name)
def test_bat_files_are_pure_ascii(path: pathlib.Path) -> None:
    raw = path.read_bytes()
    offenders = sorted({b for b in raw if b > 127})
    assert not offenders, (
        f"{path.name} 含非 ASCII 字节 {offenders}。cmd.exe 会因此解析错位并把中文行"
        f"尾部当命令执行。请把中文提示搬到 fisheries.launcher / scripts.selftest。"
    )


@pytest.mark.parametrize("path", _all_bat_files(), ids=lambda p: p.name)
def test_bat_files_use_crlf_line_endings(path: pathlib.Path) -> None:
    raw = path.read_bytes()
    lf = raw.count(b"\n")
    crlf = raw.count(b"\r\n")
    assert lf == crlf, (
        f"{path.name} 有 {lf - crlf} 个裸 LF。cmd 逐行解析批处理，LF-only 会把一行"
        f"拆坏（历史上曾因此生成 0 字节垃圾文件）。"
    )
    assert crlf > 0, f"{path.name} 没有 CRLF 换行，可能被写成了空文件"


def test_chinese_named_launchers_forward_to_ascii_core() -> None:
    """中文名的三个入口只做转发，真正的逻辑在 ASCII 名的 run-*.bat 里。

    这样中文文件名（用户的点击目标）与纯 ASCII 内容（cmd 的要求）可以并存：
    转发靠 ``%~dp0run-demo.bat`` 这种**纯 ASCII 路径**，不需要在文件里写中文。
    """
    for name in ("run-demo.bat", "run-api.bat", "run-selftest.bat"):
        core = PROJECT_ROOT / name
        assert core.is_file(), f"缺少 {name}"
        text = core.read_text(encoding="ascii")
        assert "chcp 65001" in text, f"{name} 应打开 UTF-8 代码页"
        assert "launcher" in text or "selftest.py" in text, f"{name} 没把中文输出交给 Python"

    # 中文名入口 → 转发目标必须是 ASCII 名，且用 %~dp0 定位同目录
    forwards = {
        "\u542f\u52a8\u6f14\u793a\u9875.bat": "run-demo.bat",
        "\u542f\u52a8\u63a5\u53e3\u670d\u52a1.bat": "run-api.bat",
        "\u8fd0\u884c\u81ea\u68c0.bat": "run-selftest.bat",
    }
    for fname, target in forwards.items():
        path = PROJECT_ROOT / fname
        assert path.is_file(), f"缺少中文名入口 {fname}"
        text = path.read_text(encoding="ascii")
        assert f'%~dp0{target}' in text, f"{fname} 应转发到 {target}"


# --------------------------------------------------------------------------- #
# fisheries.launcher：中文提示本身
# --------------------------------------------------------------------------- #


def _load_launcher():
    from fisheries import launcher

    return launcher


def test_webui_banner_shows_url_and_stop_hint(capsys: pytest.CaptureFixture[str]) -> None:
    launcher = _load_launcher()
    launcher.webui_banner("7860")
    out = capsys.readouterr().out
    assert "http://127.0.0.1:7860" in out
    assert "停止服务" in out
    # 必须告诉用户页面上该点哪两个按钮，否则第一次打开不知道从哪开始
    assert "填入示例数据" in out and "运行 Agent" in out


def test_api_banner_shows_docs_and_listen_address(capsys: pytest.CaptureFixture[str]) -> None:
    launcher = _load_launcher()
    launcher.api_banner("127.0.0.1", "8000")
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8000/docs" in out
    assert "http://127.0.0.1:8000/openapi.json" in out
    assert "127.0.0.1:8000" in out


def test_api_banner_only_warns_when_loopback_only(capsys: pytest.CaptureFixture[str]) -> None:
    """只监听回环时必须显式提醒；绑 0.0.0.0 时不该再吓唬用户。"""
    launcher = _load_launcher()

    launcher.api_banner("127.0.0.1", "8000")
    assert "现在只监听本机" in capsys.readouterr().out

    launcher.api_banner("0.0.0.0", "8000")
    out = capsys.readouterr().out
    assert "现在只监听本机" not in out
    # 这行是回显监听地址，绑了就照实写
    assert "监听地址   0.0.0.0:8000" in out


def test_local_ipv4_skips_loopback() -> None:
    from fisheries import launcher

    for ip in launcher.local_ipv4():
        assert not ip.startswith("127."), "回环地址不该出现在“局域网地址”里"


def test_failed_hints_differ_per_launcher(capsys: pytest.CaptureFixture[str]) -> None:
    launcher = _load_launcher()

    launcher.failed("webui", "1", "7860")
    webui = capsys.readouterr().out
    assert "7860" in webui and "退出码 1" in webui and "gradio" in webui

    launcher.failed("api", "1", "8000")
    api = capsys.readouterr().out
    assert "8000" in api and "uvicorn" in api


def test_stopped_message_per_launcher(capsys: pytest.CaptureFixture[str]) -> None:
    launcher = _load_launcher()
    launcher.stopped("webui")
    assert "演示页" in capsys.readouterr().out
    launcher.stopped("api")
    assert "接口服务" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["webui-banner", "--port", "7861"],
        ["webui-stopped"],
        ["webui-failed", "--port", "7861", "--rc", "3"],
        ["api-banner", "--host", "0.0.0.0", "--port", "8001"],
        ["api-stopped"],
        ["api-failed", "--port", "8001", "--rc", "3"],
    ],
)
def test_main_smoke(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    launcher = _load_launcher()
    assert launcher.main(argv) == 0
    assert capsys.readouterr().out.strip(), f"{argv[0]} 没有任何输出"


def test_main_requires_a_subcommand() -> None:
    launcher = _load_launcher()
    with pytest.raises(SystemExit):
        launcher.main([])


# --------------------------------------------------------------------------- #
# scripts/selftest.py：自检逻辑（不真的跑 pytest / 演示脚本）
# --------------------------------------------------------------------------- #


def _load_selftest():
    path = PROJECT_ROOT / "scripts" / "selftest.py"
    spec = importlib.util.spec_from_file_location("_selftest_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # 先登记再 exec：模块内如果有 dataclass / 相对导入，缺这步会炸
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_selftest_discovers_every_demo_script() -> None:
    selftest = _load_selftest()
    found = {p.name for p in selftest.demo_scripts()}
    expected = {p.name for p in (PROJECT_ROOT / "scripts").glob("demo_*.py")}
    assert found == expected
    # 历史上这个列表写死在 .bat 里，新增脚本会漏跑；至少要覆盖已知的 8 个
    assert len(found) >= 8, f"演示脚本只剩 {len(found)} 个？{sorted(found)}"


def test_selftest_passes_when_every_step_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    selftest = _load_selftest()
    monkeypatch.setattr(selftest, "step_python_env", lambda: True)
    monkeypatch.setattr(selftest, "step_optional_deps", lambda: None)
    monkeypatch.setattr(selftest, "step_knowledge_base", lambda: True)
    monkeypatch.setattr(selftest, "step_unit_tests", lambda: True)
    monkeypatch.setattr(selftest, "step_demos", lambda: True)
    assert selftest.main() == 0


@pytest.mark.parametrize(
    "failing",
    ["step_python_env", "step_knowledge_base", "step_unit_tests", "step_demos"],
)
def test_selftest_fails_when_a_step_fails(
    failing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    selftest = _load_selftest()
    monkeypatch.setattr(selftest, "step_python_env", lambda: True)
    monkeypatch.setattr(selftest, "step_optional_deps", lambda: None)
    monkeypatch.setattr(selftest, "step_knowledge_base", lambda: True)
    monkeypatch.setattr(selftest, "step_unit_tests", lambda: True)
    monkeypatch.setattr(selftest, "step_demos", lambda: True)
    monkeypatch.setattr(selftest, failing, lambda: False)
    assert selftest.main() == 1
