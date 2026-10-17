"""守启动器：`.bat` 必须保持纯 ASCII，中文一律由 Python 输出。

## 这条不变量是怎么来的（不是拍脑袋定的规矩）

cmd.exe 按**当前控制台代码页**分块解码批处理文件。文件里只要出现非 ASCII 字节，
cmd 的行/字节偏移就会与实际内容错位，于是它会把某一行中文的**尾部片段**当成命令
去执行，stderr 里冒出：

    '动演示页.bat' is not recognized as an internal or external command,
    operable program or batch file.

排查时逐项试过、**都不是**原因：换行符（EOF/LF 混用）、UTF-8 带 BOM、把
``chcp 65001`` 提到第一行、拿掉 ``chcp``、去掉 ``title``、去掉中文 ``rem`` 块。
决定性判据是把当时那个 `启动接口服务.bat` 的每个非 ASCII 字符替换成**相同字节数**的 ASCII
（改后仍是 2036 字节，与原文件完全一致）——stderr 立刻变空。所以是**内容**导致的，
不是长度、不是代码页本身。

文件名可以是中文（那是命令行参数，走 Unicode 路径），**文件内容**不行。

真实验证方式（人工，跑一次约 1 分钟）::

    cmd /c "启动接口服务.bat 7930 > out.txt 2> err.txt"

然后检查 `err.txt` 是空文件（uvicorn 那类 INFO 日志属正常输出，本测试不管）。
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


LAUNCHER_DIR = PROJECT_ROOT / "scripts" / "bat"

#: 根目录只留三个人能看懂的中文入口；真正的逻辑全在 scripts/bat/。
ROOT_ENTRIES = {"一键配置环境.bat", "启动接口服务.bat", "运行自检.bat"}


def _all_bat_files() -> list[pathlib.Path]:
    """工程根 + scripts/bat/ 的全部 .bat（只看仓库自己带的）。

    刻意**不扫工程上一级**：仓库是自包含的，克隆到哪个目录都不该改变测试数量。
    以前扫上一级是为了兼容「仓库放在一个更大的工作目录里」的旧布局，结果克隆进
    含其他 .bat 的目录时，参数化用例会凭空多出几项，文档里的计数也跟着对不上。
    """
    return sorted(set(PROJECT_ROOT.glob("*.bat")) | set(LAUNCHER_DIR.glob("*.bat")))


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
    """中文名的入口只做转发，真正的逻辑在 ASCII 名的 run-*.bat 里。

    这样中文文件名（用户的点击目标）与纯 ASCII 内容（cmd 的要求）可以并存：
    转发靠 ``%~dp0run-api.bat`` 这种**纯 ASCII 路径**，不需要在文件里写中文。
    """
    for name in ("run-api.bat", "run-selftest.bat"):
        core = LAUNCHER_DIR / name
        assert core.is_file(), f"缺少 scripts/bat/{name}"
        text = core.read_text(encoding="ascii")
        assert "chcp 65001" in text, f"{name} 应打开 UTF-8 代码页"
        assert "launcher" in text or "selftest.py" in text, f"{name} 没把中文输出交给 Python"

    # 中文名入口 → 转发目标是 scripts\bat\ 下的 ASCII 名，用 %~dp0 定位
    forwards = {
        "\u542f\u52a8\u63a5\u53e3\u670d\u52a1.bat": "run-api.bat",
        "\u8fd0\u884c\u81ea\u68c0.bat": "run-selftest.bat",
    }
    for fname, target in forwards.items():
        path = PROJECT_ROOT / fname
        assert path.is_file(), f"缺少中文名入口 {fname}"
        text = path.read_text(encoding="ascii")
        forwarding = f'%~dp0scripts\\bat\\{target}'
        assert forwarding in text, f"{fname} 应转发到 scripts/bat/{target}"


def test_root_only_holds_the_three_human_entry_points() -> None:
    """根目录不再堆 ASCII 启动器：留下三个中文入口就够了。"""
    assert {p.name for p in PROJECT_ROOT.glob("*.bat")} == ROOT_ENTRIES


@pytest.mark.parametrize("name", ["setup.bat", "run-api.bat", "run-selftest.bat"])
def test_launchers_climb_back_to_the_project_root(name: str) -> None:
    """逻辑搬进 scripts/bat/ 之后，.bat 必须自己 cd 回工程根。

    否则 ``.venv\\Scripts\\python.exe`` 与 ``scripts\\selftest.py`` 全按
    ``scripts\\bat\\`` 解析，双击直接找不到——「下载后即可用」就断了。
    """
    text = (LAUNCHER_DIR / name).read_text(encoding="ascii")
    assert 'cd /d "%~dp0..\\.."' in text, f"{name} 没切回工程根"


@pytest.mark.parametrize("name", ["run-api.bat", "run-selftest.bat"])
def test_launchers_bootstrap_a_missing_venv(name: str) -> None:
    """下载后直接双击也要能起来：缺 .venv 时先跑 setup，而不是报错退出。"""
    text = (LAUNCHER_DIR / name).read_text(encoding="ascii")
    assert 'call "%~dp0setup.bat"' in text, f"{name} 缺少自动配置环境的兜底"


# --------------------------------------------------------------------------- #
# fisheries.launcher：中文提示本身
# --------------------------------------------------------------------------- #


def _load_launcher():
    from fisheries import launcher

    return launcher


def test_api_banner_points_at_the_web_page_first(capsys: pytest.CaptureFixture[str]) -> None:
    """前端由接口服务同源托管，横幅必须把网页界面地址放在最显眼的位置。

    用户双击 启动接口服务.bat 之后唯一需要知道的地址就是这一个；
    ``/docs`` 是给调试接口的人看的，不该排在它前面。
    """
    launcher = _load_launcher()
    launcher.api_banner("127.0.0.1", "8000")
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8000/app/" in out
    assert "停止服务" in out
    assert out.index("/app/") < out.index("/docs"), "网页界面地址应排在 /docs 前面"


def test_api_banner_never_prints_an_unreachable_page_url(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """绑 0.0.0.0 时页面上给出的地址不能是 0.0.0.0 —— 那个地址浏览器打不开。"""
    launcher = _load_launcher()
    launcher.api_banner("0.0.0.0", "8000")
    out = capsys.readouterr().out
    assert "http://127.0.0.1:8000/app/" in out
    assert "0.0.0.0:8000/app" not in out


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


def test_failed_hints_name_the_missing_pieces(capsys: pytest.CaptureFixture[str]) -> None:
    launcher = _load_launcher()

    launcher.failed("1", "8000")
    out = capsys.readouterr().out
    assert "8000" in out and "退出码 1" in out
    assert "uvicorn" in out, "失败提示要指出缺的依赖"
    assert "启动接口服务.bat 8001" in out, "换端口的操作要能被照着抄"


def test_stopped_message(capsys: pytest.CaptureFixture[str]) -> None:
    launcher = _load_launcher()
    launcher.stopped()
    assert "接口服务" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["api-banner", "--host", "0.0.0.0", "--port", "8001"],
        ["api-banner", "--port", "8000"],
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
    # 历史上这个列表写死在 .bat 里，新增脚本会漏跑；至少要覆盖已知的 7 个
    # （原为 8 个，`demo_webui.py` 随 Gradio 演示页一起删掉了）
    assert len(found) >= 7, f"演示脚本只剩 {len(found)} 个？{sorted(found)}"


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
