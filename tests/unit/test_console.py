"""控制台编码兜底（``fisheries.console``）的单元测试。

这一层守的是一个**在中文 Windows 上必然踩到、但双击启动时看不出来**的坑：

本工程的脚本会打印 ``⚠`` ``✅`` ``❌`` ``✔`` ``▶``，它们都不在 GBK 里，
而 ``sys.stdout.encoding`` 在中文 Windows 上默认就是 ``gbk``。
``print("⚠ ...")`` 于是抛 ``UnicodeEncodeError``，**不是显示成乱码，
而是整个脚本当场中断**。

启动器里那句 ``chcp 65001`` 能遮住这个问题，所以
「双击一直好好的」和「按 README 直接跑就崩」可以同时成立。
测试因此刻意走 ``PYTHONIOENCODING=gbk`` 的子进程，而不是在当前进程里
断言——当前进程的 stdout 编码是被 pytest 换过的，测不出真实现场。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fisheries import console

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

#: 本工程用过、但 GBK(cp936) 编不出来的装饰符号。
#: 这个集合是**实测**得出的，不是猜的——增删前先跑一次
#: ``"…".encode("gbk")`` 确认。
NON_GBK_SYMBOLS = frozenset("\u2714\u2718\u2705\u274c\u26a0\u25b6\u2696\u2713\u2717\u2b50")

#: 反例：``─``(U+2500) 是制表符，GBK **能**编出来，不该被当成问题符号。
GBK_SAFE_SYMBOLS = frozenset("\u2500")


class FakeStream:
    """只有一个 ``encoding`` 属性的假流，用来模拟不同控制台。"""

    def __init__(self, encoding: str) -> None:
        self.encoding = encoding


class ReconfigurableStream:
    """支持 ``reconfigure`` 的假流，用来测 ``install()``。"""

    def __init__(self, encoding: str, errors: str = "strict") -> None:
        self.encoding = encoding
        self.errors = errors
        self.reconfigured: list[dict[str, object]] = []

    def reconfigure(self, **kwargs: object) -> None:
        self.reconfigured.append(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class BrokenStream:
    """``reconfigure`` 直接抛异常的流（关闭的流 / 不支持的包装器）。"""

    encoding = "gbk"
    errors = "strict"

    def reconfigure(self, **kwargs: object) -> None:
        raise ValueError("I/O operation on closed file")


class BareStream:
    """连 ``reconfigure`` 都没有的流（例如被重定向到自定义对象）。"""

    encoding = "gbk"


# --------------------------------------------------------------------------- #
# 前提：把"哪些符号在 GBK 下是危险的"钉死
# --------------------------------------------------------------------------- #


class TestSymbolPremises:
    def test_non_gbk_symbols_really_cannot_be_encoded(self) -> None:
        """先证明这批符号在 GBK 下确实编不出来——否则后面的测试就是空的。"""
        for ch in sorted(NON_GBK_SYMBOLS):
            with pytest.raises(UnicodeEncodeError):
                ch.encode("gbk")

    def test_gbk_safe_symbol_is_not_in_the_danger_set(self) -> None:
        """``─`` 能编码，不该被误判为问题符号（否则会误报一堆脚本）。"""
        for ch in sorted(GBK_SAFE_SYMBOLS):
            ch.encode("gbk")  # 不抛就算过
            assert ch not in NON_GBK_SYMBOLS


# --------------------------------------------------------------------------- #
# supports()
# --------------------------------------------------------------------------- #


class TestSupports:
    def test_true_for_utf8(self) -> None:
        assert console.supports(console.MARK_OK, FakeStream("utf-8")) is True

    def test_false_for_gbk(self) -> None:
        assert console.supports(console.MARK_WARN, FakeStream("gbk")) is False

    def test_true_for_ascii_under_gbk(self) -> None:
        """降级形态本身必须是任何编码都写得出的，否则降级没意义。"""
        for mark in (console.MARK_OK_ASCII, console.MARK_BAD_ASCII, console.MARK_WARN_ASCII):
            assert console.supports(mark, FakeStream("gbk")) is True

    def test_missing_encoding_attribute_defaults_to_true(self) -> None:
        """取不到 encoding 的流（多数是内存缓冲）按 UTF-8 处理，不误报。"""

        class NoEncoding:
            pass

        assert console.supports(console.MARK_OK, NoEncoding()) is True

    def test_unknown_encoding_name_is_not_an_exception(self) -> None:
        """编码名不认识时返回 False，不要把 LookupError 漏出去。"""
        assert console.supports(console.MARK_OK, FakeStream("no-such-codec")) is False


# --------------------------------------------------------------------------- #
# mark() / warn()
# --------------------------------------------------------------------------- #


class TestMark:
    def test_uses_glyphs_when_they_fit(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "stdout", FakeStream("utf-8"))
        assert console.mark(True) == console.MARK_OK
        assert console.mark(False) == console.MARK_BAD
        assert console.warn() == console.MARK_WARN

    def test_degrades_to_ascii_on_gbk(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "stdout", FakeStream("gbk"))
        assert console.mark(True) == console.MARK_OK_ASCII
        assert console.mark(False) == console.MARK_BAD_ASCII
        assert console.warn() == console.MARK_WARN_ASCII

    def test_ok_and_bad_stay_distinguishable_when_degraded(self, monkeypatch) -> None:
        """降级的**全部意义**就在这里：成功和缺失不能都变成 `?`。"""
        monkeypatch.setattr(sys, "stdout", FakeStream("gbk"))
        assert console.mark(True) != console.mark(False)

    def test_warn_is_not_confused_with_bad(self, monkeypatch) -> None:
        """``⚠`` 表示"待核对/已过期"，跟 ``✘``（失败）不是一回事。"""
        monkeypatch.setattr(sys, "stdout", FakeStream("gbk"))
        assert console.warn() != console.mark(False)

    def test_degraded_forms_are_equal_width(self) -> None:
        """等宽才能在表格里对齐——``_print_doc_table`` 依赖这一点。"""
        assert len(console.MARK_OK_ASCII) == len(console.MARK_BAD_ASCII) == len(console.MARK_WARN_ASCII)


# --------------------------------------------------------------------------- #
# install()
# --------------------------------------------------------------------------- #


class TestInstall:
    def test_switches_errors_to_replace_without_touching_encoding(self, monkeypatch) -> None:
        """关键设计：只改 ``errors``，**不动 encoding**。

        把 encoding 单方面改成 UTF-8 会让字节以 UTF-8 写进一个仍按 GBK
        解读的控制台，中文直接变乱码——那正是被这份实现替换掉的旧写法。
        """
        out, err = ReconfigurableStream("gbk"), ReconfigurableStream("gbk")
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)

        assert console.install() is True
        assert out.errors == "replace" and err.errors == "replace"
        assert out.encoding == "gbk" and err.encoding == "gbk"
        assert out.reconfigured == [{"errors": "replace"}]

    def test_tolerates_streams_without_reconfigure(self, monkeypatch) -> None:
        out, err = BareStream(), BareStream()
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        assert console.install() is False  # 什么都没改，但也不该抛

    def test_tolerates_reconfigure_raising(self, monkeypatch) -> None:
        monkeypatch.setattr(sys, "stdout", BrokenStream())
        monkeypatch.setattr(sys, "stderr", BrokenStream())
        assert console.install() is False

    def test_is_idempotent(self, monkeypatch) -> None:
        out, err = ReconfigurableStream("gbk"), ReconfigurableStream("gbk")
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)
        assert console.install() is True
        assert console.install() is False  # 第二次已经是 replace，不再动
        assert len(out.reconfigured) == 1


# --------------------------------------------------------------------------- #
# 编码锁：import ultralytics 会把 stdout 改成 UTF-8
# --------------------------------------------------------------------------- #


class FilenoStream(ReconfigurableStream):
    """带 ``fileno()`` 的假流——只有真流（或带描述符的替身）才会被锁编码。

    用显式的 fd 数字而不是 ``id()``：``_PINNED`` 是进程级全局，
    各测试必须用互不相同的 fd，否则编码锁会串台。
    """

    def __init__(self, encoding: str, fd: int, errors: str = "strict") -> None:
        super().__init__(encoding, errors)
        self._fd = fd

    def fileno(self) -> int:
        return self._fd


class TestEncodingPin:
    """`import ultralytics` 实测会把 ``sys.stdout`` 从 gbk 改成 utf-8。

    后果很隐蔽：同一个流里前半段 GBK、后半段 UTF-8，终端上后半段整体变乱码
    且不报错。所以 ``install()`` 记住初次见到的编码，可重复调用来扳回来。
    """

    def test_library_flipping_encoding_is_reverted(self, monkeypatch) -> None:
        out = FilenoStream("gbk", fd=901)
        err = FilenoStream("gbk", fd=902)
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", err)

        console.install()
        assert console.pinned_encoding(out) == "gbk"

        # 模拟 `import ultralytics`：库自己把编码改成 UTF-8
        out.reconfigure(encoding="utf-8")
        err.reconfigure(encoding="utf-8")
        assert out.encoding == "utf-8"

        # 再调一次 install()，编码必须被扳回控制台真实编码
        assert console.install() is True
        assert out.encoding == "gbk" and err.encoding == "gbk"
        assert out.errors == "replace"

    def test_pin_is_not_overwritten_by_the_flipped_encoding(self, monkeypatch) -> None:
        """扳回来之后再次调用，不应该把 UTF-8 当作"真实编码"记下来。"""
        out = FilenoStream("gbk", fd=903)
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", ReconfigurableStream("gbk"))

        console.install()
        out.reconfigure(encoding="utf-8")
        console.install()
        out.reconfigure(encoding="utf-8")
        console.install()

        assert out.encoding == "gbk"
        assert console.pinned_encoding(out) == "gbk"

    def test_streams_without_fileno_are_not_pinned(self, monkeypatch) -> None:
        """拿不到 fd 的流（测试替身 / StringIO）不做编码锁，只放宽 errors。"""
        out = ReconfigurableStream("gbk")
        monkeypatch.setattr(sys, "stdout", out)
        monkeypatch.setattr(sys, "stderr", ReconfigurableStream("gbk"))

        console.install()
        assert console.pinned_encoding(out) is None
        assert out.errors == "replace"

    def test_pinned_encoding_is_none_before_install(self) -> None:
        assert console.pinned_encoding(FilenoStream("gbk", fd=904)) is None


# --------------------------------------------------------------------------- #
# 回归守卫：新脚本不许再绕过 console
# --------------------------------------------------------------------------- #


def _scripts_printing_non_gbk_symbols() -> list[Path]:
    """找出所有"会在 stdout 上打印非 GBK 符号"的脚本。"""
    hits: list[Path] = []
    for path in sorted(SCRIPTS_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "print(" in line and (set(line) & NON_GBK_SYMBOLS):
                hits.append(path)
                break
    return hits


class TestScriptsInstallConsoleSafety:
    def test_there_is_at_least_one_such_script(self) -> None:
        """守卫自身的前提：这个集合不该是空的（否则守卫形同虚设）。"""
        assert _scripts_printing_non_gbk_symbols(), "没找到任何打印非 GBK 符号的脚本，守卫失去意义"

    def test_every_such_script_installs_console_safety(self) -> None:
        offenders = [
            str(p.relative_to(PROJECT_ROOT))
            for p in _scripts_printing_non_gbk_symbols()
            if "_console.install()" not in p.read_text(encoding="utf-8")
        ]
        assert not offenders, (
            "这些脚本会打印 GBK 编不出的符号，却没有调用 console.install()，"
            f"在中文 Windows 上直接跑会 UnicodeEncodeError：{offenders}"
        )

    def test_no_script_reconfigures_stdout_to_utf8(self) -> None:
        """禁止再出现"把 stdout 改成 UTF-8"的旧写法——它会把中文变成乱码。

        ``chcp 65001`` 是启动器的职责；Python 单方面改 encoding 只会让
        字节和终端对不上。
        """
        pattern = re.compile(r"reconfigure\(\s*encoding\s*=")
        offenders = [
            str(p.relative_to(PROJECT_ROOT))
            for p in SCRIPTS_DIR.rglob("*.py")
            if pattern.search(p.read_text(encoding="utf-8"))
        ]
        assert not offenders, f"这些脚本仍在自行把 stdout 改成 UTF-8：{offenders}"


# --------------------------------------------------------------------------- #
# 端到端：真正复现"按 README 直接跑"的现场
# --------------------------------------------------------------------------- #


def _run_under_gbk(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    """在 ``PYTHONIOENCODING=gbk`` 下跑脚本，模拟中文 Windows 的默认控制台。

    刻意用 ``PYTHONIOENCODING`` 而不是伪造 ``sys.stdout``：报告出来的缺陷
    就是"在真实 GBK 终端里跑"，只有子进程能复现它。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "gbk"
    env.pop("PYTHONUTF8", None)
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="gbk",
        errors="replace",
        timeout=180,
        check=False,
    )


class TestRealGbkConsole:
    def test_fetch_datasets_status_does_not_crash(self) -> None:
        """这条命令写在 README / 使用说明里，用户会直接照着敲。

        修复前：``UnicodeEncodeError: 'gbk' codec can't encode character
        '\\u2714'``，退出码 1。修复后：标记降级成 ``[OK]``/``[--]``，
        "可用/缺失" 这个信息仍然读得出来。
        """
        result = _run_under_gbk("fetch_datasets.py", "--status")
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
        assert "UnicodeEncodeError" not in result.stderr
        assert "UnicodeEncodeError" not in result.stdout
        # 降级形态必须出现，且成功/缺失两种标记都要在
        assert console.MARK_OK_ASCII in result.stdout
        assert console.MARK_BAD_ASCII in result.stdout

    def test_fetch_datasets_list_does_not_crash(self) -> None:
        result = _run_under_gbk("fetch_datasets.py", "--list")
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
        assert "UnicodeEncodeError" not in result.stderr

    def test_fetch_datasets_where_does_not_crash(self) -> None:
        result = _run_under_gbk("fetch_datasets.py", "--where", "fish_disease")
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
        assert "UnicodeEncodeError" not in result.stderr

    def test_build_knowledge_base_check_does_not_crash(self) -> None:
        """``scripts/selftest.py`` 的第 3 步就是这条命令（`运行自检.bat` 现在只转发）。

        向量库不入库（见 ``.gitignore``），所以全新克隆上这条命令会以退出码 1
        报「未构建」——那是正常状态，不是崩溃。本用例只管控制台编码：无论库
        建没建，都不许抛 ``UnicodeEncodeError``，也不许甩栈。
        """
        result = _run_under_gbk("build_knowledge_base.py", "--check")
        assert "UnicodeEncodeError" not in result.stderr
        assert "Traceback" not in result.stderr
        if result.returncode == 0:
            assert "语料状态" in result.stdout
        else:
            # 未构建时必须明确告诉用户下一步跑什么，而不是只给一个非零退出码
            out = result.stdout + result.stderr
            assert "未构建" in out
            assert "build_knowledge_base.py" in out
