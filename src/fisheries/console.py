"""控制台输出的编码兜底（Windows / 中文环境）。

背景
----
本工程的脚本会打印 ``⚠`` ``✅`` ``❌`` ``✔`` ``▶`` 这类装饰符号。它们
**不在 GBK（cp936）字符集里**，而中文 Windows 上 ``sys.stdout.encoding``
默认就是 ``gbk``。于是这一行::

    print("  ⚠ 找不到测试图")

会抛 ``UnicodeEncodeError: 'gbk' codec can't encode character '\\u26a0'``，
**不是显示成乱码，而是整个脚本当场中断**。

为什么双击时看不到这个问题
--------------------------
``启动接口服务.bat`` / ``运行自检.bat`` 首行都有
``chcp 65001``，把控制台切成了 UTF-8，所以双击启动时一切正常。
但下面两种跑法会踩到：

1. 按 README / 使用说明 直接敲 ``python scripts\\fetch_datasets.py --status``；
2. 把输出重定向到文件（``... > log.txt``）——此时 Python 用的是 locale
   编码而不是控制台编码，``chcp 65001`` 也救不了。

``install()`` 做的事
--------------------
把 stdout / stderr 的 ``errors`` 从 ``strict`` 改成 ``replace``，并**锁住
初次见到的编码**。编码本身不动，所以中文在任何终端里都仍然正常显示；只是
万一还有漏网的装饰符号，最坏变成一个 ``?``，而不是把脚本打断。

为什么需要"锁编码"：``import ultralytics`` 会自己把 ``sys.stdout`` 改成
UTF-8
---------------------------------------------------------------------
实测（``PYTHONIOENCODING=gbk`` 下逐库导入）：

.. code-block:: text

    import numpy:        'gbk' -> 'gbk'
    import cv2:          'gbk' -> 'gbk'
    import torch:        'gbk' -> 'gbk'
    import ultralytics:  'gbk' -> 'utf-8'   <== 只有它动了

后果很隐蔽：脚本前半段按 GBK 写、后半段按 UTF-8 写，**同一个输出流里混着
两种编码**，终端上前半段正常、加载模型之后整段变乱码，而且不报任何错。
所以 ``install()`` 是**可重复调用**的：它记住第一次见到的编码，之后每次
调用都把被库改掉的编码扳回去。凡是 ``import ultralytics`` 之后都应再调一次
（见 ``fisheries.tools.fish_disease`` 与 ``scripts/train_yolo_fish.py``）。

为什么不干脆把 stdout 改成 UTF-8
--------------------------------
把 stdout 单方面改成 UTF-8 之后，字节会以 UTF-8 写进一个仍按 GBK 解读的
控制台，**中文直接变乱码**。让控制台变成 UTF-8 是 ``chcp 65001`` 的职责，
应该由启动器做，不能由 Python 替它决定。

``mark()`` 用于状态标记
-----------------------
"成功 / 缺失" 这种**承载信息**的符号不能退化成 ``?``，否则用户分不清
"没成功" 和 "成功了但符号画不出来"。``mark(True)`` 在目标编码支持时返回
``✔``，否则返回 ``[OK]``——信息不丢，形态随环境变。
"""

from __future__ import annotations

import sys
from typing import Any

#: 编码支持时使用的标记
MARK_OK = "\u2714"  # ✔
MARK_BAD = "\u2718"  # ✘
#: "需要注意但不算失败"的标记（例如"待核对""已过期"）
MARK_WARN = "\u26a0"  # ⚠
#: 编码不支持时的降级形态。刻意保持等宽，便于对齐。
#: ``[OK]`` / ``[--]`` / ``[!!]`` 都是 4 个字符——``_print_doc_table`` 这类
#: 表格渲染依赖这一点。
MARK_OK_ASCII = "[OK]"
MARK_BAD_ASCII = "[--]"
MARK_WARN_ASCII = "[!!]"

__all__ = [
    "MARK_BAD",
    "MARK_BAD_ASCII",
    "MARK_OK",
    "MARK_OK_ASCII",
    "MARK_WARN",
    "MARK_WARN_ASCII",
    "install",
    "mark",
    "pinned_encoding",
    "supports",
    "warn",
]

#: 每个流第一次被 ``install()`` 看到时的编码名，键是 ``id(stream)``。
#: 存在的唯一理由：``import ultralytics`` 会把 stdout 改成 UTF-8，而字节
#: 仍要按控制台真实的编码写出去。
_PINNED: dict[int, str] = {}


def _encoding_of(stream: Any) -> str:
    """取流的编码名；取不到就假定 UTF-8（那种流通常什么都能写）。"""
    encoding = getattr(stream, "encoding", None)
    return encoding or "utf-8"


def _pin_key(stream: Any) -> int | None:
    """给真实流一个稳定的标识（文件描述符），拿不到就不锁。

    用 ``fileno()`` 而不是 ``id(stream)``：测试里造的一次性假流会被回收，
    ``id`` 可能被复用，导致编码锁串台。stdout / stderr 是 1 / 2。
    """
    try:
        return int(stream.fileno())
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def pinned_encoding(stream: Any = None) -> str | None:
    """该流被 ``install()`` 锁住的编码；还没锁过返回 ``None``。"""
    target = sys.stdout if stream is None else stream
    key = _pin_key(target)
    return None if key is None else _PINNED.get(key)


def supports(text: str, stream: Any = None) -> bool:
    """``text`` 能否用该流当前的编码写出去。

    只看**编码**，不看 ``errors``——``install()`` 把 ``errors`` 改成 replace
    之后 ``✔`` 仍然会被 ``mark()`` 判定为"编不出"，这正是我们要的：
    降级要基于编码能力，而不是基于"会不会抛异常"。
    """
    target = sys.stdout if stream is None else stream
    try:
        text.encode(_encoding_of(target))
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def install() -> bool:
    """让 stdout / stderr 遇到编不出的字符时降级为 ``?``，而不是抛异常。

    **可以重复调用**：第一次调用会记住每个流当时的编码，之后每次调用都把
    被第三方库改掉的编码扳回去（``import ultralytics`` 会把 stdout 改成
    UTF-8，见模块 docstring）。返回是否真的改动过什么。
    """
    changed = False
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            # 被重定向到了没有 reconfigure 的对象，只能接受现状
            continue

        current = getattr(stream, "encoding", None)
        key = _pin_key(stream)
        # 没有 fileno() 的流（测试替身、StringIO）不做编码锁，
        # 只把 errors 放宽——它们不会被第三方库偷偷改编码。
        pinned = current if key is None else _PINNED.setdefault(key, current or "utf-8")

        kwargs: dict[str, Any] = {}
        if current and pinned and current.lower() != pinned.lower():
            # 库把编码改掉了，扳回控制台真实的那个
            kwargs["encoding"] = pinned
        if getattr(stream, "errors", None) != "replace":
            kwargs["errors"] = "replace"
        if not kwargs:
            continue
        try:
            reconfigure(**kwargs)
        except (ValueError, OSError, LookupError):
            # 关闭的流 / 不支持的包装器 / 编不出的编码名
            continue
        changed = True
    return changed


def mark(ok: bool) -> str:
    """状态标记：编码支持就用 ``✔`` / ``✘``，不支持就退回 ``[OK]`` / ``[--]``。"""
    if supports(MARK_OK + MARK_BAD):
        return MARK_OK if ok else MARK_BAD
    return MARK_OK_ASCII if ok else MARK_BAD_ASCII


def warn() -> str:
    """告警标记：编码支持就用 ``⚠``，不支持就退回 ``[!]``。

    ``⚠`` 承载"待核对 / 已过期"这类**需要注意但不算失败**的信息，
    不能和 ``✘`` 混为一谈，所以单独给一个记号而不是复用 ``mark(False)``。
    """
    return MARK_WARN if supports(MARK_WARN) else MARK_WARN_ASCII
