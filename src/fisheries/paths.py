"""把本机绝对路径压成稳定占位符——给「要进版本库的产物」用。

要解决的问题
------------
仓库里有两处会生成**入库的产物**：

* ``scripts/probes/_make_api_samples.py`` → ``docs/fixtures/_api_samples.json``
* ``scripts/evaluate_water_rules.py``   → ``docs/_m8_metrics.json``

它们必须满足同一条不变量：**同一份代码在任何机器上跑，产物逐字节相同**。
只要产物里混进一条本机绝对路径，这条就断了——别人 checkout
之后跑一遍，diff 里全是和自己无关的路径差异。

占位符
------
=================  ==============================
``<工程>``         仓库自身所在目录
``<数据集根目录>`` 数据集目录（见下）
=================  ==============================

数据集的两个可能位置（工程外的历史 ``数据集/``、工程内的 ``data/datasets``）
**收敛到同一个占位符**：不然「这台机器的数据放在哪一级」会决定产物内容，
同一份代码在两台机器上生成出不同的文件，占位符就白加了。

三个容易踩的坑
--------------
1. **路径可能嵌在长字符串中间。** 错误响应的 ``detail`` 是**字符串化的 JSON**，
   里面那条路径不在开头，而且每个反斜杠都翻了倍（JSON 转义）。真实形态就是
   本模块要替换的那条绝对路径，只是前面多了 ``{"image_path": "`` 这一段。
   所以这里是**按出现位置全局替换**，不是只认前缀。
2. **必须同时认单反斜杠与双反斜杠两种写法。** 上面那层字符串化 JSON 里，
   每个反斜杠都翻倍了。只处理单反斜杠会漏掉它——数据集路径当初正是从这里
   漏进 ``docs/fixtures/_api_samples.json`` 的。
3. **前缀后面要挡住同名前缀的兄弟目录。** ``...\\project`` 不该把
   ``...\\project2`` 也吃掉。判据是「后面那个字符不能是字母数字、下划线、
   点或连字符」：``\\``、``/``、``"``、空白、串尾都是合法边界，而 ``2``
   不是。

替换按 ``ROOTS`` 的顺序进行，命中即停：``data/datasets`` 在工程根之下，
**必须排在工程根前面**，否则会被工程根先吃掉、变成 ``<工程>\\data\\datasets\\...``。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from . import datasets as _datasets

__all__ = [
    "DATASET_PLACEHOLDER",
    "PROJECT_PLACEHOLDER",
    "ROOTS",
    "stabilize",
    "stabilize_tree",
]

DATASET_PLACEHOLDER = "<数据集根目录>"
PROJECT_PLACEHOLDER = "<工程>"

#: ``(根, 占位符)``，顺序即优先级，命中即停。理由见模块 docstring。
ROOTS: tuple[tuple[Path, str], ...] = (
    (_datasets.LEGACY_ROOT, DATASET_PLACEHOLDER),
    (_datasets.LOCAL_ROOT, DATASET_PLACEHOLDER),
    (_datasets.PROJECT_ROOT, PROJECT_PLACEHOLDER),
)

#: 根后面不允许出现的字符：字母数字/下划线/点/连字符。
#: ``\\w`` 在 Python 正则里对 ``str`` 是 Unicode 语义，中文也算——正合需要，
#: ``数据集备份/`` 这种兄弟目录同样会被挡住。
_NOT_A_PATH_BOUNDARY = r"(?![\w.\-])"


def _both_slash_forms(root: Path) -> tuple[str, str]:
    """一个根需要被认出的两种写法：单反斜杠与双反斜杠。"""
    text = str(root)
    return text, text.replace("\\", "\\\\")


def stabilize(value: str) -> str:
    """把字符串里出现的本机路径换成占位符前缀；没有就原样返回。

    对整条路径与「嵌着路径的长字符串」都成立（见模块 docstring 第 1 条），
    所以调用方不需要先判断字符串是什么形态。

    刻意不做路径归一化（``..``、大小写、正斜杠混用都不管）：产物的调用方给的
    都是 ``str(Path(...))`` 或框架自己序列化出来的形态，是确定的。猜得越多，
    越容易出现「两台机器结果不同」这种最难查的偏差。
    """
    if not isinstance(value, str) or not value:
        return value
    out = value
    for root, placeholder in ROOTS:
        for form in _both_slash_forms(root):
            out = re.sub(
                re.escape(form) + _NOT_A_PATH_BOUNDARY,
                placeholder.replace("\\", "\\\\"),
                out,
            )
    return out


def stabilize_tree(node: Any) -> Any:
    """递归把树里**所有字符串值**过一遍 :func:`stabilize`。

    递归而不是逐字段点名：``data/derived/metrics.json`` 的结构归它自己的
    生成器管，这里跟着它的 schema 走，等于给自己埋一个会悄悄分叉的副本
    （本项目在别的测试上已经吃过一次这个亏）。对数字、布尔、类别名都是无操作的。
    """
    if isinstance(node, dict):
        return {key: stabilize_tree(item) for key, item in node.items()}
    if isinstance(node, list):
        return [stabilize_tree(item) for item in node]
    if isinstance(node, str):
        return stabilize(node)
    return node
