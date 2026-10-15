"""``python -m fisheries.api`` 的入口，等价于 ``python -m fisheries.api.app``。

存在的意义只有一个：让最顺手的那个写法也能用。之前 ``fisheries.api``
是个没有 ``__main__`` 的包，直接 ``python -m fisheries.api`` 会报
``No module named fisheries.api.__main__``——启动脚本和文档里都踩过这个坑。
"""

from __future__ import annotations

from .app import main

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
