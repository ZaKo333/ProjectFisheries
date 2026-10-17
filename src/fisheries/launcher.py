"""启动器（.bat）要显示的中文提示。

## 为什么这些文字放在 Python 里，而不是写进 .bat

cmd.exe 是按**当前控制台代码页**分块解码批处理文件的。文件里只要出现非 ASCII
字节，cmd 的行/字节偏移就会与实际内容错位，于是它会把某一行中文的**尾部片段**
当成命令去执行，stderr 里冒出这类噪声：

    'xx.bat' is not recognized as an internal or external command,
    operable program or batch file.

实测判据：把整个 .bat 的每个非 ASCII 字符换成**相同字节数**的 ASCII（改后字节数
与原文件完全一致），stderr 立刻变空；去掉 ``chcp``、换成带 BOM 的 UTF-8、把 ``chcp``
提到第一行都不管用。所以这是**内容**导致的，不是长度、也不是代码页本身。

结论：``.bat`` 必须保持**纯 ASCII**（``tests/unit/test_launchers.py`` 守这条），
用户可见的中文一律由本模块输出。Python 的输出编码由 ``fisheries.console`` 负责，
与 .bat 无关。

## 用法（启动器内部调用，也可以手工跑）

    python -m fisheries.launcher api-banner  --host 127.0.0.1 --port 8000
    python -m fisheries.launcher api-failed   --port 8000 --rc 1
    python -m fisheries.launcher api-stopped
"""

from __future__ import annotations

import argparse
import contextlib
import socket
import sys

from . import console

__all__ = ["api_banner", "failed", "local_ipv4", "main", "stopped", "web_page_url"]

_BAR = "  " + "-" * 60

#: 前端挂载点。与 ``fisheries.api.static.WEB_MOUNT_PATH`` 保持一致；
#: 这里不 import 那一个，是为了让 ``python -m fisheries.launcher``
#: 在没装 fastapi 的环境里也能跑（.bat 的失败提示要靠它）。
_WEB_PATH = "/app"


def local_ipv4() -> list[str]:
    """本机非回环 IPv4 地址（用于告诉用户“别人该连哪个地址”）。

    原 `.bat` 靠 `ipconfig | findstr IPv4` 取，这里换成纯 Python：先查主机名解析，
    再用一个不会真正发包的 UDP connect 兜底。
    """
    found: list[str] = []

    with contextlib.suppress(OSError):
        found.extend(
            str(info[4][0])
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        )

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            found.append(str(probe.getsockname()[0]))
        finally:
            probe.close()
    except OSError:
        pass

    unique: list[str] = []
    for ip in found:
        if ip.startswith("127.") or ip in unique:
            continue
        unique.append(ip)
    return unique


def web_page_url(host: str, port: str) -> str:
    """网页界面地址。监听 0.0.0.0 时给 127.0.0.1，否则原样用。"""
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{shown}:{port}{_WEB_PATH}/"


def api_banner(host: str, port: str) -> None:
    print()
    print("  ProjectFisheries 接口服务")
    print(_BAR)
    print(f"  演示前端   {web_page_url(host, port)}        （打开就能用）")
    print(f"  接口文档   http://127.0.0.1:{port}/docs        （可交互调试）")
    print(f"  接口定义   http://127.0.0.1:{port}/openapi.json")
    print(f"  监听地址   {host}:{port}")
    print("  停止服务   关掉这个窗口，或按 Ctrl+C")
    print()
    for ip in local_ipv4():
        print(f"  局域网地址 http://{ip}:{port}          （供别的电脑访问）")
    if host == "127.0.0.1":
        print()
        print("  注意：现在只监听本机。要让别的电脑连上，把 启动接口服务.bat 里的")
        print("        HOST 改成 0.0.0.0 再重新双击。")
    print()
    print("  正在启动……第一次要读知识库，等几秒钟。")
    print()


def stopped() -> None:
    print()
    print("  接口服务已正常停止。")


def failed(rc: str, port: str) -> None:
    print()
    print(f"  [启动失败] 退出码 {rc}")
    print()
    print("  常见原因与处理：")
    print(f"    1. 端口 {port} 被占用 —— 换个端口重试，命令行里执行：")
    print("         启动接口服务.bat 8001")
    print("    2. 缺依赖（fastapi / uvicorn / chromadb / fastembed）—— 在本窗口执行一次：")
    print('         .venv\\Scripts\\python.exe -m pip install -e ".[dev,rag,api]"')
    print("    3. 想先自检环境 —— 双击 运行自检.bat")
    print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fisheries.launcher")
    sub = parser.add_subparsers(dest="command", required=True)

    banner = sub.add_parser("api-banner")
    banner.add_argument("--port", default="8000")
    banner.add_argument("--host", default="127.0.0.1")

    stopped_cmd = sub.add_parser("api-stopped")
    stopped_cmd.add_argument("--port", default="8000")

    failed_cmd = sub.add_parser("api-failed")
    failed_cmd.add_argument("--port", default="8000")
    failed_cmd.add_argument("--rc", default="1")

    return parser


def main(argv: list[str] | None = None) -> int:
    console.install()
    args = _build_parser().parse_args(argv)
    command: str = args.command

    if command == "api-banner":
        api_banner(args.host, args.port)
    elif command == "api-stopped":
        stopped()
    else:
        failed(args.rc, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
