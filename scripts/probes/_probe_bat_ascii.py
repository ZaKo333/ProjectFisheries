"""列出所有 .bat 启动器里的非 ASCII 字节。

背景：cmd.exe 读取批处理文件时会按当前控制台代码页分块解码；文件里只要出现
非 ASCII 字符，字节偏移就会与实际行错位，cmd 会把某一行中文的**尾部片段**当成
命令去执行，于是 stderr 里冒出

    '动演示页.bat' is not recognized as an internal or external command

这类噪声（功能不受影响，但用户看得到，且是真实的解析缺陷）。
纯 ASCII 的 .bat 在任何代码页下解码结果都一样，不会错位。

用法：python scripts/probes/_probe_bat_ascii.py
"""

from __future__ import annotations

import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PARENT = PROJECT_ROOT.parent


def scan() -> int:
    bad = 0
    for p in sorted(set(PROJECT_ROOT.glob("*.bat")) | set(PARENT.glob("*.bat"))):
        raw = p.read_bytes()
        non_ascii = [b for b in raw if b > 127]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", "replace")
        chars = sorted({c for c in text if ord(c) > 127})
        status = "OK" if not non_ascii else "NON-ASCII"
        print(
            f"[{status}] {p}\n"
            f"         bytes={len(raw)} non_ascii_bytes={len(non_ascii)} "
            f"distinct_chars={len(chars)}"
        )
        if chars:
            print("         " + repr("".join(chars))[:160])
            bad += 1
    print()
    if bad:
        print(f"结论：{bad} 个 .bat 含非 ASCII 字符，会在 cmd 下产生解析错位噪声。")
    else:
        print("结论：全部 .bat 都是纯 ASCII，cmd 下不会解析错位。")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(scan())
