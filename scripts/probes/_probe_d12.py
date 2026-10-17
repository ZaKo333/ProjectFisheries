"""D12 接线自检：后端描述 / 语言判定 / 回译后端选择。

跑法：
    .\\.venv\\Scripts\\python.exe scripts\\probes\\_probe_d12.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console

_console.install()

from fisheries.agent.orchestrator import AquacultureAgent  # noqa: E402 - 先 sys.path 再导入
from fisheries.agent.translate import (  # noqa: E402 - 同上
    build_translator,
    describe_translator,
    load_translator_client,
    resolve_prompt_lang,
)


def main() -> int:
    agent = AquacultureAgent()
    print("-- describe_backends() --")
    for key, value in agent.describe_backends().items():
        print(f"  {key:<14} : {value}")

    print()
    print("-- 提示词语言判定（未显式指定时） --")
    for provider in ("mock", "aqua-7b-ollama", "aqua-7b-vllm",
                     "deepseek", "dashscope", "openai"):
        lang = resolve_prompt_lang(None, provider=provider, env={})
        print(f"  {provider:<16} -> {lang}")

    print()
    print("-- 显式覆盖优先于自动判定 --")
    for explicit in ("zh", "en"):
        for provider in ("aqua-7b-ollama", "deepseek"):
            lang = resolve_prompt_lang(explicit, provider=provider, env={})
            print(f"  explicit={explicit:<3} provider={provider:<16} -> {lang}")

    print()
    print("-- 环境变量覆盖 --")
    for value in ("en", "zh", "bogus"):
        lang = resolve_prompt_lang(None, provider="mock",
                                   env={"FISHERIES_PROMPT_LANG": value})
        print(f"  FISHERIES_PROMPT_LANG={value:<6} -> {lang}")

    print()
    client = load_translator_client(env={})
    translator = build_translator("auto", client=client, env={})
    print("-- 本机回译后端（无 API key、离线） --")
    print(f"  client     : {client}")
    print(f"  translator : {describe_translator(translator)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
