"""`scripts/deploy_windows.ps1` 的静态守卫。

这个脚本只在服务器上跑，本地测试跑不了它，但**它写出来的启动脚本决定了
线上服务连的是哪个大模型**——所以至少把「参数还在、环境变量还写、配置文件
还持久化」这几条钉住，避免有人重构时悄悄删掉。
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "deploy_windows.ps1"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


class TestLlmParameters:
    def test_script_accepts_provider_and_key(self) -> None:
        text = _text()
        for name in ("[string]$LlmProvider", "[string]$LlmApiKey", "[string]$LlmModel", "[string]$LlmBaseUrl"):
            assert name in text, f"部署脚本缺少参数 {name}"

    def test_script_can_turn_llm_off(self) -> None:
        assert "[switch]$NoLlm" in _text()

    def test_launcher_template_gets_the_llm_env(self) -> None:
        text = _text()
        # here-string 里靠 $llmLines 把四行 set 插进去，而不是硬编码成 mock。
        assert "$llmLines" in text
        for key in (
            "FISHERIES_LLM_PROVIDER",
            "FISHERIES_LLM_MODEL",
            "FISHERIES_LLM_BASE_URL",
            "FISHERIES_LLM_API_KEY",
        ):
            assert key in text, f"启动脚本模板里缺少 {key}"

    def test_config_survives_redeploy(self) -> None:
        text = _text()
        assert 'Join-Path $Root "llm.txt"' in text
        # provider=mock 或 -NoLlm 时必须把残留配置删掉，否则重跑部署会
        # 把「关掉大模型」又变回真模型。
        assert "Remove-Item -Path $LlmFile" in text


class TestScriptStaysPortable:
    def test_script_is_pure_ascii(self) -> None:
        # 服务器默认代码页（GBK/437）下的 PowerShell 5.1 读非 ASCII 脚本会出乱码。
        data = SCRIPT.read_bytes()
        bad = [b for b in data if b > 127]
        assert not bad, f"deploy_windows.ps1 里出现了 {len(bad)} 个非 ASCII 字节"

    def test_script_parses(self) -> None:
        # 轻量语法检查：至少括号/引号配平（真正的解析交给服务器上的 PowerShell）。
        text = _text()
        assert text.count("param(") == 1
        assert text.count('@"') == text.count('"@')
