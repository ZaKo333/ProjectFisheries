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


class TestEmbedderSwitching:
    """换向量化后端是线上最容易踩的坑：库里的后端与环境变量不一致时服务起不来。"""

    def test_script_accepts_embedder_and_rebuild(self) -> None:
        text = _text()
        assert "[string]$Embedder" in text
        assert "[switch]$RebuildKb" in text

    def test_reads_backend_from_the_store_manifest(self) -> None:
        text = _text()
        # 必须读 manifest 里的 embedder 来判断要不要重建，而不是只看目录在不在。
        assert "fisheries_manifest.json" in text
        assert "ConvertFrom-Json" in text
        assert ".StartsWith(" in text

    def test_rebuild_stops_service_and_drops_the_store(self) -> None:
        text = _text()
        assert "/End /TN" in text, "重建前要先停服务，否则文件被占用"
        assert "Remove-Item -Path $kbDir -Recurse -Force" in text

    def test_selected_backend_is_persisted_for_the_launcher(self) -> None:
        text = _text()
        assert 'set "FISHERIES_RAG_EMBEDDER=$Embedder"' in text
        assert 'Set-Content -Path $EmbedderFile -Value $Embedder -Encoding ascii' in text
    def test_default_embedder_stays_offline(self) -> None:
        """fastembed 要下 ONNX 模型，国内经常超时：不能当默认值。"""
        text = _text()
        assert '$Embedder = "fastembed"' not in text, "默认值不能是 fastembed"
        assert 'if (-not $Embedder) { $Embedder = "hash" }' in text

    def test_no_lowercase_root_variable(self) -> None:
        """``$root`` 与 ``$Root``（C:\fisheries）同名：PowerShell 变量不分大小写，
        一旦覆盖，后面的 ``Join-Path $Root`` 全部落到当前目录。"""
        text = _text()
        assert "$root =" not in text, "不要用 $root 作临时变量"
        assert "$rootName = " in text


class TestVisionOptIn:
    """视觉能力是「运行时 + 权重」两半，缺一半就只能 503，所以两半都要能靠参数装。"""

    def test_script_accepts_vision_and_weights_source(self) -> None:
        text = _text()
        assert "[switch]$Vision" in text
        assert "[string]$WeightsFrom" in text

    def test_vision_installs_ultralytics_and_checks_the_import(self) -> None:
        text = _text()
        assert "pip install ultralytics" in text
        # 装完必须真的 import 一次：torch 与 onnxruntime 版本不对时会装成功但跑不起来。
        assert "import ultralytics" in text

    def test_weights_are_copied_into_the_app_models_dir(self) -> None:
        text = _text()
        assert 'Join-Path $App "models"' in text
        assert "Copy-Item -LiteralPath" in text
        # 权重目录来源可能不存在（比如没开 RDP 驱动器重定向），不能因此整体失败。
        assert "weights source not found" in text

    def test_missing_weights_only_warn(self) -> None:
        text = _text()
        assert "function Warn" in text


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


class TestCodeUpdate:
    """`-CodeSha` 这个分支是「镜像给了旧代码」的唯一解药，必须钉住。"""

    def test_script_accepts_a_commit_sha(self) -> None:
        assert "[string]$CodeSha" in _text()

    def test_archive_url_is_pinned_to_the_commit(self) -> None:
        text = _text()
        # 用 main.zip 会被镜像缓存成旧树；按 SHA 取归档才是不可变的。
        assert "/archive/$CodeSha.zip" in text
        assert "ProjectFisheries/archive/refs/heads/main.zip" not in text

    def test_two_mirrors_are_tried(self) -> None:
        text = _text()
        assert "ghfast.top" in text and "ghproxy.net" in text
        assert "code archive download failed" in text

    def test_extracted_root_must_match_the_sha(self) -> None:
        text = _text()
        # GitHub 的归档根目录是 ProjectFisheries-<完整 40 位 sha>：
        # 只比对 7 位短 SHA 一定失败（线上真实发生过）。
        assert 'StartsWith("ProjectFisheries-")' in text
        assert "EndsWith($CodeSha)" in text, "必须接受完整 SHA 的归档根目录"
        assert "EndsWith($short)" in text, "也要接受短 SHA 的写法"

    def test_copy_never_deletes_data_or_models(self) -> None:
        text = _text()
        # robocopy /MIR 会顺手删掉 app\\models 与 app\\data；只允许 /E。
        assert "robocopy" in text and "/E" in text
        assert "/MIR" not in text


class TestLlmModelDefaults:
    def test_empty_model_gets_a_provider_default(self) -> None:
        text = _text()
        for provider, model in (
            ("dashscope", "qwen-plus"),
            ("deepseek", "deepseek-chat"),
            ("openai", "gpt-4o-mini"),
        ):
            assert f'"{provider}"' in text and f'"{model}"' in text

    def test_missing_api_key_is_warned(self) -> None:
        text = _text()
        assert "has no api key" in text


class TestAccessTokenOverride:
    def test_script_accepts_token_and_no_auth(self) -> None:
        text = _text()
        assert "[string]$Token" in text, "部署脚本缺少 -Token 参数"
        assert "[switch]$NoAuth" in text

    def test_explicit_token_is_persisted(self) -> None:
        # 以前只在「-Token 没给」时才写 token.txt，于是用 -Token 改完口令后，
        # 下一次部署不带它又会复活旧口令。显式口令必须同样落盘。
        text = _text()
        marker = "access token taken from -Token"
        assert marker in text
        before = text[: text.index(marker)]
        assert "Set-Content -Path $TokenFile -Value $Token -Encoding ascii" in before

    def test_short_token_is_warned(self) -> None:
        text = _text()
        assert "if ($Token.Length -lt 8)" in text
        assert "access token is short" in text

    def test_non_ascii_token_is_refused(self) -> None:
        text = _text()
        assert "[^\\x20-\\x7E]" in text
        assert "must be printable ASCII" in text


class TestPublicPortGuard:
    """公网入口是路由器 11438 -> 内网 8000，服务监听别的端口会「箱内全通、外面全死」。"""

    def test_non_default_port_is_warned(self) -> None:
        text = _text()
        assert "if ($Port -ne 8000)" in text, "缺少非 8000 端口的提醒"
        assert "is not 8000" in text
        assert "README 12.12" in text, "提醒里要指向 README 12.12"

class TestAgentBudget:
    """评估过慢：一次 /assess 最多两次模型往返，必须有个上限兜住。"""

    def test_budget_parameter_has_a_default(self) -> None:
        text = _text()
        assert "[int]$AgentBudgetSeconds = 70" in text, "部署脚本缺少 -AgentBudgetSeconds"

    def test_budget_reaches_the_service(self) -> None:
        text = _text()
        assert 'set "FISHERIES_AGENT_BUDGET_SECONDS=$AgentBudgetSeconds"' in text,             "预算没写进 run-server.bat，服务侧读不到"
        assert "time budget" in text, "部署结束没有回显预算"
