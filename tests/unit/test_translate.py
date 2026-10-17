"""D12「英文 prompt + 结果回译」的单元测试。

这个模块存在的理由：AQUA-7B 是纯英文微调模型（HF tags ``language: en``），
要它写中文没有质量保证；所以我们让它用英文思考，再把结果回译成中文给养殖户。
回译层的每一处降级都必须**可测、可见**——因为它是唯一可能让"英文模型"这条
技术路线在答辩现场翻车的地方：

* 译不出来必须退回中文模板，**不许留英文**（半英半中的报告比少一句建议更糟）；
* 退回必须写进 ``notes``，让调用方能把它记进 ``degraded_reasons``；
* ``risk_level`` / ``alerts`` 绝对不许过翻译——它们是护栏要比对的标识符。
"""

from __future__ import annotations

import json
import pathlib

import pytest

import fisheries.agent.translate as T
from fisheries.agent.translate import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_WORDS,
    DEFAULT_LOCAL_MODEL,
    GLOSSARY_POST_ZH,
    GLOSSARY_ZH,
    LANG_EN,
    LANG_ZH,
    LOCAL_DIR_ENV,
    UNIT_NORMALIZE,
    BackTranslation,
    BypassTranslator,
    GlossaryTranslator,
    LLMTranslator,
    LocalTranslator,
    back_translate_report,
    build_translator,
    chunk_text,
    describe_translator,
    load_translator_client,
    local_model_available,
    local_model_path,
    local_model_root,
    normalize_units,
    numbers_preserved,
    post_edit,
    post_edit_terms,
    resolve_prompt_lang,
    resolve_translator_mode,
)

# 这台机器上**真的**装了本地翻译模型（`models/translate/opus-mt-en-zh`，315 MB）。
# 单元测试绝不允许依赖"本机有没有权重"：凡是要测"没有本地模型时怎么回退"的，
# 都必须显式把模型目录指到一个不存在的地方，否则测试会随机器而变。
NO_LOCAL_MODEL = {LOCAL_DIR_ENV: str(pathlib.Path(__file__).resolve().parent / "_no_such_model")}


def _fake_local_generate(text_map):
    """给 ``LocalTranslator`` 造一个假的 ``_generate``：不加载 315 MB 权重。"""
    def _generate(chunk, *, sentence):
        return text_map[chunk]
    return _generate


# --------------------------------------------------------------------------- #
# 一、该用哪种语言发提示词
# --------------------------------------------------------------------------- #

class TestResolvePromptLang:
    """语言选择的优先级：显式入参 > 环境变量 > 按后端自动判断。"""

    def test_english_only_backend_switches_to_english(self):
        # AQUA-7B 的 provider 只有英文底座，这是 D12 的触发条件。
        assert resolve_prompt_lang(None, provider="aqua-7b-ollama") == LANG_EN
        assert resolve_prompt_lang(None, provider="aqua-7b-vllm") == LANG_EN

    def test_chinese_capable_backend_stays_chinese(self):
        for provider in ("deepseek", "dashscope", "openai", "mock"):
            assert resolve_prompt_lang(None, provider=provider) == LANG_ZH, provider

    def test_unknown_backend_defaults_to_chinese(self):
        # 默认必须是中文：默认英文等于默认让模型用非母语作答。
        assert resolve_prompt_lang(None, provider="") == LANG_ZH
        assert resolve_prompt_lang(None, provider="some-new-backend") == LANG_ZH

    def test_explicit_argument_beats_auto_detection(self):
        assert resolve_prompt_lang(LANG_ZH, provider="aqua-7b-ollama") == LANG_ZH
        assert resolve_prompt_lang(LANG_EN, provider="deepseek") == LANG_EN

    def test_env_var_beats_auto_detection(self):
        env = {"FISHERIES_PROMPT_LANG": "en"}
        assert resolve_prompt_lang(None, provider="deepseek", env=env) == LANG_EN
        env = {"FISHERIES_PROMPT_LANG": "zh"}
        assert resolve_prompt_lang(None, provider="aqua-7b-ollama", env=env) == LANG_ZH

    def test_explicit_argument_beats_env_var(self):
        env = {"FISHERIES_PROMPT_LANG": "en"}
        assert resolve_prompt_lang(LANG_ZH, provider="", env=env) == LANG_ZH

    def test_garbage_values_are_ignored_rather_than_crashing(self):
        # 环境变量里出现任何拼错的值都不能让链路挂掉，只能安静地退回自动判断。
        assert resolve_prompt_lang(None, provider="", env={"FISHERIES_PROMPT_LANG": "EN "}) == LANG_EN
        assert resolve_prompt_lang(None, provider="aqua-7b-ollama",
                                   env={"FISHERIES_PROMPT_LANG": "fr"}) == LANG_EN


# --------------------------------------------------------------------------- #
# 二、回译后端的构造
# --------------------------------------------------------------------------- #

class TestTranslatorMode:
    def test_default_mode_is_auto(self):
        assert resolve_translator_mode(None, env={}) == "auto"

    def test_explicit_mode_is_honoured(self):
        for mode in ("local", "glossary", "llm", "bypass"):
            assert resolve_translator_mode(mode, env={}) == mode

    def test_env_var_is_honoured(self):
        assert resolve_translator_mode(None, env={"FISHERIES_TRANSLATOR_MODE": "bypass"}) == "bypass"

    def test_unknown_mode_falls_back_to_auto(self):
        assert resolve_translator_mode("nonsense", env={}) == "auto"

    def test_auto_without_a_client_falls_back_to_the_glossary(self):
        # 关键约束：没网、没 key、**也没本地模型**的评委机器上链路必须照常走完。
        t = build_translator("auto", client=None, env=NO_LOCAL_MODEL)
        assert isinstance(t, GlossaryTranslator)

    def test_auto_with_a_client_uses_the_llm(self):
        t = build_translator("auto", client=object(), env=NO_LOCAL_MODEL)
        assert isinstance(t, LLMTranslator)

    def test_forced_modes_ignore_the_client(self):
        assert isinstance(build_translator("glossary", client=object(), env=NO_LOCAL_MODEL),
                          GlossaryTranslator)
        assert isinstance(build_translator("bypass", client=object(), env=NO_LOCAL_MODEL),
                          BypassTranslator)

    def test_describe_translator_is_human_readable(self):
        # 这一行会显示在 Agent 后端清单里，得让答辩现场的人一眼看懂用的是哪一种。
        assert "词表" in describe_translator(GlossaryTranslator())
        assert "翻译模型" in describe_translator(LLMTranslator(None))


class TestTranslatorClientConfig:
    """翻译模型客户端的解析：单配优先，其次复用调度角色。"""

    def test_no_config_at_all_returns_none(self):
        # 离线机器的路径：不能因此造出一个会失败的客户端。
        assert load_translator_client(env={}) is None

    def test_explicit_translator_config_is_used(self):
        env = {"FISHERIES_TRANSLATOR_PROVIDER": "deepseek",
               "FISHERIES_TRANSLATOR_API_KEY": "sk-x"}
        assert load_translator_client(env=env) is not None

    def test_scheduler_config_is_reused_when_translator_is_unset(self):
        # D12 的前提是"另一个中文模型负责回译"，而推荐部署里调度角色本来
        # 就是中文强模型。不借用它，用户得把同一个 key 配两遍。
        env = {"FISHERIES_LLM_PROVIDER": "deepseek",
               "FISHERIES_LLM_API_KEY": "sk-x"}
        assert load_translator_client(env=env) is not None

    def test_scheduler_fallback_can_be_turned_off(self):
        env = {"FISHERIES_LLM_PROVIDER": "deepseek",
               "FISHERIES_LLM_API_KEY": "sk-x"}
        assert load_translator_client(env=env, allow_scheduler_fallback=False) is None

    def test_a_scheduler_without_a_key_does_not_produce_a_translator(self):
        # provider=deepseek 但没填 key 时 is_mock 为假，可每次请求都会 401。
        # 这种"注定失败的调用"必须在构造阶段挡掉，否则每次运行都白等一次超时。
        assert load_translator_client(env={"FISHERIES_LLM_PROVIDER": "deepseek"}) is None

    def test_a_local_scheduler_needs_no_key(self):
        # 本地 ollama / vLLM 不校验 key，回译也走它。
        assert load_translator_client(
            env={"FISHERIES_LLM_PROVIDER": "aqua-7b-ollama"}) is not None

    def test_explicit_translator_config_beats_the_scheduler_fallback(self):
        env = {"FISHERIES_TRANSLATOR_PROVIDER": "dashscope",
               "FISHERIES_TRANSLATOR_API_KEY": "sk-translator",
               "FISHERIES_LLM_PROVIDER": "deepseek",
               "FISHERIES_LLM_API_KEY": "sk-scheduler"}
        client = load_translator_client(env=env)
        assert client is not None
        # 用的是 TRANSLATOR_* 那一份，不是调度角色的那一份。
        assert client.config.provider == "dashscope"

    def test_auto_mode_ends_up_with_the_llm_backend_when_a_model_exists(self):
        # "有调度角色的模型"必须能兑现成 llm 后端——但前提是**没有本地模型**，
        # 否则 local 会排在它前面。所以这里显式屏蔽本地模型。
        env = {"FISHERIES_LLM_PROVIDER": "deepseek",
               "FISHERIES_LLM_API_KEY": "sk-x", **NO_LOCAL_MODEL}
        t = build_translator("auto", client=load_translator_client(env=env), env=NO_LOCAL_MODEL)
        assert isinstance(t, LLMTranslator)


# --------------------------------------------------------------------------- #
# 三、词表回译本体
# --------------------------------------------------------------------------- #

class TestGlossaryTranslator:
    def test_short_phrase_is_translated(self):
        # 默认按"句子"处理，所以会补上句号；名词短语用法见下一条。
        out = GlossaryTranslator().translate("dissolved oxygen")
        assert out.ok and out.text == "溶氧。"

    def test_the_sentence_flag_decides_whether_a_full_stop_is_added(self):
        t = GlossaryTranslator()
        assert t.translate("dissolved oxygen", sentence=False).text == "溶氧"
        assert t.translate("dissolved oxygen", sentence=True).text == "溶氧。"

    def test_longest_match_wins(self):
        # "dissolved oxygen" 必须先于 "oxygen" 命中，否则会译成"溶氧气"。
        out = GlossaryTranslator().translate("dissolved oxygen is low")
        assert "溶氧" in out.text

    def test_noun_phrase_gets_no_full_stop(self):
        # 建议条目的 target 是名词短语，"氨氮。" 读起来是病句。
        out = GlossaryTranslator().translate("ammonia nitrogen", sentence=False)
        assert out.ok and out.text == "氨氮"

    def test_sentence_gets_a_full_stop(self):
        out = GlossaryTranslator().translate("dissolved oxygen is low")
        assert out.ok and out.text.endswith("。")

    def test_spaces_between_chinese_characters_are_collapsed(self):
        out = GlossaryTranslator().translate("turn on the aerator now")
        assert " " not in out.text.replace(" mg/L", "")

    def test_numbers_survive_translation(self):
        out = GlossaryTranslator().translate("dissolved oxygen 2.6 mg/L")
        assert out.ok and "2.6" in out.text

    def test_a_real_sentence_is_refused_rather_than_garbled(self):
        # 逐词硬译一整个句子会产出读不通的中文，那比退回模板更糟。
        long_en = ("turn on the aerator immediately and keep it running "
                   "until tomorrow morning because the oxygen will drop")
        out = GlossaryTranslator().translate(long_en)
        assert not out.ok
        assert "上限" in out.note
        assert out.text == ""

    def test_unknown_words_are_reported_rather_than_silently_dropped(self):
        out = GlossaryTranslator().translate("dissolved oxygen xyzzy")
        assert not out.ok
        assert out.coverage < 1.0

    def test_empty_input_is_a_no_op_not_a_failure(self):
        out = GlossaryTranslator().translate("   ")
        assert out.ok and out.text == ""

    def test_glossary_has_no_duplicate_keys(self):
        # 重复 key 会按"后者覆盖前者"生效，容易让同一词在不同句子译得不一样。
        keys = [k for k, _ in GLOSSARY_ZH]
        assert len(keys) == len(set(keys))

    def test_glossary_covers_the_terms_the_prompts_actually_use(self):
        # 这四条是英文提示词里出现频率最高的词，缺一个就会大面积退回模板。
        joined = " ".join(k for k, _ in GLOSSARY_ZH)
        for term in ("dissolved oxygen", "ammonia", "aerator", "feed"):
            assert term in joined, term


class TestNumberPreservation:
    """LLM 回译最容易犯的错是悄悄改数字——那会让建议与实际读数矛盾。"""

    def test_identical_numbers_pass(self):
        assert numbers_preserved("DO 2.6 mg/L", "溶氧 2.6 mg/L")

    def test_dropped_number_fails(self):
        assert not numbers_preserved("DO 2.6 mg/L", "溶氧 mg/L")

    def test_changed_number_fails(self):
        assert not numbers_preserved("DO 2.6 mg/L", "溶氧 2.8 mg/L")

    def test_no_numbers_at_all_passes(self):
        assert numbers_preserved("turn on the aerator", "开增氧机")

    def test_reordering_is_allowed(self):
        assert numbers_preserved("after 6 hours expect 3.0", "预计 3.0，6 小时后")


class TestLLMTranslator:
    """翻译模型不可用时必须返回失败而不是抛异常——它只是链路的一个可选环节。"""

    class _Client:
        def __init__(self, content: str = "", exc: Exception | None = None):
            self.content = content
            self.exc = exc
            self.calls: list[list[dict]] = []

        def chat(self, messages, tools=None):
            self.calls.append(messages)
            if self.exc:
                raise self.exc

            class R:
                pass

            r = R()
            r.content = self.content
            return r

    def test_translation_is_returned(self):
        out = LLMTranslator(self._Client("溶氧偏低，请立即开增氧机。")).translate("DO is low")
        assert out.ok and out.text == "溶氧偏低，请立即开增氧机。"

    def test_missing_client_is_reported_not_raised(self):
        out = LLMTranslator(None).translate("DO is low")
        assert not out.ok and "未配置" in out.note

    def test_network_failure_is_reported_not_raised(self):
        out = LLMTranslator(self._Client(exc=RuntimeError("boom"))).translate("DO is low")
        assert not out.ok and "RuntimeError" in out.note

    def test_rewritten_numbers_are_rejected(self):
        out = LLMTranslator(self._Client("溶氧 2.8 mg/L")).translate("DO 2.6 mg/L")
        assert not out.ok and "数值" in out.note

    def test_empty_reply_is_rejected(self):
        out = LLMTranslator(self._Client("   ")).translate("DO is low")
        assert not out.ok

    def test_the_noun_phrase_hint_is_sent_to_the_model(self):
        client = self._Client("氨氮")
        LLMTranslator(client).translate("ammonia nitrogen", sentence=False)
        sent = client.calls[0][-1]["content"]
        assert "noun phrase" in sent


class TestBypassTranslator:
    def test_english_is_returned_untouched(self):
        out = BypassTranslator().translate("Dissolved oxygen is low")
        assert out.text == "Dissolved oxygen is low"
        assert "未回译" in out.note

    def test_a_sentence_flag_is_accepted(self):
        assert BypassTranslator().translate("x", sentence=False).text == "x"


# --------------------------------------------------------------------------- #
# 四、整份报告的回译
# --------------------------------------------------------------------------- #

def _english_report() -> dict:
    return {
        "risk_level": "预警",
        "headline": "Turn on the aerator now and keep it running until 8 am.",
        "alerts": [{"code": "LEVEL_FLOOR", "level": "warning"}],
        "water_actions": [
            {"priority": 1, "target": "dissolved oxygen",
             "action": "turn on the aerator", "timing": "tonight"},
            {"priority": 2, "target": "ammonia nitrogen",
             "action": "stop feeding", "timing": "today"},
        ],
        "disease_actions": [
            {"priority": 1, "target": "gill disease",
             "action": "stop feeding for 1-2 days then use medicated feed",
             "timing": "starting today"},
        ],
        "observation_window": "measure dissolved oxygen tonight",
        "confidence_note": "healthy fish no disease found",
    }


def _chinese_fallback() -> dict:
    return {
        "risk_level": "预警",
        "headline": "【预警】溶氧开始不对，今天盯紧点。",
        "alerts": [],
        "water_actions": [
            {"priority": 1, "target": "溶氧", "action": "开增氧机", "timing": "今晚"},
            {"priority": 2, "target": "氨氮", "action": "停料", "timing": "今天"},
        ],
        "disease_actions": [
            {"priority": 1, "target": "烂鳃", "action": "停料 1–2 天改投药饵", "timing": "今天开始"},
        ],
        "observation_window": "6 小时后复测溶氧和氨氮。",
        "confidence_note": "鱼体健康，未见病害。",
    }


class TestBackTranslateReport:
    def test_chinese_input_is_returned_unchanged(self):
        parsed = _english_report()
        out = back_translate_report(parsed, translator=GlossaryTranslator(),
                                    fallback=_chinese_fallback(), lang=LANG_ZH)
        assert out.parsed["headline"] == parsed["headline"]

    def test_risk_level_and_alerts_are_never_touched(self):
        # 护栏要比对 risk_level；它一旦被翻译层改写，整条防线的意义就没了。
        parsed = _english_report()
        out = back_translate_report(parsed, translator=GlossaryTranslator(),
                                    fallback=_chinese_fallback(), lang=LANG_EN)
        assert out.parsed["risk_level"] == "预警"
        assert out.parsed["alerts"] == parsed["alerts"]

    def test_untranslatable_headline_falls_back_to_the_chinese_template(self):
        out = back_translate_report(_english_report(), translator=GlossaryTranslator(),
                                    fallback=_chinese_fallback(), lang=LANG_EN)
        assert out.parsed["headline"] == _chinese_fallback()["headline"]
        assert any("headline" in n for n in out.notes)

    def test_translatable_short_fields_are_actually_translated(self):
        out = back_translate_report(_english_report(), translator=GlossaryTranslator(),
                                    fallback=_chinese_fallback(), lang=LANG_EN)
        assert out.parsed["water_actions"][0]["target"] == "溶氧"

    def test_noun_phrase_fields_carry_no_full_stop(self):
        out = back_translate_report(_english_report(), translator=GlossaryTranslator(),
                                    fallback=_chinese_fallback(), lang=LANG_EN)
        for action in out.parsed["water_actions"]:
            assert not action["target"].endswith("。"), action

    def test_a_failed_item_uses_the_same_index_from_the_template(self):
        # 逐条补位而不是整表替换：不该因为一条译不出来就丢掉其余译得好的建议。
        out = back_translate_report(_english_report(), translator=GlossaryTranslator(),
                                    fallback=_chinese_fallback(), lang=LANG_EN)
        assert out.parsed["water_actions"][1]["target"] == "氨氮"
        assert out.n_replaced >= 1
        assert out.n_translated >= 1

    def test_notes_are_recorded_for_every_degradation(self):
        out = back_translate_report(_english_report(), translator=GlossaryTranslator(),
                                    fallback=_chinese_fallback(), lang=LANG_EN)
        assert out.notes, "回译降级必须留下痕迹，否则等于静默降级"

    def test_without_a_fallback_a_failed_field_is_dropped_not_left_english(self):
        out = back_translate_report(_english_report(), translator=GlossaryTranslator(),
                                    fallback=None, lang=LANG_EN)
        # 字段被整个删掉（而不是留一个空串），调用方因此能看出"这一段没了"。
        assert "headline" not in out.parsed or "Turn on the aerator" not in out.parsed["headline"]
        assert out.parsed.get("headline", "") == ""

    def test_without_a_fallback_failed_items_are_dropped(self):
        out = back_translate_report(_english_report(), translator=GlossaryTranslator(),
                                    fallback=None, lang=LANG_EN)
        for action in out.parsed["water_actions"] + out.parsed["disease_actions"]:
            assert "stop feeding for" not in action["action"]

    def test_bypass_mode_keeps_english_for_prompt_debugging(self):
        out = back_translate_report(_english_report(), translator=BypassTranslator(),
                                    fallback=_chinese_fallback(), lang=LANG_EN)
        assert out.parsed["headline"] == _english_report()["headline"]
        assert out.backend == "bypass"

    def test_missing_keys_do_not_crash(self):
        out = back_translate_report({}, translator=GlossaryTranslator(),
                                    fallback=_chinese_fallback(), lang=LANG_EN)
        assert isinstance(out, BackTranslation)

    def test_priority_is_not_translated(self):
        out = back_translate_report(_english_report(), translator=GlossaryTranslator(),
                                    fallback=_chinese_fallback(), lang=LANG_EN)
        assert out.parsed["water_actions"][0]["priority"] == 1


# --------------------------------------------------------------------------- #
# 五、端到端：英文提示词 + 英文 JSON → 中文报告
# --------------------------------------------------------------------------- #

class _EnglishAdvisor:
    """假扮一个只会说英文的 AQUA-7B：无论问什么都用英文 JSON 回答。"""

    def __init__(self, payload: dict | None = None):
        self.payload = payload if payload is not None else _english_report()
        self.seen: list[dict] = []

    def chat(self, messages, tools=None):
        self.seen = messages

        class R:
            pass

        r = R()
        r.content = json.dumps(self.payload, ensure_ascii=False)
        return r


class TestOrchestratorEnglishPath:
    def _agent(self, advisor, translator=None):
        from fisheries.agent.orchestrator import AquacultureAgent
        # 默认钉死成词表后端。理由有两层：
        # ① 本机真的装了本地翻译模型，不钉死的话 ``auto`` 会去加载 315 MB 权重，
        #    单文件测试从 3 秒变成 20 秒，而且结果随机器而变；
        # ② 这个类测的是**编排层**（提示词语言、护栏、降级留痕），
        #    不是回译后端的分派——分派在 TestLocalBackend 里单独测。
        if translator is None:
            translator = GlossaryTranslator()
        return AquacultureAgent(advisor_client=advisor, use_knowledge=False,
                                prompt_lang=LANG_EN, translator=translator)

    def _run(self, agent):
        from datetime import datetime, timedelta

        from fisheries.agent.orchestrator import AgentInput

        base = datetime(2026, 10, 13, 8, 0, 0)
        readings = [
            {"timestamp": (base + timedelta(hours=i)).isoformat(),
             "water_temp": 27.0, "ph": 7.8,
             "dissolved_oxygen": round(6.4 - 0.09 * i, 3),
             "ammonia_n": round(0.10 + 0.010 * i, 3),
             "nitrite_n": round(0.05 + 0.004 * i, 3)}
            for i in range(24)
        ]
        return agent.run(AgentInput(pond_id="t", water_readings=readings))

    def test_prompt_sent_to_the_model_is_english(self):
        advisor = _EnglishAdvisor()
        self._run(self._agent(advisor))
        system = advisor.seen[0]["content"]
        assert "aquaculture" in system.lower()
        assert "养殖" not in system

    def test_the_advisory_handed_back_to_the_user_is_chinese(self):
        res = self._run(self._agent(_EnglishAdvisor()))
        assert res.prompt_lang == LANG_EN
        assert res.headline
        assert not any("a" <= ch.lower() <= "z" for ch in res.headline), res.headline

    def test_a_degraded_translation_is_reported(self):
        # 钉死成词表后端：headline 是整句英文，词表译不出来 →
        # 必须留下"退回模板"的痕迹。这条测的是**降级路径本身**，不是分派。
        res = self._run(self._agent(_EnglishAdvisor(), translator=GlossaryTranslator()))
        assert res.translation_backend == "glossary"
        assert res.degraded_reasons
        assert any("回译" in r for r in res.degraded_reasons)

    def test_risk_level_still_comes_from_the_fusion_layer(self):
        payload = dict(_english_report(), risk_level="正常")
        res = self._run(self._agent(_EnglishAdvisor(payload)))
        # 模型说"正常"、融合层说"预警" → 护栏作废整份输出，等级仍是融合层的。
        assert res.risk_level != "正常"

    def test_chinese_path_does_not_invoke_the_translator(self):
        from fisheries.agent.orchestrator import AquacultureAgent

        advisor = _EnglishAdvisor(_chinese_fallback())
        agent = AquacultureAgent(advisor_client=advisor, use_knowledge=False,
                                 prompt_lang=LANG_ZH, translator=GlossaryTranslator())
        assert agent.prompt_lang == LANG_ZH
        assert agent.advisor is advisor
        assert not agent.translation_notes

    def test_translation_state_does_not_leak_between_runs(self):
        advisor = _EnglishAdvisor()
        agent = self._agent(advisor, translator=GlossaryTranslator())
        self._run(agent)
        assert agent.translation_notes
        # 第二次换成中文路径：上一次的"回译失败"不能继续挂在结果上。
        agent.prompt_lang = LANG_ZH
        res = self._run(agent)
        assert res.translation_notes == []


# --------------------------------------------------------------------------- #
# 六、本地离线后端（D12 收益真正兑现的路径）
# --------------------------------------------------------------------------- #

class TestLocalModelAvailability:
    """模型可用性真值表。

    这一层出错的表现是**静默退回词表**——报告照样出得来，只是长句全被丢掉、
    每段都退回中文模板。所以四条前置条件（目录 / config.json / 权重 / 依赖）
    各自都要有一条测试，而且**不能依赖开发机上装没装模型**。
    """

    def test_default_root_is_models_translate(self):
        root = local_model_root({})
        assert root.name == "translate"
        assert root.parent.name == "models"

    def test_dir_env_overrides_the_root(self, tmp_path):
        assert local_model_root({LOCAL_DIR_ENV: str(tmp_path)}) == tmp_path

    def test_model_name_is_appended_to_the_root(self, tmp_path):
        got = local_model_path("foo", {LOCAL_DIR_ENV: str(tmp_path)})
        assert got == tmp_path / "foo"

    def test_absolute_model_path_is_used_as_is(self, tmp_path):
        assert local_model_path(str(tmp_path), {}) == tmp_path

    def test_the_default_model_name_is_the_opus_one(self, tmp_path):
        got = local_model_path(None, {LOCAL_DIR_ENV: str(tmp_path)})
        assert got.name == DEFAULT_LOCAL_MODEL

    def test_missing_directory_reason_says_what_to_run(self, tmp_path):
        ok, why = local_model_available(str(tmp_path / "nope"))
        assert not ok
        assert "fetch_translate_model.py" in why

    def test_config_json_alone_is_not_enough(self, tmp_path):
        target = tmp_path / "m"
        target.mkdir()
        (target / "config.json").write_text("{}", encoding="utf-8")
        ok, why = local_model_available(str(target))
        assert not ok
        assert "权重" in why

    def test_weight_alone_is_not_enough(self, tmp_path):
        target = tmp_path / "m"
        target.mkdir()
        (target / "pytorch_model.bin").write_bytes(b"x")
        ok, why = local_model_available(str(target))
        assert not ok
        assert "config.json" in why

    def test_safetensors_counts_as_a_weight(self, tmp_path):
        pytest.importorskip("transformers")
        target = tmp_path / "m"
        target.mkdir()
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"x")
        ok, why = local_model_available(str(target))
        assert ok, why

    def test_a_complete_directory_is_available_with_no_reason(self, tmp_path):
        pytest.importorskip("transformers")
        target = tmp_path / "m"
        target.mkdir()
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "pytorch_model.bin").write_bytes(b"x")
        ok, why = local_model_available(str(target))
        assert ok
        assert why == ""

    def test_local_translator_reports_the_same_verdict(self, tmp_path):
        tr = LocalTranslator(model=str(tmp_path / "nope"))
        ok, why = tr.available()
        assert not ok
        assert "fetch_translate_model.py" in why
        assert tr.backend == "local"


class TestChunkText:
    """分块：解决"整句译到一半丢尾部"。

    标尺是一条 17 词的完整句——不做分块时模型必然丢掉结尾的 ``at 6``，
    而数值校验会因此把整个字段判为失败。
    """

    LONG = ("Turn on the aerator tonight and measure dissolved oxygen "
            "and ammonia nitrogen again tomorrow morning at 6.")

    def test_short_text_stays_in_one_piece(self):
        assert chunk_text("Dissolved oxygen is low.") == ["Dissolved oxygen is low."]

    def test_blank_text_yields_nothing(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_the_long_sentence_is_actually_split(self):
        assert len(chunk_text(self.LONG)) > 1

    def test_every_piece_respects_the_default_budget(self):
        for piece in chunk_text(self.LONG):
            assert len(piece) <= DEFAULT_CHUNK_CHARS, piece
            assert len(piece.split()) <= DEFAULT_CHUNK_WORDS, piece

    def test_no_word_is_lost_in_the_split(self):
        joined = " ".join(chunk_text(self.LONG))
        for token in ("aerator", "dissolved", "oxygen", "ammonia", "nitrogen", "6"):
            assert token in joined

    def test_a_number_and_its_unit_are_never_separated(self):
        # 数值校验看的是"数字多重集"，把 5.0 和 mg/L 拆到两块里会让
        # 两块各自都通不过校验，于是整字段失败——所以必须粘在一起。
        text = ("Check the pond at 6 and again at 12 and keep the dissolved "
                "oxygen above 5.0 mg/L until the morning and then report back.")
        parts = chunk_text(text)
        assert any("5.0 mg/L" in piece for piece in parts)
        for piece in parts:
            assert not piece.rstrip().endswith("5.0"), piece
            assert not piece.lstrip().startswith("mg/L"), piece

    def test_the_word_budget_is_the_real_limit_not_the_char_budget(self):
        # 这就是 110/16 的来由：17 词的句子把字符预算放宽到 200 也切不开，
        # 词预算一收到 16 就切开了。
        assert len(chunk_text(self.LONG, max_chars=200, max_words=40)) == 1
        assert len(chunk_text(self.LONG, max_chars=110, max_words=16)) == 2

    def test_a_tight_budget_still_terminates_and_keeps_every_word(self):
        parts = chunk_text(self.LONG, max_chars=20, max_words=3)
        assert parts
        assert " ".join(parts).split() == self.LONG.split()


class TestPostEdit:
    """后订正：让 NMT 负责语序、让词表负责术语。"""

    @pytest.mark.parametrize("before,after", [
        ("溶解的氧气", "溶氧"),
        ("溶解氧气", "溶氧"),
        ("溶氧气", "溶氧"),
        ("亚硝酸", "亚硝酸盐"),
        ("亚硝酸氮", "亚硝酸盐"),
        ("多发性", "浊度"),
        ("发炎率", "浊度"),
        ("易发性", "浊度"),
        ("细菌刺痛病", "烂鳃病"),
        ("细菌刺状疾病", "烂鳃病"),
        ("细菌红病", "细菌性败血症"),
        ("萨普罗格尼亚", "水霉病"),
        ("呼吸器", "增氧机"),
        ("气动器", "增氧机"),
        ("加速器", "增氧机"),
        ("增加空气循环", "增氧"),
        ("增加气流", "增氧"),
        ("丝袜密度", "放养密度"),
        ("先生生物", "益生菌"),
        ("佐利特", "沸石"),
        ("快生石灰灰", "生石灰"),
        ("速力石", "生石灰"),
        ("漂白粉粉", "漂白粉"),
        ("网状图", "拉网"),
        ("海底沉积物", "底泥"),
        ("药物喂养", "药饵"),
        ("水交换", "换水"),
        ("人口", "存塘量"),
        ("今夜今晚", "今晚"),
    ])
    def test_a_known_mistranslation_is_corrected(self, before, after):
        assert post_edit_terms(before) == after

    @pytest.mark.parametrize("before,after", [
        ("1.5毫克/升", "1.5 mg/L"),
        ("0.3微克/升", "0.3 μg/L"),
        ("28.5摄氏度", "28.5 ℃"),
        ("42NTU", "42 NTU"),
        ("7.2pH", "7.2 pH"),
        ("纽芬兰先令", "NTU"),
    ])
    def test_a_unit_is_normalised(self, before, after):
        assert normalize_units(before) == after

    def test_cascade_protection_does_not_corrupt_a_correct_term(self):
        # 表里有 "亚硝酸" -> "亚硝酸盐"。逐条 sub、按插入顺序扫描会把
        # 已经正确的 "亚硝酸盐" 再切一次变成 "亚硝酸盐盐"，
        # 所以实现必须是一遍扫描 + 最长优先。
        assert post_edit_terms("亚硝酸盐") == "亚硝酸盐"
        assert post_edit_terms("亚硝酸盐超标") == "亚硝酸盐超标"

    def test_a_correct_chinese_phrase_is_left_alone(self):
        assert post_edit_terms("采取措施") == "采取措施"
        assert post_edit_terms("溶氧偏低，建议立即增氧。") == "溶氧偏低，建议立即增氧。"

    def test_post_edit_applies_both_tables(self):
        out = post_edit("溶解氧气 42NTU")
        assert "溶氧" in out
        assert "42 NTU" in out

    def test_deliberately_excluded_words_are_absent_from_the_table(self):
        # 这三个词在别的语境里是正当中文，收进来会误伤——
        # 这个"不收"的决定要有测试守着，否则以后很容易被顺手加回去。
        for word in ("措施", "发病率", "中度"):
            assert all(source != word for source, _ in GLOSSARY_POST_ZH), word

    def test_both_tables_have_unique_sources(self):
        for table in (GLOSSARY_POST_ZH, UNIT_NORMALIZE):
            sources = [source for source, _ in table]
            assert len(sources) == len(set(sources))
            assert all(source.strip() for source, _ in table)
            assert all(target.strip() for _, target in table)

    def test_the_only_self_mapping_is_the_documented_placeholder(self):
        # 恒等规则是"占位"用的（见 GLOSSARY_POST_ZH 里"亚硝酸盐"那条的注释）：
        # 它存在的唯一理由是让一个**正确写法**作为整体被消费掉，
        # 从而不被另一条更短的规则切碎。除此以外不该有恒等规则——
        # 恒等规则是噪声，多了就没人知道哪条是必要的。
        identity = {source for source, target in GLOSSARY_POST_ZH if source == target}
        assert identity == {"亚硝酸盐"}
        assert not any(source == target for source, target in UNIT_NORMALIZE)

    def test_the_placeholder_is_actually_needed(self):
        # 去掉占位规则，"亚硝酸盐"就会被"亚硝酸"切碎成"亚硝酸盐盐"。
        # 这条锁住"它是必要的、不是顺手加的"。
        without = tuple((s, t) for s, t in GLOSSARY_POST_ZH if s != "亚硝酸盐")
        lookup = dict(without)
        broken = T._compile_alternation(without).sub(
            lambda m: lookup[m.group(0)], "亚硝酸盐")
        assert broken == "亚硝酸盐盐"

    def test_every_replacement_is_itself_stable(self):
        """**替换结果必须能被同一个正则原样消费掉。**

        这条不变量抓的是一类很难肉眼发现的缺陷：交替正则按最长优先匹配，
        只要某条规则的**替换结果**（或任何本工程认定的正确写法）以另一条
        规则的**键**开头、而它自己又不是键，就会被那条款规则切掉前半截。

        实际踩到的例子：``亚硝酸 -> 亚硝酸盐`` 会让已经正确的
        ``亚硝酸盐`` 变成 ``亚硝酸盐盐``。修法是给 ``亚硝酸盐`` 补一条
        恒等规则，让它作为整体被消费。
        """
        for table, fn in ((GLOSSARY_POST_ZH, post_edit_terms),
                          (UNIT_NORMALIZE, normalize_units)):
            for source, target in table:
                assert fn(target) == target, f"{source} -> {target} 不稳定"


def _fake_model_dir(tmp_path):
    """造一个"结构上齐全"的模型目录，让 ``available()`` 放行。"""
    target = tmp_path / "m"
    target.mkdir(exist_ok=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "pytorch_model.bin").write_bytes(b"x")
    return target


def _stub_local(tmp_path, generate):
    """造一个已"加载"好的 ``LocalTranslator``，并把 ``_generate`` 换成假的。

    必须同时把 ``_model`` 设成非 ``None``：``_one_call`` 会先调 ``_load()``，
    而 ``_load()`` 一旦真去读那个只有 ``{}`` 的 ``config.json`` 就会抛
    ``ValueError: Unrecognized model``。填上 ``_model`` 让它提前返回。
    """
    tr = LocalTranslator(model=str(_fake_model_dir(tmp_path)))
    tr._model = object()
    tr._tokenizer = object()
    tr._kind = "seq2seq"
    tr._generate = generate
    return tr


class TestLocalTranslatorPipeline:
    """用假的 ``_generate`` 测管线本身。

    **绝不加载 315 MB 权重**：``_model`` 被填成占位对象（``_load()`` 提前返回），
    ``_generate`` 被替换成假的。
    """

    @staticmethod
    def _translator(tmp_path, mapping):
        return _stub_local(tmp_path, _fake_local_generate(mapping))

    def test_a_piece_is_translated_and_post_edited(self, tmp_path):
        tr = self._translator(tmp_path, {"Dissolved oxygen is low.": "溶解氧气低。"})
        out = tr.translate("Dissolved oxygen is low.")
        assert out.ok, out.note
        assert out.backend == "local"
        assert out.text == "溶氧低。"

    def test_a_number_that_disappears_fails_the_field(self, tmp_path):
        tr = self._translator(tmp_path, {"Keep it above 5.0 mg/L.": "保持在以上。"})
        out = tr.translate("Keep it above 5.0 mg/L.")
        assert not out.ok
        assert "数值" in out.note

    def test_a_number_kept_verbatim_passes(self, tmp_path):
        tr = self._translator(tmp_path, {"Keep it above 5.0 mg/L.": "保持在 5.0 mg/L 以上。"})
        out = tr.translate("Keep it above 5.0 mg/L.")
        assert out.ok, out.note

    def test_blank_output_is_a_failure(self, tmp_path):
        tr = self._translator(tmp_path, {"Dissolved oxygen is low.": "   "})
        out = tr.translate("Dissolved oxygen is low.")
        assert not out.ok
        assert "空" in out.note

    def test_blank_input_succeeds_without_calling_the_model(self, tmp_path):
        calls = []

        def _generate(chunk, *, sentence):
            calls.append(chunk)
            return "不该被调用"

        tr = _stub_local(tmp_path, _generate)
        out = tr.translate("   ")
        assert out.ok
        assert out.text == ""
        assert calls == []

    def test_one_failed_piece_fails_the_whole_field(self, tmp_path):
        # 底线是"绝不半英半中"：只要有一块没译出来，整个字段就不能算成功，
        # 调用方会把该字段退回中文模板。
        seen = []

        def _generate(chunk, *, sentence):
            seen.append(chunk)
            return "收到。" if len(seen) == 1 else ""

        tr = _stub_local(tmp_path, _generate)
        out = tr.translate(
            "One two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen."
        )
        assert len(seen) > 1, "这条测试的前提是句子被切成了多块"
        assert not out.ok

    def test_too_many_pieces_is_refused_instead_of_hanging(self, tmp_path):
        tr = LocalTranslator(model=str(_fake_model_dir(tmp_path)),
                             chunk_words=2, max_chunks=3)
        tr._model = object()
        tr._tokenizer = object()
        tr._generate = _fake_local_generate({})
        out = tr.translate(" ".join(f"w{i}" for i in range(40)) + ".")
        assert not out.ok

    def test_an_inference_error_is_reported_not_raised(self, tmp_path):
        def _boom(chunk, *, sentence):
            raise RuntimeError("CUDA out of memory")

        tr = _stub_local(tmp_path, _boom)
        out = tr.translate("Dissolved oxygen is low.")
        assert not out.ok
        assert "推理失败" in out.note
        assert "CUDA out of memory" in out.note

    def test_a_load_failure_is_reported_not_raised(self, tmp_path):
        # 模型目录不存在 → _load() 抛 RuntimeError → 必须变成 note，而不是异常。
        # 这条**故意不填 _model**，要的就是真走一次 _load() 的失败路径。
        tr = LocalTranslator(model=str(tmp_path / "nope"))
        out = tr.translate("Dissolved oxygen is low.")
        assert not out.ok
        assert "加载失败" in out.note


class TestLocalDispatch:
    """``auto`` 的顺序是这一层收益的关键：本地 → 远端 client → 词表。"""

    def test_explicit_local_without_a_model_falls_back_to_the_glossary(self):
        got = build_translator("local", env=NO_LOCAL_MODEL)
        assert isinstance(got, GlossaryTranslator)

    def test_auto_without_local_and_without_a_client_uses_the_glossary(self):
        got = build_translator("auto", env=NO_LOCAL_MODEL)
        assert isinstance(got, GlossaryTranslator)

    def test_auto_without_local_but_with_a_client_uses_the_llm(self):
        got = build_translator("auto", client=object(), env=NO_LOCAL_MODEL)
        assert isinstance(got, LLMTranslator)

    def test_auto_prefers_the_local_model_when_it_exists(self, monkeypatch):
        monkeypatch.setattr(T, "local_model_available", lambda *a, **k: (True, ""))
        assert isinstance(build_translator("auto"), LocalTranslator)

    def test_auto_prefers_local_over_a_configured_client(self, monkeypatch):
        # 这一条就是"收益名义化"的修复：以前 auto 只看 client，
        # 离线机器必然落到词表后端，长句全丢。
        monkeypatch.setattr(T, "local_model_available", lambda *a, **k: (True, ""))
        got = build_translator("auto", client=object())
        assert isinstance(got, LocalTranslator)

    def test_auto_falls_back_when_the_local_model_disappears(self, monkeypatch):
        monkeypatch.setattr(T, "local_model_available",
                            lambda *a, **k: (False, "模型被删了"))
        got = build_translator("auto", client=object())
        assert isinstance(got, LLMTranslator)

    def test_describe_names_the_model_directory(self):
        text = describe_translator(LocalTranslator(model=DEFAULT_LOCAL_MODEL))
        assert "本地" in text
        assert DEFAULT_LOCAL_MODEL in text
