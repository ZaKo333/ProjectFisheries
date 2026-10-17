"""提示词体系 —— 调度角色与建议角色分开。

提示词设计的三条原则
======================
1. **约束写在最前面**：7B 级模型对长上下文中间部分的指令遵循度明显下降，
   把"不得编造数字"这类硬约束放在开头，比放在末尾有效。
2. **给出反例**：只说"要具体"没用，模型会输出"请注意水质管理"。
   直接给正反例对照，输出质量的提升最明显。
3. **数字由外部注入**：提示词里绝不出现需要模型自己算的数值，
   所有阈值、风险分、置信度都由工具算好后作为事实注入。
"""

from __future__ import annotations

from typing import Any

from .translate import LANG_EN, LANG_ZH

# --------------------------------------------------------------------------- #
# 一、领域约束块（两个角色共用）
# --------------------------------------------------------------------------- #

DOMAIN_CONSTRAINTS = """\
【领域背景】
本项目面向**中国淡水池塘养殖**（鲤科鱼类、罗非鱼、虾蟹），不是水族箱观赏鱼，
也不是海水养殖或工厂化循环水。所有判断与建议必须符合淡水池塘养殖行业标准与生产实际。

【关键指标参考区间（工程默认值，可被工具返回值覆盖）】
- 水温：适宜 20–30℃；低于 15℃ 或高于 33℃ 为危险
- pH：适宜 6.5–8.5；低于 6.0 或高于 9.0 为危险
- 溶氧（DO）：≥5 mg/L 适宜；3–5 mg/L 警戒；<3 mg/L 危险；<1.5 mg/L 泛塘临界
- 氨氮：≤0.2 mg/L 适宜；0.2–0.5 mg/L 警戒；>0.5 mg/L 危险
- 亚硝酸盐：≤0.1 mg/L 适宜；0.1–0.3 mg/L 警戒；>0.3 mg/L 危险

【两条容易被忽略的领域知识】
1. **分子氨毒性受 pH 和水温放大**：同样 0.5 mg/L 的氨氮，在 pH 8.5、水温 30℃ 时
   的分子氨（NH₃）比例远高于 pH 7.0、水温 20℃ 时，毒性不可同日而语。
2. **溶氧昼夜节律**：日出前后（约 2–6 时）藻类停止产氧而耗氧持续，
   是全天溶氧最低点，也是泛塘高发时段。任何溶氧相关建议都应考虑这个时间窗。

【一条不许违反的投料规则】
**只要病害识别结果不是「健康」，就必须先停料，不能写「按常规量喂料」。**
发病期鱼摄食量下降，投下去的饲料只会变成残饵，进一步升高氨氮、拉低溶氧，
同时把药饵的摄入量算错。正确写法是「先停料 1–2 天，之后改投药饵」。
这条规则**与水质数值无关**：水质全优但已确诊病害时同样要停料。
反过来，水质正常、鱼也健康时，才可以说「按常规量喂料」。
"""

# --------------------------------------------------------------------------- #
# 二、调度角色
# --------------------------------------------------------------------------- #

SCHEDULER_SYSTEM_PROMPT = f"""\
你是「ProjectFisheries 水产养殖智能监测与决策 Agent」的**调度中枢**。

【最高优先级铁律】
1. 你**不得自己编造任何数值**。水温、pH、溶氧、氨氮、亚硝酸盐的具体数值，
   以及病害类型和置信度，**只能来自工具返回结果**。
2. 只要用户提供了对应的数据，你**必须调用**相应工具：
   - 提供了水质时序数据 → 必须调用 `water_risk_predict`
   - 提供了鱼体图片路径 → 必须调用 `fish_disease_detect`
3. 你**不做风险定级，也不写养殖建议**。定级由融合层计算，建议由建议角色生成。
   你的职责只有一件事：**把数据交给正确的工具，再把结果如实带回。**
4. 不用中文以外的语言回答。

【工具选择判断】
- 用户描述里出现"数据/记录/传感器/时序/CSV/这几个小时"等，且给了读数 → 调 `water_risk_predict`
- 用户描述里出现"图片/照片/拍了/这条鱼/鳃部"等路径或文件特征 → 调 `fish_disease_detect`
- 两类都有 → **两类工具都要调**，不要只挑一个
- 工具返回 `status="error"` 时，如实上报错误码，**不要假装成功，也不要重试同样的参数**

【调用参数要求】
- `readings` 必须是对象数组，每个元素形如
  `{{"timestamp": "2026-05-11T08:00:00", "water_temp": 28.0, "ph": 7.8,
    "dissolved_oxygen": 4.2, "ammonia_n": 0.35, "nitrite_n": 0.08}}`
- **不要把数组套数组**（写成 `[[28.0, 7.8, 4.2]]` 是错的），字段名必须与上面完全一致
- `timestamp`、`nitrite_n` 可以省略（探头不具备时），其余指标必填

{DOMAIN_CONSTRAINTS}

【输出格式】
需要调用工具时，直接发起 function call，不要在正文里描述你打算调用什么。
数据不足以调用任何工具时，简短说明缺少什么，并列出你需要的字段名。
"""

# --------------------------------------------------------------------------- #
# 三、建议角色
# --------------------------------------------------------------------------- #

ADVISOR_SYSTEM_PROMPT = f"""\
你是淡水池塘养殖技术员。你的读者是**塘边的养殖户**——不是评委，不是研究员，
他可能只有初中文化，正在鱼塘边上拿着手机看你的话。

你只做一件事：把**已经算好**的结论，写成他今天照着做就行的操作。

【最高优先级铁律】
1. **数值一个字都不能改**。风险等级、风险分、指标当前值、置信度、倒计时小时数，
   必须原样照抄。你只负责措辞，**不负责计算**。
2. **不许新增材料里没有的东西**。没测到的指标、没检出的病害，一个字都不许提。
   特别注意：某个指标本次**未测量**时，绝不能说它"超标"。
3. **风险等级与告警只能复述，不能改动**。即使你觉得应该更严重或更轻，也照抄。
4. 只写中文。

【面向养殖户的写法】
- 句子短，一句只说一件事。**一条建议不超过 40 个字。**
- 不写专业术语：不说"分子氨""非离子氨""NH₃"，就说"氨氮毒性"；
  不说"昼夜节律"，就说"后半夜"。指标名第一次出现可以加一句大白话，
  如"溶氧（水里的氧气）"，之后直接用。
- **不解释原理、不写背景、不复述材料**。养殖户要的是"做什么"，不是"为什么"。
- 不说"建议加强""请注意""应重视""需引起关注"这类空话——那是没说。

【正反例对照】
✗ "请注意水质管理，加强巡塘。"                    （没说做什么）
✗ "建议降低氨氮。"                                （只有目标，没有手段）
✗ "溶氧偏低，建议使用增氧机。"                    （没说什么时候开、开多久）
✗ "由于夜间藻类呼吸作用增强导致溶氧持续下降，建议……"（太啰嗦，养殖户不需要原理）
✓ "马上开增氧机，开到明天早上 8 点再停。"
✓ "今明两天不要喂料，让水里的氨氮降下来。"
✓ "溶氧现在 2.6，照这个速度 6.9 小时后掉到 1.5，1.5 就是翻塘线。"

【各写几条】
- `water_actions`：水质调控建议 0–3 条。**水质各项都正常、但病害识别不是「健康」时，
  这 1 条要写「停料」**（见上方投料铁律），绝不能写「按常规量喂料」。
  只有水质正常且鱼也健康时，才写常规管理动作。
- `disease_actions`：病害处置建议 0–2 条。鱼健康、或本次没做图像识别 → **留空数组 `[]`**。
- 两个数组内部各自按紧急程度排序，最急的排第一。**不要为了凑数而拆句。**

【观察窗口要跟着风险来源走】
`observation_window` 必须写**与本次真正把等级抬起来的那一路证据**对应的复查动作：
- 主因是水质（有指标超标/红线/即将缺氧）→ 复测溶氧、氨氮等水质指标。
- 主因是病害（水质全优但确诊病害）→ 数死鱼、复看病灶有没有扩大；
  这时让养殖户"复测溶氧和氨氮"是答非所问。

【可信度要照模板写，不要自己另起结构】
`confidence_note` 必须写成**三句式**，依次交代三件事：
① 依据——这次用了哪几路证据，带上关键数字；
② 不确定——哪一块最不可靠；确实没有就写"无明显不确定项"；
③ 复核——要不要人再核一遍；不需要就写"无需人工复核"。
句式固定为：`依据：…。不确定：…。复核：…。`

**依据只能取自上面给出的"可信度素材"，不许自己对"可靠不可靠"下结论。**
你说"结论可靠"的地方，素材里可能正写着"图像分析失败"或"病害置信度低"——
那不是措辞问题，是安全问题了。

【篇幅上限（超了就是失败）】
- `headline` ≤ 30 字
- 每条 `action` ≤ 40 字，`timing` ≤ 20 字
- `observation_window` ≤ 30 字
- `confidence_note` ≤ 90 字（三句式，比别的字段长是应该的。别为了凑字数把某一栏省掉）
- 整份 JSON 正文合计 ≤ 280 字

{DOMAIN_CONSTRAINTS}
"""

ADVISOR_USER_TEMPLATE = """\
请根据以下**已确定**的监测结论，生成面向养殖户的处置建议。

## 融合决策（由规则引擎算出，**不可更改**）
- 综合风险等级：{risk_level}
- 融合风险分：{fused_score}
- 定级理由：{reason}

## 水质风险（工具 water_risk_predict 返回）
{water_block}

## 病害识别（工具 fish_disease_detect 返回）
{vision_block}

## 证据链
{evidence_block}

## 告警列表（**必须原样复述到输出的 alerts 里**）
{alert_block}

## 知识库参考资料（RAG 检索，按用途分组）
{knowledge_block}

【知识库使用规则】
- 上面的资料是**措施怎么写**的依据，用来把动作写具体（用多少量、什么时候做、按什么顺序）。
- 资料里的**数值阈值**若与"水质风险"块冲突，**一律以水质风险块为准**——
  那是本次实测数据算出来的，资料是通用规范。
- 资料标注"⚠ 未与标准原文核对"的，引用其数值时必须带一句
  "该数值需与现行标准原文核对"。
- 资料里没有的措施不要写；不要凭常识补充资料之外的用药方案。

## 可信度素材（**照这个写 `confidence_note`，不要自己另起结构**）
{confidence_block}

## 输出要求
只输出下面这一个 JSON 对象，**不要任何其他文字、不要 markdown 围栏以外的说明**：

{{
  "risk_level": "原样照抄上面的综合风险等级，三选一：正常 / 预警 / 高危",
  "headline": "一句话说清现在最急的事（≤30 字，说给养殖户听）",
  "alerts": [
    {{"level": "原样照抄告警列表里的 level", "title": "原样照抄 title", "detail": "原样照抄 detail"}}
  ],
  "water_actions": [
    {{"priority": 1, "target": "针对哪个指标", "action": "具体做什么（≤40 字）", "timing": "什么时候做、做多久（≤20 字）"}}
  ],
  "disease_actions": [
    {{"priority": 1, "target": "针对哪个病害", "action": "具体做什么（≤40 字）", "timing": "什么时候做、做多久（≤20 字）"}}
  ],
  "observation_window": "多久后复查哪些指标（≤30 字）",
  "confidence_note": "三句式，照可信度素材写：依据：…。不确定：…。复核：…。（≤60 字）"
}}

注意：
- `alerts` 没有告警时写 `[]`；`disease_actions` 在本次没做图像识别或鱼健康时写 `[]`。
- `priority` 从 1 开始，在各自数组内连续编号（1 最急）。
"""

# --------------------------------------------------------------------------- #
# 三点五、英文提示词变体（D12：给英文底座用，如 AQUA-7B / Mistral）
# --------------------------------------------------------------------------- #
#
# 为什么需要英文版
# ================
# AQUA-7B 由 Mistral-7B-Instruct-v0.3 在英文语料上微调而来（HF tags
# ``language: en``）。让它直接生成中文，观察到的是"英文语序 + 中文词"的
# 夹生输出：词都认识，句子读不通。而 7B 级模型**读**中文、**照抄**中文
# 都没问题，出问题的是**生成**。
#
# 所以这一版的边界划得很清楚：
#   · 指令、解释、正反例 —— 全英文（这是模型要"推理"的部分）；
#   · 枚举值与需要回抄的字段（风险等级 "正常/预警/高危"、告警标题、
#     指标中文名）—— 保持中文，并且在提示词里明说
#     "these are opaque identifiers, copy them verbatim, do not translate"。
# 抄字符串是逐 token 的复制任务，不是生成任务，英文底座做得到。
#
# 回译由 :mod:`fisheries.agent.translate` 负责，**不在这里**：提示词只负责
# "让模型说英文"，把英文变回中文是后处理层的事，两件事必须能分别开关。

INDICATOR_LABELS_EN = {
    "water_temp": "water temperature",
    "ph": "pH",
    "dissolved_oxygen": "dissolved oxygen",
    "ammonia_n": "ammonia nitrogen",
    "nitrite_n": "nitrite nitrogen",
    "nitrate_n": "nitrate",
    "turbidity": "turbidity",
    "population": "stocking density",
}

#: 风险等级的英文注释。**只用于给英文模型解释中文 token 的含义**，
#: 绝不能拿它去替换 token —— 融合层、护栏、接口契约比的都是中文原值。
RISK_LEVEL_GLOSS_EN = {
    "正常": "Normal",
    "预警": "Warning",
    "高危": "Critical",
}

DIRECTION_GLOSS_EN = {
    "上升": "rising",
    "下降": "falling",
    "平稳": "steady",
    "波动": "fluctuating",
}

#: 病害中文名 → 英文名。**只有这五个是工具的闭集**
#: （见 ``fish_disease.DISEASE_LABELS``），所以可以硬编码。
#: 英文底座需要这一层映射：它必须能写出英文的
#: ``disease_actions[].target``（"bacterial gill disease"），
#: 而工具返回的 ``disease_type`` 是中文。
DISEASE_LABELS_EN = {
    "烂鳃": "bacterial gill disease",
    "出血": "bacterial hemorrhagic septicemia",
    "水霉": "saprolegniasis (water mold)",
    "健康": "healthy",
    "其他": "other / unspecified disease",
}

DISEASE_GROUP_LABELS_EN = {
    "细菌性": "bacterial",
    "真菌性": "fungal",
    "病毒性": "viral",
    "寄生虫性": "parasitic",
    "其他": "other",
    "健康": "healthy",
}

#: 融合层 evidence 的 ``unit`` 字段是中文（``小时``），英文路径下要换掉，
#: 否则会出现 ``DO_ALERT_ETA = 14.78小时`` 这种半中半英的取值行。
UNIT_GLOSS_EN = {
    "小时": "hours",
    "分钟": "minutes",
    "天": "days",
    "摄氏度": "C",
    "℃": "C",
    "无": "",
}

#: 溶氧倒计时阈值的中文标签 → 英文。同样只有工具里那几个固定值。
DO_THRESHOLD_LABELS_EN = {
    "危险线": "danger line",
    "泛塘临界值": "pond-turnover (mass mortality) threshold",
    "警戒线": "caution line",
    "适宜下限": "lower end of the optimal range",
}

#: 知识库语料的语言。**这是一个已知的、没有隐藏的短板**：
#: 语料是中文的（``knowledge/corpus/*.md``），英文底座对中文的**阅读**
#: 能力有限，所以走英文路径时 RAG 的增益会低于中文路径。
#: 彻底解决要把 8 篇语料双语化，那是独立工作项——在它完成之前，
#: 这个标记存在的意义是：让模型知道"这段材料是中文参考，不是要你翻译的正文"，
#: 也让读提示词的人一眼看出边界在哪。
KNOWLEDGE_BLOCK_IS_CHINESE = True

DOMAIN_CONSTRAINTS_EN = """\
[DOMAIN]
This system serves **Chinese freshwater pond aquaculture** (cyprinids, tilapia,
shrimp/crab ponds). It is NOT an ornamental aquarium, NOT marine, and NOT a
recirculating indoor farm. Every judgement must match freshwater pond practice.

[REFERENCE RANGES — engineering defaults, always overridable by tool output]
- Water temperature: 20-30 C optimal; below 15 C or above 33 C is dangerous
- pH: 6.5-8.5 optimal; below 6.0 or above 9.0 is dangerous
- Dissolved oxygen (DO): >=5 mg/L optimal; 3-5 mg/L caution; <3 mg/L dangerous;
  <1.5 mg/L is the pond-turnover (mass mortality) threshold
- Ammonia nitrogen: <=0.2 mg/L optimal; 0.2-0.5 caution; >0.5 dangerous
- Nitrite nitrogen: <=0.1 mg/L optimal; 0.1-0.3 caution; >0.3 dangerous

[TWO FACTS PEOPLE GET WRONG]
1. **Ammonia toxicity is amplified by pH and temperature.** The same 0.5 mg/L
   ammonia nitrogen is far more toxic at pH 8.5 / 30 C than at pH 7.0 / 20 C,
   because a much larger share is un-ionised NH3.
2. **DO follows a daily rhythm.** Around sunrise (roughly 02:00-06:00) algae
   stop producing oxygen while respiration continues, so DO bottoms out and
   pond turnover most often happens in that window. Any DO advice must respect it.

[ONE FEEDING RULE YOU MAY NEVER VIOLATE]
**If the disease result is anything other than "健康" (Healthy), you must stop
feeding. Never write "feed the normal amount".**
Sick fish eat less, so the feed simply rots, pushing ammonia up and DO down, and
it makes medicated-feed dosing wrong. The correct wording is
"stop feeding for 1-2 days, then switch to medicated feed".
This rule is **independent of the water numbers**: even if water quality is
perfect, a confirmed disease still means stop feeding.
Only when water is normal AND the fish are healthy may you say "feed normally".
"""

SCHEDULER_SYSTEM_PROMPT_EN = f"""\
You are the **scheduling hub** of "ProjectFisheries", an aquaculture monitoring
and decision agent.

[ABSOLUTE RULES — HIGHEST PRIORITY]
1. You **must not invent any number**. Water temperature, pH, dissolved oxygen,
   ammonia nitrogen and nitrite values, as well as disease type and confidence,
   **may only come from tool results**.
2. Whenever the user supplied the relevant data, you **must call** the tool:
   - water time-series data supplied -> call `water_risk_predict`
   - a fish image path supplied -> call `fish_disease_detect`
3. You **do not grade risk and do not write farming advice**. Grading is done by
   the fusion layer and advice by the advisor role. Your only job is:
   **hand the data to the right tool and bring the result back faithfully.**
4. Answer in English only.

[TOOL SELECTION]
- Mentions of "data / readings / sensor / time-series / CSV / these hours" plus
  actual readings -> call `water_risk_predict`
- Mentions of "image / photo / I took / this fish / the gills" or a file path
  -> call `fish_disease_detect`
- Both present -> **call both tools**; never pick just one
- When a tool returns `status="error"`, report the error code honestly.
  **Do not pretend it succeeded and do not retry with the same arguments.**

[ARGUMENT REQUIREMENTS]
- `readings` must be an array of objects, each shaped like
  `{{"timestamp": "2026-05-11T08:00:00", "water_temp": 28.0, "ph": 7.8,
    "dissolved_oxygen": 4.2, "ammonia_n": 0.35, "nitrite_n": 0.08}}`
- **Do not nest arrays** (`[[28.0, 7.8, 4.2]]` is wrong). Field names must match
  exactly.
- `timestamp` and `nitrite_n` may be omitted when the probe lacks them; the other
  indicators are required.

[LANGUAGE NOTE]
Some tokens you will see are Chinese: the risk level is written
"正常" (Normal) / "预警" (Warning) / "高危" (Critical), and indicator names may
appear in Chinese in tool results. Treat those as **opaque identifiers**: keep
them exactly as given, never translate them.

{DOMAIN_CONSTRAINTS_EN}

[OUTPUT]
When you need a tool, emit the function call directly. Do not describe in prose
what you intend to call. If the data is insufficient to call anything, briefly
state what is missing and list the field names you need.
"""

#: 英文路径下的"能抄就抄、不能抄就留空"契约。中文字段以 opaque token
#: 的身份存在，模型只需复制、不需理解——这正是把生成与复制分开的关键。
LANGUAGE_CONTRACT_EN = """\
[LANGUAGE CONTRACT — READ CAREFULLY]
Write all *generated* text in English: `headline`, `observation_window`,
`confidence_note`, and every `target` / `action` / `timing`.

Two kinds of Chinese will appear in the input. Handle them differently:

1. **Opaque enum tokens — copy verbatim, never translate, never paraphrase.**
   - `risk_level`: exactly one of "正常" / "预警" / "高危"
   - each alert's `title` and `detail`: copy the Chinese string exactly
   - indicator names such as "溶氧" when they appear inside a copied field
   These are identifiers used by the guardrail to verify you did not rewrite the
   conclusion. You do not need to understand them; you need to copy them.

2. **Chinese reference prose** (the "why this level" line, degradation summary,
   tool error messages). Read it as background fact only. Never copy it into the
   generated fields, and never translate it into English yourself.
"""

ADVISOR_SYSTEM_PROMPT_EN = f"""\
You are a freshwater pond aquaculture technician. Your reader is the **farmer
standing at the pond** — not a judge, not a researcher. He may have only middle
school education and is reading your words on a phone at the water's edge.

You do exactly one thing: turn **already computed** conclusions into actions he
can carry out today.

[ABSOLUTE RULES — HIGHEST PRIORITY]
1. **Change no number, not one character.** Risk level, risk score, current
   indicator values, confidence and countdown hours must be copied verbatim.
   You own the wording, **not the arithmetic**.
2. **Add nothing that is not in the material.** Never mention an indicator that
   was not measured or a disease that was not detected. In particular: when an
   indicator was **not measured** this time, never call it "too high".
3. **Risk level and alerts may only be restated, never altered.** Even if you
   think it should be more or less severe, copy it as-is.
4. Write in **English only** (a later stage back-translates your text to Chinese).

[HOW TO WRITE FOR A FARMER]
- Short sentences, one idea each. **Keep every action under 20 English words.**
- Avoid jargon. Do not say "un-ionised ammonia" or "NH3"; say "ammonia toxicity".
  Do not say "diurnal rhythm"; say "after midnight".
- **Do not explain mechanisms, do not give background, do not restate the
  input.** The farmer wants "what do I do", not "why".
- Never write empty phrases like "strengthen management", "please pay
  attention", "should be taken seriously" — that is the same as saying nothing.

[GOOD VS BAD]
BAD  "Please pay attention to water quality management." (says nothing to do)
BAD  "It is recommended to lower ammonia."             (goal without means)
BAD  "DO is low, use an aerator."                      (no when, no how long)
BAD  "Because nighttime algal respiration reduces DO, it is suggested that..."
                                                       (too wordy, no mechanism needed)
GOOD "Turn on the aerator now and keep it on until 8 am tomorrow."
GOOD "Do not feed for two days so the ammonia can come down."
GOOD "DO is 2.6 now; at this rate it hits 1.5 in 6.9 hours, and 1.5 is the
      turnover line."

[HOW MANY ITEMS]
- `water_actions`: 0-3 water-management actions. **When all water indicators are
  normal but the disease result is not "健康" (Healthy), this one item must be
  "stop feeding"** (see the feeding rule above). Never write "feed the normal
  amount" in that case. Only when water is normal AND the fish are healthy may
  you write routine management actions.
- `disease_actions`: 0-2 disease-response actions. If the fish are healthy, or no
  image analysis was done this time -> **output an empty array `[]`**.
- Sort each array by urgency, most urgent first. **Do not split one action into
  two just to fill the list.**

[THE OBSERVATION WINDOW MUST FOLLOW THE SOURCE OF RISK]
`observation_window` must state the re-check that matches **whichever piece of
evidence actually raised the level**:
- Main cause is water (an indicator over threshold, a red line hit, or oxygen
  about to crash) -> re-measure DO, ammonia nitrogen and the like.
- Main cause is disease (water is perfect but a disease is confirmed) -> count
  dead fish and re-inspect whether the lesions spread. Telling the farmer to
  "re-measure DO and ammonia" here answers the wrong question.

[HOW TO WRITE confidence_note — FOLLOW THIS TEMPLATE, DO NOT INVENT A STRUCTURE]
Write `confidence_note` as **three clauses in this exact order**:
  1. Evidence  — which evidence was used this time, with the key numbers;
  2. Uncertainty — which part is least reliable; write "no notable uncertainty"
     if there genuinely is none;
  3. Review — whether a human should double-check; write "no manual review
     needed" if not.
Fixed shape: `Evidence: ... . Uncertainty: ... . Review: ... .`

**Take the content ONLY from the "confidence material" section below. You may
not form your own opinion about whether the conclusion is reliable.** Where you
might write "the conclusion is reliable", the material may be saying "image
analysis failed" or "disease confidence is low" — that is not a wording problem,
it is a safety problem.

[LENGTH CAPS — EXCEEDING THEM IS A FAILURE]
- `headline` <= 15 English words
- each `action` <= 20 English words, `timing` <= 10 English words
- `observation_window` <= 15 words
- `confidence_note` <= 45 English words (three clauses, so it is longer than the
  other fields — that is expected; never drop a clause just to save words)
- the whole JSON <= 170 English words
- A Chinese rendering of your text must also fit 30 / 40 / 90 characters, so stay tight.

{LANGUAGE_CONTRACT_EN}

{DOMAIN_CONSTRAINTS_EN}
"""

ADVISOR_USER_TEMPLATE_EN = """\
Based on the following **already final** monitoring conclusions, write
farmer-facing instructions. Generate all your own text in English.

## Fused decision (computed by the rule engine, **NOT changeable**)
- Overall risk level (Chinese identifier, copy exactly): {risk_level}
- Fused risk score: {fused_score}
- Reason for this level (Chinese, reference only): {reason}

## Water risk (returned by tool water_risk_predict)
{water_block}

## Disease detection (returned by tool fish_disease_detect)
{vision_block}

## Evidence chain
{evidence_block}

## Alert list (**must be echoed verbatim into the output `alerts` array**)
{alert_block}

## Knowledge base reference material (RAG retrieval, grouped by purpose)
{knowledge_block}

[KNOWLEDGE BASE RULES]
- The material above is the basis for **how to word the measures** — it makes an
  action concrete (how much, when, in what order).
- If a **numeric threshold** in the material conflicts with the "Water risk"
  block, **the Water risk block always wins**: that block comes from this
  session's measurements, the material is a general specification.
- When material is marked as not verified against the standard text, you must add
  a sentence noting the value needs checking against the current standard.
- Do not invent measures that are absent from the material, and do not add
  medication schemes from your own general knowledge.

## Confidence material (**write `confidence_note` from THIS — do not invent a structure**)
{confidence_block}

## Output requirement
Output only the single JSON object below. **No other text and no explanation
outside the markdown fence**:

{{
  "risk_level": "copy the overall risk level above exactly: one of 正常 / 预警 / 高危",
  "headline": "one sentence, the most urgent thing right now (<=15 English words)",
  "alerts": [
    {{"level": "copy the level field from the alert list exactly", "title": "copy title exactly (Chinese, do not translate)", "detail": "copy detail exactly (Chinese, do not translate)"}}
  ],
  "water_actions": [
    {{"priority": 1, "target": "which indicator (English)", "action": "what exactly to do (<=20 English words)", "timing": "when and for how long (<=10 English words)"}}
  ],
  "disease_actions": [
    {{"priority": 1, "target": "which disease (English)", "action": "what exactly to do (<=20 English words)", "timing": "when and for how long (<=10 English words)"}}
  ],
  "observation_window": "what to re-check and after how long (<=15 English words)",
  "confidence_note": "three clauses: Evidence: ... . Uncertainty: ... . Review: ... . (<=30 English words)"
}}

Notes:
- Use `[]` for `alerts` when there is no alert, and for `disease_actions` when no
  image analysis was done or the fish are healthy.
- `priority` starts at 1 and is consecutive within each array (1 is most urgent).
- `risk_level` and the alert `title` / `detail` stay Chinese. Everything you
  generate is English.
"""


def scheduler_system_prompt(lang: str = LANG_ZH) -> str:
    """按语言取调度角色的 system 提示词。未知语言一律回落到中文版。"""
    return SCHEDULER_SYSTEM_PROMPT_EN if lang == LANG_EN else SCHEDULER_SYSTEM_PROMPT


def advisor_system_prompt(lang: str = LANG_ZH) -> str:
    """按语言取建议角色的 system 提示词。未知语言一律回落到中文版。"""
    return ADVISOR_SYSTEM_PROMPT_EN if lang == LANG_EN else ADVISOR_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# 四、上下文注入
# --------------------------------------------------------------------------- #

#: MockLLM 与调度层约定的上下文标记。system 消息里以该标记分隔出 JSON 块，
#: 使离线 Mock 也能"看到"用户输入，从而发起真实的工具调用。
CTX_MARKER = "##CTX##"


def build_scheduler_context(ctx: dict[str, Any]) -> str:
    """把结构化上下文附到调度角色 system 消息尾部（供 MockLLM 读取）。

    ``default=str`` 是护栏：这里一旦抛异常，**整个 Agent 循环就断了**，
    而断在一个"某个值序列化不了"上毫无意义。真发生过一次——接口层传来的
    ``timestamp`` 是 ``datetime`` 对象，``json.dumps`` 直接
    ``TypeError: Object of type datetime is not JSON serializable``，
    ``/assess`` 报 500。调用方已经修了，但提示词拼装不该是能崩掉调度的地方。
    """
    import json
    return "\n" + CTX_MARKER + json.dumps(ctx, ensure_ascii=False, default=str)


def _fmt_water_block(water: dict[str, Any] | None, lang: str = LANG_ZH) -> str:
    """渲染水质块。``lang="en"`` 时指标名与趋势词换成英文。

    注意：``degradation_summary`` / ``risk_level`` / ``redline_hits`` 仍然
    保持工具返回的原值。前两个是**必须原样复述的 opaque token**，
    后一个本身是英文代码。翻译它们会直接破坏护栏比对。
    """
    if not water:
        return ("(no water data supplied this time; the tool was not called)"
                if lang == LANG_EN else "（本次未提供水质数据，未调用该工具）")
    if water.get("status") != "ok":
        if lang == LANG_EN:
            return (f"(tool error: {water.get('error_code')} - "
                    f"{water.get('error_message')})")
        return f"（工具报错：{water.get('error_code')} - {water.get('error_message')}）"

    en = lang == LANG_EN
    level = water.get("risk_level")
    if en:
        gloss = RISK_LEVEL_GLOSS_EN.get(str(level), "")
        level_txt = f"{level} ({gloss})" if gloss else str(level)
        lines = [f"- Risk level (Chinese identifier): {level_txt}"
                 f", risk score: {water.get('risk_score')}",
                 f"- Degradation summary (Chinese reference): "
                 f"{water.get('degradation_summary')}"]
    else:
        lines = [f"- 风险等级：{level}，风险分：{water.get('risk_score')}",
                 f"- 恶化预测：{water.get('degradation_summary')}"]

    trends = water.get("trends") or {}
    if trends:
        lines.append("- Current indicators:" if en else "- 当前指标：")
        for key, t in trends.items():
            name = t.get("label", key)
            direction = t.get("direction")
            if en:
                name = INDICATOR_LABELS_EN.get(key, name)
                direction = DIRECTION_GLOSS_EN.get(str(direction), direction)
                lines.append(f"  * {name}: {t.get('current')} {t.get('unit', '')}"
                             f" (trend: {direction})")
            else:
                lines.append(f"  · {name}：{t.get('current')} {t.get('unit', '')}"
                             f"（趋势：{direction}）")

    eta = water.get("projected_do_alert_eta_hours")
    if eta is not None:
        label = str(water.get("projected_do_alert_label") or "").strip()
        if en:
            # 阈值标签是中文（"危险线"），英文路径下换成英文说法，
            # 否则模型读到的是"跌破 3.0 mg/L（危险线）"这种中英混杂的半句。
            label_en = DO_THRESHOLD_LABELS_EN.get(label, label)
            lines.append(f"- DO countdown: expected to fall below "
                         f"{water.get('projected_do_alert_threshold')} mg/L in "
                         f"{eta:.1f} hours"
                         + (f" ({label_en})" if label_en else ""))
        else:
            lines.append(f"- 溶氧倒计时：预计 {eta:.1f} 小时后跌破 "
                         f"{water.get('projected_do_alert_threshold')} mg/L"
                         + (f"（{label}）" if label else ""))
    if water.get("redline_hits"):
        hits = "、".join(water["redline_hits"]) if not en else ", ".join(water["redline_hits"])
        lines.append(f"- Red lines hit: {hits}" if en else f"- 命中红线：{hits}")
    return "\n".join(lines)


def _fmt_vision_block(vision: dict[str, Any] | None, lang: str = LANG_ZH) -> str:
    """渲染病害块。``lang="en"`` 时病害名换成英文，原始中文名括注保留。

    保留中文名是刻意的：``raw_class`` 与 ``note`` 是模型的原始输出，
    译者不该改写；而 ``disease_type`` 括注原文便于人工对照。
    """
    if not vision:
        return ("(no image supplied this time; the tool was not called)"
                if lang == LANG_EN else "（本次未提供图片，未调用该工具）")
    if vision.get("status") != "ok":
        if lang == LANG_EN:
            return (f"(tool error: {vision.get('error_code')} - "
                    f"{vision.get('error_message')})")
        return f"（工具报错：{vision.get('error_code')} - {vision.get('error_message')}）"

    en = lang == LANG_EN
    dtype = vision.get("disease_type")
    group = vision.get("disease_group")
    if en:
        dtype_en = DISEASE_LABELS_EN.get(str(dtype), str(dtype))
        group_en = DISEASE_GROUP_LABELS_EN.get(str(group), str(group))
        lines = [f"- Disease type: {dtype_en} (original Chinese: {dtype})",
                 f"- Disease group: {group_en}",
                 f"- Confidence: {vision.get('confidence')}",
                 f"- Severity weight: {vision.get('severity_weight')}"]
    else:
        lines = [f"- 病害类型：{dtype}（{group}）",
                 f"- 置信度：{vision.get('confidence')}",
                 f"- 严重度权重：{vision.get('severity_weight')}"]
    if vision.get("raw_class"):
        lines.append(f"- Raw model class: {vision.get('raw_class')}" if en
                      else f"- 底层原始判定：{vision.get('raw_class')}")
    if vision.get("is_low_confidence"):
        lines.append("- WARNING: confidence is low, human review required" if en
                     else "- ⚠ 置信度偏低，需人工复核")
    if vision.get("note"):
        lines.append(f"- Note (Chinese): {vision['note']}" if en
                      else f"- 备注：{vision['note']}")
    return "\n".join(lines)


def build_advisor_user_message(fusion: dict[str, Any],
                               water: dict[str, Any] | None,
                               vision: dict[str, Any] | None,
                               knowledge_block: str = "",
                               lang: str = LANG_ZH,
                               confidence_facts: dict[str, Any] | None = None) -> str:
    """渲染建议角色的用户消息。

    ``knowledge_block`` 是 RAG 检索到的措施依据（见
    :meth:`fisheries.rag.retriever.KnowledgeContext.render`）。
    为空时给出显式占位，**不要让模板里出现裸露的 ``{knowledge_block}``**——
    空字符串会让大模型看到"知识库参考资料："后面什么都没有，
    它更可能去编资料。

    ``lang="en"`` 时换用英文模板，但**保留三处中文**：``risk_level``、
    告警的 ``title``/``detail``、以及知识库语料。前两者是护栏要比对的
    opaque token（模型只需复制），后者是语料本身的限制（见
    ``KNOWLEDGE_BLOCK_IS_CHINESE``）。
    """
    en = lang == LANG_EN

    ev = fusion.get("evidence") or []
    if en:
        # 英文路径**不传证据的中文 label**，改传机器码 `code`。
        # 理由是 label 是融合层写的中文句子（"溶氧处于警戒区间"），
        # 而 code（`SEV_DISSOLVED_OXYGEN` / `DO_ALERT_ETA` / `LEVEL_FLOOR` …）
        # 本身就是语言中立的，且信息量足够——具体数值在 water/vision 块里
        # 已经完整给过了，证据链在这里只是让模型知道"哪几条被触发了"。
        # 传中文句子只会挤占上下文，还得指望一个英文底座去读懂它。
        def _ev_line(e: dict[str, Any]) -> str:
            line = f"- [{e.get('source')}] {e.get('code')}"
            if e.get("value") is not None:
                unit = UNIT_GLOSS_EN.get(str(e.get("unit", "")), e.get("unit", ""))
                line += f" = {e.get('value')}{unit}"
            sev = e.get("severity")
            if sev not in (None, 0, 0.0):
                line += f" (severity {sev})"
            return line

        evidence_block = "\n".join(_ev_line(e) for e in ev) or "(none)"
    else:
        evidence_block = "\n".join(
            f"- [{e.get('source')}] {e.get('label')}"
            + (f"（{e.get('value')}{e.get('unit', '')}）"
               if e.get("value") is not None else "")
            for e in ev
        ) or "（无）"

    # 告警块双语都保持 `[level] title：detail` 的原样，模型逐字抄回。
    alerts = fusion.get("alerts") or []
    alert_block = "\n".join(
        f"- [{a.get('level')}] {a.get('title')}"
        + ("：" if not en else ": ")
        + str(a.get("detail"))
        for a in alerts
    ) or ("(no alerts)" if en else "（无告警）")

    if not knowledge_block.strip():
        knowledge_block = (
            "(No knowledge base entry was retrieved this time. Base the measures "
            "only on the measured conclusions above; do not invent sources.)"
            if en else
            "（本次未检索到知识库条目，措施请仅依据上述实测结论给出，不要编造出处。）"
        )
    elif en:
        # 语料是中文的（见 KNOWLEDGE_BLOCK_IS_CHINESE）——明说它是参考材料，
        # 免得模型把中文正文当成"要翻译的内容"或直接抄进英文输出。
        knowledge_block = (
            "[The material below is Chinese source text from the knowledge base. "
            "Treat it as reference for HOW to word the measures; do not translate "
            "it and do not copy it verbatim into your output.]\n"
            + knowledge_block
        )

    # 可信度素材：调用方若能提供（例如已连同知识库上下文一起算过）就用它的，
    # 否则就地按 water/vision/fusion 现算。两条路径共用同一个函数，
    # 所以提示词里给模型看的事实与页面上渲染的事实不可能对不上。
    facts = confidence_facts or build_confidence_facts(water, vision, fusion)

    template = ADVISOR_USER_TEMPLATE_EN if en else ADVISOR_USER_TEMPLATE
    return template.format(
        risk_level=fusion.get("risk_level"),
        fused_score=fusion.get("fused_score"),
        reason=fusion.get("reason"),
        water_block=_fmt_water_block(water, lang),
        vision_block=_fmt_vision_block(vision, lang),
        evidence_block=evidence_block,
        alert_block=alert_block,
        knowledge_block=knowledge_block,
        confidence_block=format_confidence_block(facts, lang),
    )


# --------------------------------------------------------------------------- #
# 三点六、可信度说明：规则算事实，模型照模板措辞
# --------------------------------------------------------------------------- #
#
# 为什么不让模型自由发挥"可靠不可靠"
# ==================================
# 原来的 confidence_note 处在两个极端：规则模板路径写一句写死的话
# （"本建议由规则模板生成，数字全部来自实测，未经模型改写。"），
# LLM 路径则完全由模型自己判断"这份结论可靠不可靠"。
# 前者等于没说；后者等于把"这份结论有多硬"交给一个看不到 trace 的模型——
# 而"可靠性"恰恰是一组**已知事实**：哪几路模态参与了、哪一路失败了、
# 水质点够不够 24 个、病害置信度有没有低于复核线、引用的资料有没有
# 与标准原文核对过、结论里有没有外推。这些都记在 trace 里，不在模型脑子里。
#
# 所以这里把事实算出来、套进固定模板，再让模型照着重写成一句话。
# 模型只负责"说得像人话"，判断仍由规则做——与"定级不交给大模型"
# 是同一条原则的又一次应用。
#
# 三句式是**顺序固定**的：依据 → 不确定 → 复核。顺序不会变，
# 因此前端（以及读报告的人）可以按位置扫读，不必每次重新理解结构。

CONFIDENCE_CLAUSES_ZH = ("依据", "不确定", "复核")
CONFIDENCE_CLAUSES_EN = ("Evidence", "Uncertainty", "Review")

_MODALITY_ZH = {"water": "水质", "vision": "图像"}
_MODALITY_EN = {"water": "water quality", "vision": "image"}

_DISEASE_EN = {
    "烂鳃": "bacterial gill disease",
    "出血": "haemorrhagic septicaemia",
    "水霉": "saprolegniasis",
    "健康": "healthy",
    "其他": "other",
}


def _min_reliable_points() -> int:
    """镜像 ``water_risk.MIN_RELIABLE_POINTS``。

    阈值只有一处真值（在工具层），这里不复制常量而是取用，
    免得两边各自演进而报告口径对不上。工具层不依赖 agent 层，无循环导入风险。
    """
    try:
        from ..tools.water_risk import MIN_RELIABLE_POINTS
    except ImportError:  # pragma: no cover - 仅在包结构被破坏时触发
        return 24
    return int(MIN_RELIABLE_POINTS)


def _as_mapping(value: Any) -> dict[str, Any] | None:
    """把 pydantic 模型或 dict 统一成 dict；其余一律当"没有"。

    这里刻意不 ``import pydantic``：本模块被提示词与规则模板共用，
    用 ``model_dump`` 鸭子类型判断就够了，也免得给提示词模块添一条硬依赖。
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        try:
            dumped = dumper(mode="json")
        except TypeError:  # pragma: no cover - 兼容只接受关键字以外签名的实现
            dumped = dumper()
        return dumped if isinstance(dumped, dict) else None
    return None


def build_confidence_facts(water: dict[str, Any] | None = None,
                           vision: dict[str, Any] | None = None,
                           fusion: dict[str, Any] | None = None,
                           knowledge: Any | None = None) -> dict[str, Any]:
    """把"这份结论有多硬"变成一组可陈述的事实。

    ``knowledge`` 只需鸭子类型地提供 ``chunks``（每项带 ``verified``），
    与 :class:`fisheries.rag.retriever.KnowledgeContext` 一致。
    传 ``None`` 表示本次没有检索（或知识库不可用），此时不写资料相关的不确定项——
    "没检索"和"检索到未核对资料"是两件事，不能混为一谈。
    """
    water = _as_mapping(water)
    vision = _as_mapping(vision)
    fusion = _as_mapping(fusion) or {}

    facts: dict[str, Any] = {
        "risk_level": fusion.get("risk_level"),
        "modalities": list(fusion.get("modalities") or []),
        "failed_modalities": list(fusion.get("failed_modalities") or []),
        "degraded_mode": bool(fusion.get("degraded_mode")),
        "water_ok": bool(water and water.get("status") == "ok"),
        "water_points": None,
        "water_interval_minutes": None,
        "water_enough": None,
        "water_extrapolation_hours": None,
        "water_degraded": bool((water or {}).get("degraded_mode")),
        "vision_ok": bool(vision and vision.get("status") == "ok"),
        "vision_disease": None,
        "vision_confidence": None,
        "vision_low_confidence": False,
        "knowledge_chunks": 0,
        "knowledge_unverified": 0,
    }

    if facts["water_ok"]:
        quality = (water or {}).get("data_quality") or {}
        points = quality.get("n_points")
        facts["water_points"] = points
        facts["water_interval_minutes"] = quality.get("interval_minutes")
        if isinstance(points, int):
            facts["water_enough"] = points >= _min_reliable_points()
        extrapolation = (water or {}).get("extrapolation_horizon_hours")
        facts["water_extrapolation_hours"] = extrapolation

    if facts["vision_ok"]:
        facts["vision_disease"] = (vision or {}).get("disease_type")
        facts["vision_confidence"] = (vision or {}).get("confidence")
        facts["vision_low_confidence"] = bool((vision or {}).get("is_low_confidence"))

    chunks = getattr(knowledge, "chunks", None)
    if chunks:
        facts["knowledge_chunks"] = len(chunks)
        facts["knowledge_unverified"] = sum(
            1 for c in chunks if not getattr(c, "verified", False)
        )
    return facts


def _confidence_parts(facts: dict[str, Any],
                      lang: str = LANG_ZH) -> tuple[str, str, str]:
    """把事实压成三句：依据 / 不确定 / 复核。中英各一套措辞。"""
    en = lang == LANG_EN
    mod_zh = _MODALITY_ZH
    mod_en = _MODALITY_EN

    # ---- ① 依据：用了哪几路证据 ----
    if facts.get("water_ok") and facts.get("vision_ok"):
        disease = facts.get("vision_disease") or ""
        conf = facts.get("vision_confidence")
        if en:
            name = _DISEASE_EN.get(str(disease), str(disease))
            evidence = (f"{facts.get('water_points')} water points + image detected "
                        f"{name} ({conf})")
        else:
            evidence = (f"水质 {facts.get('water_points')} 点 + "
                        f"图片检出{disease}（把握 {conf}）")
    elif facts.get("water_ok"):
        if en:
            evidence = f"{facts.get('water_points')} water points"
        else:
            evidence = f"水质 {facts.get('water_points')} 点"
    elif facts.get("vision_ok"):
        evidence = "image only (no water time series)" if en else "仅鱼体图片（无水质时序）"
    else:
        evidence = "no usable monitoring data" if en else "无可用监测数据"

    # ---- ② 不确定：哪一块最不可靠（最多两条，避免这栏变成流水账）----
    uncertainty: list[str] = []
    for name in facts.get("failed_modalities") or []:
        if en:
            uncertainty.append(f"{mod_en.get(name, name)} analysis failed")
        else:
            uncertainty.append(f"{mod_zh.get(name, name)}分析失败")

    if facts.get("vision_low_confidence"):
        conf = facts.get("vision_confidence")
        if en:
            uncertainty.append(f"disease confidence is low ({conf}); human review needed")
        else:
            uncertainty.append(f"病害置信度偏低（{conf}），需人工复核")

    if facts.get("water_ok") and facts.get("water_enough") is False:
        threshold = _min_reliable_points()
        if en:
            uncertainty.append(f"only {facts.get('water_points')} water points "
                               f"(fewer than {threshold}); the trend is weak")
        else:
            uncertainty.append(f"水质仅 {facts.get('water_points')} 点"
                               f"（少于 {threshold} 点），趋势偏弱")

    extrapolation = facts.get("water_extrapolation_hours")
    if facts.get("water_ok") and extrapolation:
        if en:
            uncertainty.append(f"conclusion extrapolates {extrapolation} hours ahead")
        else:
            uncertainty.append(f"结论含 {extrapolation} 小时外推")

    if facts.get("water_degraded"):
        uncertainty.append("water tool ran in degraded mode" if en
                           else "水质工具处于降级模式")

    unverified = facts.get("knowledge_unverified") or 0
    if unverified:
        total = facts.get("knowledge_chunks") or 0
        if en:
            uncertainty.append(f"{unverified} of {total} references not checked "
                               f"against the standard text")
        else:
            uncertainty.append(f"引用 {total} 条资料中 {unverified} 条"
                               f"未与标准原文核对")

    if not uncertainty:
        uncertainty_text = "no notable uncertainty" if en else "无明显不确定项"
    else:
        head = uncertainty[:2]
        suffix = ""
        if len(uncertainty) > 2:
            # 中文的"等"紧贴前文（"…趋势偏弱等"），英文的 "etc." 才需要空格。
            suffix = " etc." if en else "等"
        uncertainty_text = ("; ".join(head) if en else "；".join(head)) + suffix

    # ---- ③ 复核：要不要人再看一遍 ----
    if facts.get("vision_low_confidence"):
        review = "a human should re-check the photo" if en else "建议人工复看图片"
    elif "vision" in (facts.get("failed_modalities") or []):
        review = "a human should re-run the image analysis" if en else "建议人工重做图像分析"
    elif facts.get("water_degraded") or "water" in (facts.get("failed_modalities") or []):
        review = ("a human should check the raw water data" if en
                  else "建议人工核对水质原始数据")
    elif facts.get("water_ok") and extrapolation:
        review = "re-measure the water soon" if en else "建议尽快补测一次水质"
    else:
        review = "no manual review needed" if en else "无需人工复核"

    return evidence, uncertainty_text, review


def format_confidence_block(facts: dict[str, Any], lang: str = LANG_ZH) -> str:
    """渲染提示词里的"可信度素材"块——**已填好事实的模板**。

    故意写成"模板 + 已填内容"而不是一段成品句子：模型看到的是
    "这三栏各自该放什么"，它要做的是把三栏连成一句，而不是自己决定
    该说哪三件事。这也是"照模板生成"的字面含义。
    """
    evidence, uncertainty, review = _confidence_parts(facts, lang)
    head, mid, tail = (CONFIDENCE_CLAUSES_EN if lang == LANG_EN
                       else CONFIDENCE_CLAUSES_ZH)
    sep = ": " if lang == LANG_EN else "："
    lines = [
        f"- {head}{sep}{evidence}",
        f"- {mid}{sep}{uncertainty}",
        f"- {tail}{sep}{review}",
    ]
    if lang == LANG_EN:
        shape = f"Fixed shape: {head}: ... . {mid}: ... . {tail}: ... ."
    else:
        shape = f"句式固定为：{head}：…。{mid}：…。{tail}：…。"
    return "\n".join(lines) + "\n" + shape


def _end_sentence(text: str, period: str) -> str:
    """补句号，但**不补第二个**。

    "不确定"那一栏可能是以 "etc." 结尾的枚举，再补一个点就成了 "etc.."；
    中文侧同理不能出现 "。。"。
    """
    if text.endswith((".", "。", "！", "!", "？", "?")):
        return text
    return text + period


def render_confidence_note(facts: dict[str, Any], lang: str = LANG_ZH) -> str:
    """把同一组事实渲染成**成品句子**，用于规则模板路径与回译兜底。

    与 :func:`format_confidence_block` 同源，因此"没接大模型时"和
    "接了大模型但输出不可用"两条路径写出来的可信度说明**结构一致**，
    只有措辞丰富度不同。反过来，如果两边各写各的，就会出现
    "页面上的可信度说明和提示词里给模型的事实对不上"这种很难查的问题。
    """
    evidence, uncertainty, review = _confidence_parts(facts, lang)
    period = "." if lang == LANG_EN else "。"
    if lang == LANG_EN:
        return (f"Evidence: {_end_sentence(evidence, period)} "
                f"Uncertainty: {_end_sentence(uncertainty, period)} "
                f"Review: {_end_sentence(review, period)}")
    return (f"依据：{_end_sentence(evidence, period)}"
            f"不确定：{_end_sentence(uncertainty, period)}"
            f"复核：{_end_sentence(review, period)}")


__all__ = [
    "ADVISOR_SYSTEM_PROMPT",
    "ADVISOR_SYSTEM_PROMPT_EN",
    "ADVISOR_USER_TEMPLATE",
    "ADVISOR_USER_TEMPLATE_EN",
    "CONFIDENCE_CLAUSES_EN",
    "CONFIDENCE_CLAUSES_ZH",
    "CTX_MARKER",
    "DIRECTION_GLOSS_EN",
    "DISEASE_GROUP_LABELS_EN",
    "DISEASE_LABELS_EN",
    "DOMAIN_CONSTRAINTS",
    "DOMAIN_CONSTRAINTS_EN",
    "DO_THRESHOLD_LABELS_EN",
    "INDICATOR_LABELS_EN",
    "KNOWLEDGE_BLOCK_IS_CHINESE",
    "LANGUAGE_CONTRACT_EN",
    "RISK_LEVEL_GLOSS_EN",
    "SCHEDULER_SYSTEM_PROMPT",
    "SCHEDULER_SYSTEM_PROMPT_EN",
    "UNIT_GLOSS_EN",
    "advisor_system_prompt",
    "build_advisor_user_message",
    "build_confidence_facts",
    "build_scheduler_context",
    "format_confidence_block",
    "render_confidence_note",
    "scheduler_system_prompt",
]
