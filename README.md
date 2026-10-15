# ProjectFisheries —— 水产养殖智能监测与决策 Agent

面向**中国淡水池塘养殖**的智能监测与决策智能体。接收两类输入：

- **① 时序传感器数据**：水温、pH、溶氧、氨氮、亚硝酸盐
- **② 鱼体图片**：识别烂鳃、细菌性出血、水霉等病害

经 **数据清洗 → 水质时序风险预测 → 图像识别 → 多源融合 → 风险定级与调控建议** 全链路，
输出风险等级、结构化告警、证据链与可落地的养殖建议。

---

## 1. 快速开始（零依赖、离线可跑）

```powershell
# 1) 建虚拟环境（只需一次）
cd D:\设置\项目\Agent\project
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pydantic pytest ^
    langchain-core langchain-chroma chromadb fastembed rank-bm25 jieba ^
    fastapi python-multipart

# 2) 跑 Agent 全链路 demo（6 个场景，含 RAG 检索）
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe scripts\demo_agent.py

# 3) 跑 RAG 检索 demo（4 个场景，直观看检索质量）
.\.venv\Scripts\python.exe scripts\demo_rag.py

# 4) 跑提示词与护栏 demo（3 个场景：系统提示词 / 注入的用户消息 / 听话 vs 不听话的模型）
.\.venv\Scripts\python.exe scripts\demo_prompt.py

# 5) 跑后端接口 demo（6 个场景，用 TestClient 进程内跑完整 HTTP 链路，不用起服务）
.\.venv\Scripts\python.exe scripts\demo_api.py

# 6) 跑 D12 回译 demo（4 个场景：语言判定表 / 英文提示词逐条自检 / 英文 JSON→中文报告 / 四种回译后端）
.\.venv\Scripts\python.exe scripts\demo_translate.py

# 7) 跑 Web 演示页离线 demo（不开浏览器，把页面上每一栏的内容原样打印出来）
.\.venv\Scripts\python.exe scripts\demo_webui.py
# 加 --app 会额外构建一次 Gradio 界面，验证组件树能装配（需要先装 gradio）

# 8) 重建知识库（改了 knowledge/corpus/ 里的语料之后必须跑）
.\.venv\Scripts\python.exe scripts\build_knowledge_base.py

# 9) 启动后端接口（FastAPI，含 Swagger 文档）
.\.venv\Scripts\python.exe -m uvicorn fisheries.api.app:app --host 127.0.0.1 --port 8812
# 然后打开 http://127.0.0.1:8812/docs 试接口

# 10) 启动 Web 演示页（Gradio，给现场答辩用；需先装可选依赖）
.\.venv\Scripts\python.exe -m pip install -e ".[webui]"
.\.venv\Scripts\python.exe -m fisheries.webui.app
# 然后打开 http://127.0.0.1:7860 上传表格 + 图片，点「运行 Agent」

# 11) （可选）下本地离线翻译模型，让 D12 回译层完全离线可用（约 315 MB，见 §2.5）
.\.venv\Scripts\python.exe -m pip install -e ".[translate]"
.\.venv\Scripts\python.exe scripts\fetch_translate_model.py

# 12) 跑 M8 外部验证，重新生成评测报告（约 25 秒；要读 11 个 CSV / 111 万行）
.\.venv\Scripts\python.exe scripts\evaluate_water_rules.py
# 报告写到 docs\M8_评测报告.md；指标 JSON 写到 docs\_m8_metrics.json
# 换自己的数据：--data-dir <目录>；想先试跑：--limit-files 2 --limit-windows 300

# 13) （可选）训练病害识别模型，产出 models\yolov8-fish.pt
#     需要先装训练栈，见 docs\模型训练说明.md
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install ultralytics
.\.venv\Scripts\python.exe scripts\train_yolo_fish.py --report   # 先体检，不训练
.\.venv\Scripts\python.exe scripts\train_yolo_fish.py --epochs 60

# 14) 跑单元测试（939 项）
.\.venv\Scripts\python.exe -m pytest
```

**数据集是可选的，工程里一份都没带。** 上面的 demo / 测试 / 演示页在**没有数据集**的机器上
也能跑：缺数据的用例会 `skip`（不是失败），脚本会打印一行「该数据集未导入」而不是崩掉。
要接上数据，先看这一行：

```powershell
# 看注册表里有哪些数据集、本机已经有了哪些（默认动作）
.\.venv\Scripts\python.exe scripts\fetch_datasets.py --status
# 缺的按名字下载（需要联网 + pip install -e ".[datasets]"）
.\.venv\Scripts\python.exe scripts\fetch_datasets.py --fetch aquaponics_pond
```

历史目录 `D:\设置\项目\Agent\数据集` 会被自动认到，**不用改任何路径**。细则见 §12。

**不需要 API key，不需要联网。** 未配置大模型时，Agent 自动走**离线确定性 MockLLM + 规则模板建议**，
整条链路（LLM 决策工具 → 工具执行 → 护栏校验 → 融合定级 → RAG 定向检索 → 建议生成）
与接入真实模型时完全一致。
中文向量模型（`bge-small-zh-v1.5`，90.8 MB）已内置在 `models/fastembed/`，
**断网、无外网、比赛现场都能直接跑**。

**病害识别模型也是可选的。** 本机训练好的权重 `models/yolov8-fish.pt`
（YOLOv8n-cls 四分类，测试集 Top-1 93.0%）**同样不随仓库分发**：
权重缺失时 `fish_disease_detect` 自动落回 mock 兜底并在响应里标
`degraded_mode=true`，`/api/v1/health` 也会如实申报 `mock`。
训练与复现步骤见 [`docs/模型训练说明.md`](docs/模型训练说明.md)。

当前状态：`939 passed`（工具层 83 + Agent 层 114 + 回译层 144 + 评测层 181 + RAG 层 100 + 接口层 111 + 演示页 69 + 数据集层 75 + 控制台层 27 + 启动器层 35）。

> ⚠ 只跑工具层与 Agent 层、不想装 RAG 依赖时：设 `FISHERIES_DISABLE_KNOWLEDGE=1`，
> Agent 会跳过检索并降级为「无依据建议」，不会启动失败。

---

## 2. 接入真实大模型（代码一行不用改）

### 2.1 为什么是"双角色"架构

实测 `KurmaAI/AQUA-7B`（首个水产领域大模型，Apache-2.0）有两个短板：

| 短板 | 影响 |
|---|---|
| 训练数据 `language: en`，纯英文 | 中文养殖建议质量无保证 |
| 微调数据是水产问答，**无工具调用训练** | 能否稳定走 function calling 未经验证 |

而"决定调哪个工具"恰恰是整条链路最不能出错的一步。因此本项目把 LLM 拆成两个角色：

- **调度角色（scheduler）**：必须可靠地输出 function call。
  推荐 **DeepSeek-V3 / Qwen** 等中文工具调用成熟的模型。
- **建议角色（advisor）**：把已算好的结构化结论翻译成专业养殖建议。
  这里才是 **AQUA-7B** 的用武之地——它的水产领域知识是真的。

两个角色可以用**不同的模型**，互不影响。

### 2.2 配置方式

```powershell
# 调度角色：DeepSeek（可靠的中文 function calling）
$env:FISHERIES_LLM_SCHEDULER_PROVIDER = "deepseek"
$env:FISHERIES_LLM_SCHEDULER_API_KEY  = "sk-xxxxxxxx"

# 建议角色：本地 AQUA-7B（Ollama，水产领域知识）
$env:FISHERIES_LLM_ADVISOR_PROVIDER   = "aqua-7b-ollama"
$env:FISHERIES_LLM_ADVISOR_BASE_URL   = "http://localhost:11434/v1"
$env:FISHERIES_LLM_ADVISOR_MODEL      = "aqua-7b"
```

内置预置后端（`src/fisheries/agent/llm.py` 的 `PROVIDER_PRESETS`）：

| provider | 说明 |
|---|---|
| `mock` | 离线确定性 Mock（默认） |
| `aqua-7b-ollama` | 本地 Ollama 跑 AQUA-7B 量化版 |
| `aqua-7b-vllm` | 本地 vLLM 跑 AQUA-7B |
| `deepseek` | DeepSeek 官方 API |
| `openai` | OpenAI 官方 API |
| `dashscope` | 阿里云百炼（Qwen） |

### 2.3 关于 AQUA-7B 的部署提醒

`KurmaAI/AQUA-7B` 是 7B 模型，**fp16 需要约 15GB 显存**。
本机 RTX 5070 Laptop 只有 **8GB**，必须用量化版本：

```powershell
# 社区已有现成 GGUF 量化（约 4.4GB，8GB 显存可跑）
ollama pull hf.co/DevQuasar/KurmaAI.AQUA-7B-GGUF
```

> ⚠ 另注意 RTX 50 系是 Blackwell 架构（sm_120），需要 CUDA 12.8+ 与较新的 PyTorch。

### 2.4 AQUA-7B 只会英文，但报告必须是中文——回译层（D12）

`KurmaAI/AQUA-7B` 是 `Mistral-7B-Instruct-v0.3` 的**英文微调**（HuggingFace 标签
`language: en`）。要它直接写中文，质量没有保证；可竞赛与养殖户要的都是中文。

本项目的做法：**让它用母语（英文）思考，再用一个中文模型把结果整句回译。**

```
水质/图像数据 ──► 英文提示词 ──► AQUA-7B 用英文写建议 ──► 回译层 ──► 中文报告
                  (prompts.py)        (risk_level 仍要求照抄中文枚举)   (translate.py)
```

**语言是自动选的**，也可以强制：

| 后端 | 提示词语言 | 理由 |
|---|---|---|
| `aqua-7b-ollama` / `aqua-7b-vllm` | **英文** | 英文底座，让它用母语思考 |
| `deepseek` / `dashscope` / `openai` / `mock` | 中文 | 中文能力够，省一次回译 |

```powershell
# 覆盖自动判断（一般不需要）
$env:FISHERIES_PROMPT_LANG = "en"          # zh / en

# 可选：远端翻译模型。不配就复用「调度角色」的模型（推荐部署里它就是 DeepSeek/Qwen）
$env:FISHERIES_TRANSLATOR_PROVIDER = "deepseek"
$env:FISHERIES_TRANSLATOR_API_KEY  = "sk-xxxxxxxx"
$env:FISHERIES_TRANSLATOR_MODE     = "auto"   # auto / local / glossary / llm / bypass
```

**推荐做法：用本地离线模型，不配任何 key。** 权重不在 PyPI 上，要单独下（约 315 MB）：

```powershell
pip install -e ".[translate]"                          # transformers + sentencepiece + sacremoses
.\.venv\Scripts\python.exe scripts\fetch_translate_model.py     # 下到 models/translate/opus-mt-en-zh
.\.venv\Scripts\python.exe scripts\fetch_translate_model.py --list   # 看还有哪些可选（含 Qwen2.5-0.5B-Instruct）
```

下完以后 `FISHERIES_TRANSLATOR_MODE=auto`（默认）会**优先用本地模型**，完全离线。
想换成别的位置或别的模型，用这两个环境变量：

```powershell
$env:FISHERIES_TRANSLATOR_LOCAL_DIR   = "D:\models\translate"   # 模型根目录
$env:FISHERIES_TRANSLATOR_LOCAL_MODEL = "opus-mt-en-zh"         # 目录名或绝对路径
```

回译层的三条设计约束（都能在 `tests/unit/test_translate.py` 里看到断言）：

1. **只碰"话怎么说"，碰不到"风险几级"。** `risk_level` 与 `alerts` 一个字都不
   过翻译——它们是护栏要比对的标识符与融合层的确定性结果。
2. **译不出来就退回规则模板的中文表述，绝不保留英文。** 半英半中的报告比少一句
   建议更糟。每一次退回都写进 `degraded_reasons` 并记进推理链路。
3. **回译模型挂掉不能拖垮整条链路。** 调用失败只上报原因、退回模板，
   不抛异常。

四级回译后端，按可用性自动降级：

| 后端 | 何时启用 | 能力 |
|---|---|---|
| `local` | **有本地模型时优先**（离线、零 key、确定性） | 整句翻译。分块 + 术语后订正，见下 |
| `llm` | 没本地模型但配了 `FISHERIES_TRANSLATOR_*`，或调度角色有可用模型 | 整句翻译，质量最好 |
| `glossary` | **兜底**（离线、零依赖、确定性） | 短句与指标名能直译；超过 6 个英文词的自由长句一律拒绝并退回模板 |
| `bypass` | 显式指定 | 不回译，保留英文——**只在调提示词时用**，用来把"提示词写得对不对"和"回译准不准"分开排查 |

### 2.5 本地翻译模型为什么还要"分块 + 后订正"

通用 NMT 不懂水产。这两步都是实测逼出来的，不是设计洁癖：

**分块**——模型一次吞下整段英文会**把尾部整段丢掉**。拿
`"Turn on the aerator tonight and measure dissolved oxygen and ammonia nitrogen again tomorrow morning at 6."`
（106 字符 / 17 词）当标尺，预算放到 200 字符 / 40 词时它照样丢掉 `at 6`，
**连数字都没了**，数值校验只能整条放弃。逐档收紧后测到：**词预算是那条线**，
17 词的句子只要不切就必丢尾部，切到 16 词以内（`DEFAULT_CHUNK_CHARS = 110` /
`DEFAULT_CHUNK_WORDS = 16`）数字就保住了。切分不会切在数字与单位之间，
`1.5 mg/L` 是一个整体。

**后订正**——本地模型语序对、术语全错：`turbidity` → "多发性"、
`bacterial gill disease` → "细菌刺痛病"、`aerator` → "呼吸器"、
`stocking density` → "丝袜密度"、`NTU` → "纽芬兰先令"。词表后端的失败正好相反
（词认得、语序拼不出），所以两者互补：**让 NMT 负责语序、让词表负责术语**。
NMT 先出通顺中文，再用一张中文→中文的后订正表（`GLOSSARY_POST_ZH`）把术语扳回来，
最后把单位规范化回 `mg/L` / `℃`。这张表只收"本领域里不可能有第二种解释"的错法——
`措施`、`发病率`、`中度` 都**故意不收**，它们在别的句子里是正当中文。

两件事的完整实测数据、复现命令与已知限制见
[`docs/M10_回译层说明.md`](docs/M10_回译层说明.md)。

> 离线看这套机制实际怎么跑：`scripts\demo_translate.py`（4 个场景：语言判定表 /
> 英文系统提示词逐条自检 / 英文 JSON → 中文报告逐字段对照 / 各回译后端对照）。

---

## 3. 工程结构

```
project/
├── pyproject.toml
├── README.md
├── 使用说明.txt             # 面向非技术使用者（★ 双击 .bat 之前先看这个）
├── run-demo.bat            # 实际逻辑。三个 run-*.bat 全是纯 ASCII（见 §9 启动器层）
├── run-api.bat
├── run-selftest.bat
├── 启动演示页.bat           # ★ 最常用：双击它打开 Web 演示页（只负责转发到 run-demo.bat）
├── 启动接口服务.bat         # 双击它起 HTTP 服务给前端联调（转发到 run-api.bat）
├── 运行自检.bat             # 双击它做环境体检（转发到 run-selftest.bat）
├── docs/
│   ├── 技术方案设计.md      # 完整技术方案（18 节）
│   ├── 数据集说明.md        # 数据集盘点、质量缺陷清单、许可证
│   ├── 模型训练说明.md      # 病害识别模型：环境、数据、训练、指标与复现
│   ├── M8_评测报告.md       # 两组外部验证报告（自动生成，勿手改）
│   ├── M10_回译层说明.md    # 回译层：模型下载、分块预算实测、后订正表、已知限制
│   └── 前端对接说明.md      # 交给前端同学重做页面时看的接口/功能/建议
├── src/fisheries/
│   ├── datasets.py          # 「数据集名字 → 磁盘位置」的唯一真相来源（只用标准库，见 §12.1）
│   ├── console.py           # 控制台编码兜底：只放宽 errors、绝不改编码（见 §9 控制台层）
│   ├── launcher.py          # 启动器要显示的全部中文（.bat 是纯 ASCII，中文只能从这里出，见 §9）
│   ├── tools/               # 工具层：给大模型调用的"手"
│   │   ├── definitions.py   #   function-calling schema（纯契约，无业务逻辑）
│   │   ├── water_risk.py    #   工具① water_risk_predict
│   │   └── fish_disease.py  #   工具② fish_disease_detect
│   └── agent/               # Agent 层：调度与决策的"脑"
│       ├── llm.py           #   OpenAI 兼容客户端 + 离线 Mock（零依赖）
│       ├── prompts.py       #   调度角色 / 建议角色提示词（中文 + 英文两套）
│       ├── translate.py     #   回译层：英文建议 → 中文报告（D12，四种后端，见 §2.5）
│       ├── fusion.py        #   M4 多源融合（确定性规则）
│       └── orchestrator.py  #   M6 主调度循环 + 护栏 + RAG 接入 + 回译接线
│   └── rag/                 # M5 混合检索知识库
│       ├── loader.py        #   front-matter 解析 + 语料发现
│       ├── splitter.py      #   标题感知分块 + 表格整体保留 + 整节粘合
│       ├── embedder.py      #   fastembed 封装（离线缓存）+ 单测用 HashEmbeddings
│       ├── store.py         #   Chroma 持久化 + 清单一致性守卫
│       └── retriever.py     #   稠密 + BM25 + RRF + 按融合结论定向检索
│   └── api/                 # M7 后端接口（FastAPI）
│       ├── schemas.py       #   请求/响应契约（_Strict 基类禁止多余字段）
│       ├── deps.py          #   应用状态、配置、内存态会话缓冲
│       ├── routes.py        #   10 个端点
│       └── app.py           #   应用装配 + 中文错误页 + OpenAPI 描述
│   └── webui/               # M9 现场演示页（Gradio，可选依赖）
│       ├── table.py         #   表格接入：列名识别 / 时间轴 / 缺失与量纲核查
│       ├── runner.py        #   WebSession：跑 Agent 并摊平成界面好取用的结构
│       ├── render.py        #   把结果渲染成中文 Markdown / HTML（不含业务判断）
│       └── app.py           #   Gradio 界面装配 + 命令行入口（★ 唯一 import gradio 的文件）
│   ├── evaluation/          # M8 外部验证（只做评测，不参与线上链路）
│   │   ├── ponds.py         #   读 11 个 IoTPond*.csv：列名别名 / 单位换算 / 量程清洗 / 滑窗
│   │   ├── labels.py        #   按行业阈值现算 正常·预警·高危（最差指标决定论）+ 生长自洽性
│   │   ├── metrics.py       #   混淆矩阵 / 宏平均 P-R-F1 / Cohen's κ / Spearman
│   │   ├── runner.py        #   批量调规则引擎，把预测与真值配成对
│   │   ├── do_trend.py      #   溶氧趋势外推 vs 持久性基线（真值独立的那一组）
│   │   └── report.py        #   渲染成 Markdown 报告（含第 8 节图像模态）
├── examples/
│   └── pond_demo_water.csv  # 演示用 24 小时水质读数（页面「一键载入示例」用它）
├── data/                    # 运行期产物；datasets/ 与 dataset_index.json 是可选的数据集落点
│   ├── datasets/            #   想跟工程放一起就放这（默认空，工程不自带数据集，见 §12.1）
│   ├── dataset_index.json   #   手工「名字 → 路径」映射（可选）
│   └── derived/             #   从原始数据集派生的训练/评测用数据（可重建，见 §12.2）
│       ├── fish_disease_4cls/      #   四分类图集：train 559 / val 186 / test 186
│       ├── fish_disease_4cls_det/  #   弱监督检测布局（每图一个全图框）
│       ├── runs/            #   ultralytics 训练产物（权重曲线、混淆矩阵）
│       └── metrics.json     #   留出集指标（评测报告第 8 节读它）
├── knowledge/
│   ├── README.md            # 语料可信度声明与组织约定（★ 先读这个）
│   ├── corpus/              # 9 篇种子语料（水质阈值/三种病害/投料/调控/应急/标准依据/病害处置规程）
│   └── chroma/              # 向量库产物（可重建，不入版本库）
├── models/
│   ├── fastembed/           # bge-small-zh-v1.5 ONNX 缓存（90.8 MB，断网可跑）
│   ├── translate/           # 本地离线翻译模型（opus-mt-en-zh，约 315 MB，见 §2.5 / docs/M10）
│   ├── pretrained/          # yolov8n-cls.pt 预训练骨干（训练脚本自动下载，可重建）
│   └── yolov8-fish.pt       # ★ 训练产出的四分类病害权重（约 2.8 MB，不随仓库分发）
├── scripts/
│   ├── demo_tools.py        # 两个工具的独立演示（6 场景）
│   ├── demo_agent.py        # Agent 全链路演示（6 场景）
│   ├── demo_rag.py          # RAG 检索演示（4 场景，含注入提示词原文）
│   ├── demo_prompt.py       # 提示词与护栏演示（3 场景：系统提示词 / 注入消息 / 听话 vs 不听话）
│   ├── demo_translate.py    # D12 回译演示（4 场景：语言判定 / 英文提示词自检 / 逐字段回译 / 四种后端）
│   ├── demo_api.py          # 后端接口演示（6 场景，TestClient 进程内跑完整 HTTP 链路）
│   ├── demo_webui.py        # Web 演示页离线演示（不开浏览器，把页面上每栏原样打印）
│   ├── evaluate_water_rules.py  # M8 外部验证 CLI（重跑即重新生成评测报告）
│   ├── build_knowledge_base.py  # 官方建库 CLI（支持 --check / --dry-run）
│   ├── fetch_datasets.py    # 数据集 CLI（--list/--status/--where/--fetch，见 §12.1）
│   ├── fetch_translate_model.py # 下本地翻译模型到 models/translate/（--list/--model/--force，见 §2.5）
│   ├── build_yolo_dataset.py    # 从三份原始图集去重、切分，产出训练用四分类图集
│   ├── train_yolo_fish.py   # 训练 + 留出集评估 CLI（--report 只体检），产出 models/yolov8-fish.pt
│   ├── selftest.py          # 「运行自检.bat」的 4 步体检本体（中文全在这里，.bat 只转发）
│   └── probes/              # 一次性环境探测脚本（非交付物，保留备查）
└── tests/unit/
    ├── test_tools.py        # 83 项
    ├── test_agent.py        # 114 项
    ├── test_translate.py    # 144 项（D12 回译层：语言判定 / 词表 / 逐字段回译 / 端到端）
    ├── test_evaluation.py   # 181 项（M8：评测代码自身的正确性）
    ├── test_rag.py          # 100 项（用 HashEmbeddings，不加载 90 MB 模型）
    ├── test_api.py          # 79 项（TestClient，不启服务）
    ├── test_api_rag.py      # 25 项（接口层 × 真实知识库的一致性）
    ├── test_webui.py        # 69 项（表格接入层；不 import gradio）
    ├── test_datasets.py     # 75 项（数据集注册表 / 五级解析 / 绝对路径回归守卫）
    ├── test_console.py      # 27 项（编码兜底 / 符号降级 / 编码锁 / 旧写法回归守卫）
    └── test_launchers.py    # 35 项（启动器：.bat 纯 ASCII 守卫 / 转发 / 横幅文案）
```

---

## 4. 架构与关键设计决策

### 4.1 数据流

```
用户输入（水质时序 + 鱼体图片）
    │
    ├─► ① 调度角色 LLM：看数据 → 决定调哪些工具 → 发起 function call
    │        │
    │        └─► TOOL_HANDLERS[name](**arguments)
    │                 ├─ water_risk_predict   → 风险等级 / 恶化描述 / 倒计时
    │                 └─ fish_disease_detect  → 病害类型 / 置信度 / 严重度
    │
    ├─► ② 结果回灌 LLM → 它再判断"还要调别的吗"
    │
    ├─► ③ 护栏校验：该调的工具漏调了吗？（漏了就补调，并留痕）
    │
    ├─► ④ 融合层（确定性规则）→ 风险等级 + 告警 + 证据链
    │
    ├─► ⑤ RAG 定向检索：按融合结论决定查哪几类资料（应急/水质/病害/投料）
    │        知识库不可用时降级并留痕，不让整条链路失败
    │
    └─► ⑥ 建议角色 LLM → 可落地养殖建议（数字只能引用，不能改）
              │
              └─► 输出：风险等级、告警、建议、证据链、**引用出处**、完整推理轨迹
```

### 4.2 五条核心设计决策

**决策一：融合决策绝不交给大模型。**

风险等级、融合分、告警全部由 `fusion.py` 的规则算出。原因：

1. **可复现**——同一份数据必须永远得出同一个等级，否则养殖户无法信任告警；
2. **可审计**——被问"为什么判高危"时能给出规则 ID 和数值依据，而不是"模型觉得"；
3. **可测试**——只有确定性逻辑才能写单测钉住边界。

大模型只负责**措辞**，不得改动**数字**。这条写进了建议角色的提示词铁律。

**决策二：加"护栏"（guardrail）兜住 LLM 漏调工具。**

纯 LLM 驱动的工具选择有个致命失败模式：**该调工具却没调**。
7B 级模型尤其容易漏调第二个工具。一旦漏调，下游融合就少一个模态，
结论会**静默地**变得片面，而调用方完全看不出来。

所以调度层在 LLM 循环结束后做一次必调工具覆盖检查：输入里提供了数据、
但 LLM 没调的工具，由调度层直接补调，并在 `trace` 里记为 `guardrail`，**不隐瞒**。
模型靠谱时它是冗余的，模型抽风时它救命。

**决策三：模态失败必须显式上报。**

用户传了照片、图片工具却报错时，绝不能静默把图像模态丢掉、
再在理由里写"本次仅有水质单模态输入"——那会让用户以为照片被分析了。
`FusionResult.failed_modalities` 与 `MODALITY_UNAVAILABLE` 告警负责这件事。

**决策四：不引入 LangGraph / LlamaIndex。**

OpenAI 兼容的 `/chat/completions` 是 Ollama、vLLM、DeepSeek、OpenAI、DashScope
的公共接口。自己写一层薄客户端（`llm.py`，仅标准库 `urllib`，**零依赖**），
换模型只是改一个环境变量；引入框架反而把"换模型"绑死在框架的适配层上。
本项目链路是线性的四步，LangGraph 的价值在复杂多智能体状态机上，这里纯属依赖风险。

**决策五：RAG 只决定「措施怎么写」，绝不参与定级；知识库里的阈值与实测冲突时一律以实测为准。**

风险等级在第 ④ 步就由规则钉死了，检索发生在定级**之后**。这样做有两个后果，
都是刻意要的：① 知识库缺失、损坏、未构建时，告警依然准确，系统只是"说不出依据"，
而不是"算不出结论"；② 大模型不会因为检索到某条"通用标准说氨氮 0.5 才危险"
就把本次实测的 0.62 mg/L 改口成"正常"。

检索本身也是**定向**的，不是每次都把同一句 query 查一遍：

| # | 用途 | 触发条件 |
|---|---|---|
| ① | 应急处置 | **仅在高危或命中红线时检索**——预案全是「立即/紧急」祈使句，正常状态下混入会让模型把日常巡检写成紧急告警 |
| ② | 水质调控 | 永远检索，带上主控风险因子 |
| ③ | 病害处置 | 检出病害时检索，用 `disease` 硬过滤病种 |
| ④ | 投料管理 | 有病害时**也要**检索——「停料几天、怎么恢复」只有投料规范里写了 |

双路召回（Chroma 稠密 + BM25 稀疏）用 **RRF 在排名层融合**，而不是把分数归一化后相加：
余弦相似度在 0–1 有界，BM25 无上界，两者量纲不可比；加权相加需要为每个语料库调参，
RRF 只用排名，换语料不用重调。

> **语料可信度的坦白**：数据集里**没有任何规范文档**（实测 28,313 张图 + 17 份 CSV + 0 份 PDF/MD），
> 所以 `knowledge/corpus/` 是**自编的种子语料**而非标准原文。
> 2026-10-15 已用国家标准全文公开系统与生态环境部公开的 GB 11607-1989 官方扫描原文
> 逐条核验引用编号与水质数值，9 篇全部 `verified: true`，检索结果的提示块会带上
> 「依据 GB 11607-1989；核验日期 2026-10-15」。核验覆盖的只是「引用的标准编号与数值是否正确」，
> 工程分级（警戒线/危险线）仍是本项目自定的。
> 详见 [knowledge/README.md](knowledge/README.md)。

### 4.3 多源协同规则（融合的核心增量）

单看水质不严重、单看鱼病也不严重，但**同时出现就危险**——这正是多源融合的价值：

| 规则 ID | 触发条件 | 加成 |
|---|---|---|
| `SYN_BACTERIAL_ON_WARNING` | 水质 ≥ 预警 ∧ 检出细菌性病害 | +12 |
| `SYN_NITROGEN_WITH_DISEASE` | 氨氮或亚硝酸盐超警戒 ∧ 检出任何病害 | +8 |
| `SYN_LOW_DO_WITH_GILL` | 溶氧低位/预测下降 ∧ 检出烂鳃 | +15 |
| `SYN_SAPRO_ON_COLD` | 水温 15–25℃ ∧ 检出水霉 | +10 |

融合公式：

```
weighted_score = Σ(w_模态 × S_模态) + Σ(协同加成)     # 纯公式值，保留供审计
fused_score    = max(weighted_score, 等级下限对应的分数线)
```

**单模态输入时权重在剩余模态上重新归一化**——否则"只有图片"的满分级病害
也只能得 40 分，永远进不了高危，单模态场景会被权重配置直接锁死。

#### 4.3.1 等级下限：为什么加权平均不够

加权平均隐含一个假设：**两个模态在测同一个潜在量，所以可以互相抵消**。
这里不成立——`S_water` 衡量的是**环境危害**（溶氧 1.9，今晚就可能泛塘），
`S_vision` 衡量的是**已见病征**。「水质高危 + 照片上鱼看着还行」不等于风险减半，
只是**症状还没出来**。实测反例：溶氧 1.9 / 氨氮 0.72 / 亚硝酸盐 0.35（水质工具独立判
高危 100 分）+ 一张健康鱼照片 → 加权只有 `0.6×1.0×100 = 60` 分，被压成**预警**。

因此定级不是单纯按分数分档，而是取**分数档位与若干下限的最高档**：

| 下限 ID | 触发条件 | 档位 |
|---|---|---|
| `REDLINE` | 水质工具命中红线 | 高危 |
| `SEVERE_DISEASE` | 高致病性病害置信度 ≥ 0.70 | 高危 |
| `WATER_LEVEL:高危` / `WATER_LEVEL:预警` | 水质工具**自身**的判决等级 | 高危 / 预警 |
| `DISEASE_DETECTED` | 非健康 ∧ 严重度 ≥ 0.70 ∧ 置信度 ≥ 0.40 | 预警 |

- 水质下限**只信任工具自己的 `risk_level`**，不从 `risk_score` 反推——
  否则会与"按分数分档"的既有语义打架。
- 病害下限的置信度门槛取 **0.40 而非工具自身的低置信线 0.60**：两者语义不同。
  低于 0.60 表示"需要人工复核"，但"需要复核" ≠ "可以按正常处理"，
  确诊流程仍要启动，只是同时附一条复核提示；0.40 以下才是纯噪声。
- 下限一旦生效，**`fused_score` 被抬到对应档线**（预警 40 / 高危 70），
  同时把纯公式值记在 `weighted_score`、抬升值记在 `score_floor`、
  触发原因记在 `level_floors`，并追加一条 `LEVEL_FLOOR` 告警。
  接口消费方常按分数排序或筛阈值，若让"高危 / 32.3 分"并存，按分数筛的调用方会漏报。

> 定级只可能被下限**抬高**，不会被压低；下限本身也不改变各模态分数，
> 只是拒绝让一个模态的高风险被另一个模态的"正常"稀释。

### 4.4 建议角色的输出契约（面向养殖户，不面向评委）

建议角色的提示词（`agent/prompts.py` 的 `ADVISOR_SYSTEM_PROMPT`）把输出钉成**四块 JSON**：

```json
{
  "risk_level": "高危",
  "headline": "溶氧快见底了，先开增氧机。",
  "alerts": [{"level": "critical", "title": "水质红线告警", "detail": "…"}],
  "water_actions":  [{"priority": 1, "target": "溶氧", "action": "…", "timing": "…"}],
  "disease_actions":[{"priority": 1, "target": "烂鳃", "action": "…", "timing": "…"}],
  "observation_window": "6 小时后复测溶氧…",
  "confidence_note": "依据：水质 24 点 + 图片检出烂鳃（把握 0.86）。不确定：结论含 12.0 小时外推。复核：建议尽快补测一次水质。"
}
```

**水质块与病害块为什么必须分开**：两块的读者动作完全不同——水质块是"今晚就要动手"，
病害块往往要隔离、要用药、有停药期，混在一个数组里前端无法分别渲染，
模型也容易把"开增氧机"和"捞病鱼"写成同一条。

**语言规则（写死在提示词里，附正反例对照）**：

| 规则 | ✗ 反例 | ✓ 正例 |
|---|---|---|
| 说人话 | 加强水质管理，密切关注溶氧变化 | 把增氧机开到最大，天亮前不要停 |
| 给数字 | 注意氨氮浓度 | 氨氮 0.62，超警戒线 3 倍 |
| 给时限 | 适时增氧 | 现在就开，至少守到早上 8 点 |
| 给顺序 | 采取综合措施 | ①开增氧机 ②停料 ③换 1/3 水 |

**四条铁律**：① 数值一个字都不能改；② 不许新增融合层没给出的结论
（没测的指标绝不能说超标，健康鱼绝不能说有病）；③ 建议必须可落地
（禁止"加强管理""注意观察"这类无法执行的措辞）；④ 只输出中文。

**护栏：模型写错等级 → 整份作废。** 风险等级与告警一律由融合层强制回填。
但只回填等级会得到一个自相矛盾的报告——顶部写「高危」、正文写「水质正常，一切良好」，
**这种报告比没有报告更危险**。所以一旦模型复述的等级与融合层不一致，
就判定它**误读了输入**，整份输出作废、退回规则模板，并在 `trace` 里留一条
`guardrail` 轨迹、写进 `degraded_reasons`，绝不静默覆盖。

### 4.5 可信度说明：事实由规则算，措辞照模板给

`confidence_note` 是整份报告里唯一描述「这份结论有多硬」的字段，因此它**不能**：

- 交给模型自由发挥——评估可靠性需要看到 `trace`、数据质量、模态失败记录、
  知识库核对状态，而模型只看到一小段文本，它写的「本次结论可靠」是凭印象说的；
- 写成一句放之四海皆准的套话（「请结合实际判断」）——那句话等于没说；
- 在规则模板路径和 LLM 路径各写各的——两边措辞一旦分叉，
  页面上的可信度说明就会和提示词里给模型的事实对不上，这种问题极难排查。

所以实现是**两段式**（`agent/prompts.py`）：

1. **规则算事实**：`build_confidence_facts(water, vision, fusion, knowledge)` 把
   「水质几个点 / 是否够 24 点 / 外推几小时 / 是否降级 / 图片查出什么、把握多少 /
   是否低置信 / 哪个模态失败了 / 引用的资料里有几条未与标准原文核对」
   算成一个扁平 dict。**这里没有任何措辞**，全是可断言的数值与布尔。
2. **模板给措辞**：`format_confidence_block(facts)` 把事实填进固定三句式
   （`依据：…。不确定：…。复核：…。`）交给模型，`render_confidence_note(facts)`
   用同一份事实渲染成品句，供规则模板路径与回译兜底使用。
   **两条路径同源**，所以「没接大模型」和「接了大模型但输出不可用」写出来的
   可信度说明结构一致，只有措辞丰富度不同。

实测输出（`docs/_probe_confidence.txt`，同一个函数在两种语言下）：

| 场景 | 输出 |
|---|---|
| 水质 24 点 + 烂鳃 0.86 + 1 条未核对资料 | 依据：水质 24 点 + 图片检出烂鳃（把握 0.86）。不确定：结论含 12.0 小时外推；引用 3 条资料中 1 条未与标准原文核对。复核：建议尽快补测一次水质。 |
| 水质 48 点、无病害输入 | 依据：水质 48 点。不确定：无明显不确定项。复核：无需人工复核。 |
| 水质仅 12 点 + 工具降级 + 图像分析失败 | 依据：水质 12 点。不确定：图像分析失败；水质仅 12 点（少于 24 点），趋势偏弱等。复核：建议人工重做图像分析。 |
| 只有图片、病害置信度 0.45 | 依据：仅鱼体图片（无水质时序）。不确定：病害置信度偏低（0.45），需人工复核。复核：建议人工复看图片。 |

三条设计细节：

- **「不确定」为空时必须明说**「无明显不确定项」，不能留空——
  留空读者会以为我们忘了写；同理「复核」为空写「无需人工复核」。
- **同一个事实同时决定「不确定」和「复核」**：图像分析失败，
  前半句说"这份结论里没有图像证据"，后半句说"请人工重做图像分析"。
  两栏由一份事实推导，不会出现「说了不确定却没说该怎么办」。
- **英文路径独立成句**（`Evidence: … Uncertainty: … Review: …`），
  疾病名走 `DISEASE_LABELS_EN` 映射。D12 的前提是英文底座用英文思考，
  如果给它的素材还是中文，等于先让它做一次阅读理解的翻译。

---

## 5. 演示场景与实测输出

`scripts/demo_agent.py` 跑出的确定性结果（可重复验证）：

| # | 输入 | 风险等级 | 融合分 | 关键机制 |
|---|---|---|---|---|
| 1 | 仅水质（24h 恶化） | **高危** | 100.0 | 命中红线 `DO_REDLINE_PROJECTED`；单模态权重归一化 |
| 2 | 仅图片（细菌性烂鳃） | **高危** | 85.0 | 检出高致病性病害直接升级；图像 0.85×1.00 |
| 3 | 水质恶化 + 细菌性出血 | **高危** | 100.0 | 2 条协同规则，图像 0.90×0.40 |
| 4 | 水质良好 + 健康鱼 | 正常 | 0.0 | 双模态均无风险 |
| 5 | 低温 22℃ + 水霉 | 预警 | 55.8 | 命中 `SYN_SAPRO_ON_COLD`，协同加成 18 |
| 6 | 图片路径不存在 | 正常 | 0.0 | ⚠ 模态失败显式上报，Agent 不断链 |

> **场景 2/3/5 的融合分随图像后端而变**，上表是本机**已有训练权重**
> （`models/yolov8-fish.pt`）时的实测值：标签与置信度由模型推理给出，
> **与图片所在文件夹名无关**。若把权重删掉（或换一台没训练过的机器），
> `fish_disease_detect` 落回 mock 兜底、demo 传入的 `mock_source="folder_hint"`
> 才生效（从文件夹名取标签），同样三行的融合分变成
> **73.08 / 100.0 / 49.28**——等级与命中机制不变，只有分值不同。
>
> `mock_source` **默认关闭**（默认 `hash` 随机播种），且真实 YOLO 推理成功时
> 代码路径**根本不会读文件名**。它只用于链路联调，**不用于任何准确率评测**。
> 想确认当前跑的是哪条路径：看 demo 开头那行【演示说明】，或
> `fish_disease_detect(...)["model_version"]`（`yolov8-fish.pt` = 真实模型，
> `mock-v0.1` = 兜底）。

### 5.1 边界场景回归：等级下限为什么必须存在

`scripts/demo_cases.py` 专门打融合层的边界（完整输出见 `docs/demo_cases_output.txt`）。
下面这 6 组是**等级下限这个机制的直接证据**——其中第 3 组在修复前会被判成预警。

| # | 输入 | 等级 | `fused_score` | `weighted_score` | 生效下限 |
|---|---|---|---|---|---|
| 1 | 溶氧 2.8 / 氨氮 0.35 + 烂鳃 0.87 | 高危 | 100.0 | 100.00 | `SEVERE_DISEASE`、`WATER_LEVEL:高危`、`DISEASE_DETECTED` |
| 2 | 水质全优 + 水霉 0.62（22℃） | 预警 | 40.0 | 27.36 → 40 | `DISEASE_DETECTED` |
| 3 | 溶氧 1.9 / 氨氮 0.72 / 亚硝酸盐 0.35 + **健康鱼 0.95** | **高危** | 70.0 | 60.00 → 70 | `WATER_LEVEL:高危` |
| 4 | 溶氧 4.2 + 健康鱼 0.95 | 预警 | 40.0 | 17.12 → 40 | `WATER_LEVEL:预警` |
| 5 | 水质全优 + 烂鳃 0.95 | 高危 | 70.0 | 32.30 → 70 | `SEVERE_DISEASE`、`DISEASE_DETECTED` |
| 6 | 水质全优 + 烂鳃 0.42 | 预警 | 40.0 | 14.28 → 40 | `DISEASE_DETECTED` |

三组典型反例，修复前的行为与原因：

- **第 3 组（单调性违反）**：水质工具独立判**高危 / 100 分**（三项 severity = 1.0），
  只因照片上的鱼"看着健康"就把结论稀释成**预警**。理由见 §4.3.1。
- **第 5 组（等级与分数自相矛盾）**：等级被 `SEVERE_DISEASE` 抬到高危，
  但 `fused_score` 只有 32.3，报告出现「高危 / 32.3 分」。按分数筛的调用方会漏报，
  所以下限一旦生效就把 `fused_score` 一并抬到档线。
- **第 2、6 组（漏报）**：水质满分时，任何**非**高致病性的检出都被 0.4 的图像权重
  压到 40 分线以下，结论落回**正常**、标题写"水好鱼也好，照常管理就行"。
  这正是"权重配置把真实场景锁死"，与单模态归一化要解决的是同一类问题。

> 演示用的图片标签与置信度是**人工指定**的（字段形状取自真实工具，
> 便于观察融合层行为），**不得用于精度评测**。

### 5.2 病害检出时的两条文案约束

边界回归还暴露了一个纯文案、但同样会误导生产的缺陷：水质满分、只查出病害时，
规则模板原来会输出

```
P1 【当前无异常】按常规量喂料，夜里开增氧机，早晚各巡塘一次。 @照常做
```

**给一塘确诊病害的鱼投"常规量"饲料是明确的生产错误**——病鱼摄食下降，
投下去的料只会变成残饵，进一步升高氨氮、拉低溶氧，还要把药饵的摄入量算错。
项目自己的知识库把这条写成硬规则
（`knowledge/corpus/05_投料管理规范.md:71`「发病期应先停料 1–2 天再改投药饵」），
所以现在：

1. **只要病害识别不是「健康」，`water_actions` 的第一条就是「停料」**，
   与水质数值无关；`DOMAIN_CONSTRAINTS`（LLM 路径）与 `_fallback_advisories`
   （模板路径）两条路都钉住了同一条规则，避免"接上大模型反而变差"。
2. **`observation_window` 与 `headline` 跟着风险来源走**：主因是水质就复测水质，
   主因是病害就"数死鱼、复看病灶"。修复前水质满分的案例标题写的是
   "水质开始不对"，养殖户会照着去调水，而真正该做的是捞病鱼、用药。

---

## 6. 后端接口（FastAPI）

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m uvicorn fisheries.api.app:app --host 127.0.0.1 --port 8812
# Swagger: http://127.0.0.1:8812/docs
```

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/health` | 健康检查（含知识库状态、降级原因、启动提示） |
| POST | `/api/v1/assess` | **主接口**：水质时序 +（可选）图片 → 风险等级 / 告警 / 建议 / 依据 |
| POST | `/api/v1/tools/call` | 跳过 LLM 直调工具，便于前端分步演示与调试 |
| POST | `/api/v1/ingest/sensor` | 传感器读数追加到会话缓冲（内存态） |
| GET | `/api/v1/ponds/{pond_id}/readings` | 取回缓冲的读数 |
| POST | `/api/v1/ingest/image` | 上传鱼体图片，返回 `upload_id` 与即时识别结果 |
| POST | `/api/v1/chat` | 知识库问答（RAG 检索，不经过 Agent） |
| GET | `/api/v1/ponds/{pond_id}/history` | 该塘口的历史评估记录 |
| GET | `/api/v1/config/thresholds` | 当前生效的阈值表与物种档位（前端画仪表盘用） |
| GET | `/api/v1/config/tools` | 导出 function-calling schema（前端/其他 Agent 复用） |

> 上表由 `scripts/demo_api.py` 逐个跑过，路径以 `src/fisheries/api/routes.py` 的
> `router` 为准（`python -c "from fisheries.api.routes import router; ..."` 可导出全量路由）。

**接口层的设计约束：**

- 接口层**不重新实现任何业务逻辑**，只做四件事：校验入参、调用已实现的工具/Agent、
  把结果转成稳定的响应契约、把异常转成明确的 HTTP 状态码。
- 契约用 `_Strict` 基类（`extra="forbid"`）——请求里多写一个字段是 **422 而不是静默忽略**。
  竞赛演示时字段拼错会立刻报错，不会变成"传了但没生效"的幽灵 bug。
- **图片默认只接受 `upload_id`**，不接受服务器任意路径。`FISHERIES_API_ALLOW_LOCAL_PATH=1`
  才开放 `image_path`（演示方便，但那是任意文件读取面，公网必须关）。
- 知识库不可用时 `/assess` 仍返回 200，只是 `degraded_reasons` 里会写明
  「知识库不可用，建议未引用规范依据」；只有 `/chat` 本身是知识库功能才返回 503。

```powershell
# 最小调用示例
$body = @{
  pond_id = "P-A"
  readings = @(
    @{ timestamp="2026-10-14T00:00:00+08:00"; water_temp=28.0; ph=8.05;
       dissolved_oxygen=6.2; ammonia_n=0.15; nitrite_n=0.05 }
    # …至少 3 条
  )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8812/api/v1/assess `
  -ContentType "application/json" -Body $body
```

---

## 7. Web 演示页（Gradio，现场答辩用）

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[webui]"   # 只需一次
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m fisheries.webui.app
# 打开 http://127.0.0.1:7860
```

页面上做的事只有一件：**上传水质表格 +（可选）上传鱼体图片 → 点「运行 Agent」→ 看结论**。
不用记命令、不用拼 JSON、不用打开 Swagger，评委可以自己上手点。

| 区域 | 内容 |
|---|---|
| 输入区 | 水质表格（上传 CSV/TSV/XLSX，或直接**粘贴**表格文本）、鱼体图片、塘口编号、养殖品种、病害识别模式、是否引用知识库 |
| 顶部大结论 | 风险等级徽章 + `fused_score` + 一句话结论 + `reason` |
| ① 告警 | 按 `critical` / `warning` / `info` 三档配色的卡片（红/橙/蓝） |
| ② 处置建议 | **水质调控**与**病害处置**两张表分开列，含优先级、对象、怎么做、什么时候做 |
| ③ 数据接入质量 | 列名映射、被忽略的列、采样间隔及其来源、时间范围、缺失格、量纲提醒 |
| ④ 图片识别 | 判定病害、置信度、五个标签的完整概率分布、是否降级模式 |
| ⑤ 融合依据 | `weighted_score` / `score_floor` / `fused_score` / 协同加成 / 有效权重 / **触发的等级下限** |
| ⑥ 知识库引用 | 检索引擎、命中块数、按用途分组、每条出处标题链（含"未核对标准原文"标记） |
| ⑦ 推理链路 | 五步链路逐条展开（调度 LLM → 调工具 → 工具返回 → 融合 → 建议），页脚给耗时与降级原因 |

### 7.1 三个刻意的工程选择

- **表格接入不直接丢给工具，先过一层 `table.py`。** 真实数据集里列名有
  `Dissolved Oxygen(g/ml)` / `DISOLVED OXYGEN`（拼写错）/ `PH` / `Ammonia(g/ml)`
  等至少四种写法，直接喂给工具只会得到 pydantic 校验错。`table.py` 做四件事：
  ① 精确别名表匹配列名（**不做模糊匹配**——`Total_Phosphorus_mg_L` 里含 `ph`、
  `Nitrate` 与 `Nitrite` 只差三个字母，模糊匹配会把列认错）；
  ② 解析四种时间格式并重排时间轴；③ 把 `-127`/`-999` 这类设备哨兵值按缺失处理
  （否则"水温 -127℃"会直接顶到危险档）；④ 核查量纲并提醒。
- **缺失的必填格按「丢整行」处理，而不是留个 `None` 键。** 水质工具要求每个点
  同时具备水温、pH、溶氧、氨氮四项；留 `None` 键会让**整个请求**被校验拒绝，
  用户看到的是"工具调用失败"，而真正原因只是某一格是空的。丢整行 + 说清丢了几行，
  降级范围才是可控的。
- **补插时间戳必须只动缺失行。** 工具的 `_resolve_interval` 要求**所有**点都有时间戳
  才肯用时间戳推断间隔，否则整体退回 `interval_minutes` 参数，趋势外推的时长含义就错了。
  所以缺时间戳的行要补，但补法必须是「按相邻已知点的行序线性内插」——
  按"首点往后每步加一个间隔"去铺，补出来的点会抢走真实时间戳的位置，
  把后面所有实测点的时刻整体推后一格（数值与时刻错位）。

### 7.2 依赖边界

`gradio` 是**可选依赖**（`pyproject.toml` 的 `[project.optional-dependencies] webui`）。
整个 `webui/` 包里**只有 `app.py` 一个文件 import gradio**，其余三个模块
（`table.py` / `runner.py` / `render.py`）纯标准库 + pydantic，
所以 `tests/unit/test_webui.py`（69 项）能在没装 gradio 的机器上跑。

> Gradio 6 把 `theme` / `css` / `js` / `head` 从 `Blocks()` 挪到了 `launch()`，
> 且在 `Blocks()` 上传会告警。`pyproject.toml` 里因此钉的是 `gradio>=6.0`。

演示页默认走**离线 MockLLM + 规则模板建议**，与命令行 demo 完全同一条链路；
配置了真实模型（§2）后，页面上会自动显示当前用的是哪个后端。

不装 gradio 也想看页面长什么样：

```powershell
.\.venv\Scripts\python.exe scripts\demo_webui.py          # 把页面上每栏原样打印到终端
.\.venv\Scripts\python.exe scripts\demo_webui.py --app    # 额外验证 Gradio 组件树能装配
```

真实跑出来的完整输出留了一份在 [docs/demo_webui_output.txt](docs/demo_webui_output.txt)
（24 小时水质 + 数据集里的真实烂鳃图，等级【预警】、融合分 68.9），
不装任何依赖也能直接看页面每一栏的原文。

### 7.3 把前端交给别人重做

如果页面要交给另一套前端重做，看 [docs/前端对接说明.md](docs/前端对接说明.md)：
接口清单、请求/响应字段表、**已经实现了哪些功能**、以及"哪些东西绝不能由前端自己算"。
配套的真实响应样例在 [docs/_sample_api_responses.json](docs/_sample_api_responses.json)
（预警 / 高危两份，都是真实调用产生的）。

---

## 8. 工具契约

### 工具① `water_risk_predict`

```python
water_risk_predict(
    readings: list[dict],        # 时序读数，≥3 条
    pond_id: str = "unknown",
    interval_minutes: int = 60,
    horizon_hours: int = 72,
    species: str = "鲤科",       # 鲤科 / 罗非鱼 / 虾蟹
) -> dict
```

每条读数：`timestamp`（可选）、`water_temp`、`ph`、`dissolved_oxygen`、
`ammonia_n` 必填；`nitrite_n` **可选**。

> `nitrite_n` 做成可选是刻意的：实测所有 IoTPond 数据集**只有硝酸盐没有亚硝酸盐**，
> 设为必填会让工具在真实数据上完全不可用。缺失时该指标整列剔除、其余权重自动归一化。

### 工具② `fish_disease_detect`

```python
fish_disease_detect(
    image_path: str,
    conf_threshold: float = 0.25,
    top_k: int = 4,
    weights_path: str | None = None,
    force_mock: bool = False,
) -> dict
```

输出病害类型为 **烂鳃 / 出血 / 水霉 / 健康**，另有第 5 个安全兜底值 **「其他」**——
底层数据集是 7 类（含寄生虫病、病毒性白尾病），
强行塞进 4 类等于把病鱼判成健康鱼，生产环境是安全事故。
同时用 `raw_class` 字段无损保留底层 7 分类原始判定。

---

## 9. 测试

```powershell
.\.venv\Scripts\python.exe -m pytest                 # 全部 939 项
.\.venv\Scripts\python.exe -m pytest tests/unit/test_agent.py -v
.\.venv\Scripts\python.exe scripts\demo_rag.py --only e2e   # 端到端含 RAG
```

> ⚠ 统计行被吞时加 `-o addopts=""`：`pyproject.toml` 里已有 `addopts="-q"`，
> 命令行再叠一个 `-q` 会变成 `-qq`，pytest 就**不打印**汇总行了。

覆盖重点：

- **工具层**（85 项）：5 个指标每档阈值端点逐一钉死；红线触发；历史尖峰不粘连；
  分子氨协同；亚硝酸盐缺失权重重归一化；NaN/超量程拒绝；图像确定性；失败路径；
  **真实权重下标签只由模型决定**（同一张图放进不同文件夹名必须得到逐字段一致的输出）；
  **训练脚本的类别与产品代码的类别映射必须对齐**（训练类别 ⊆ `RAW_CLASS_TO_LABEL` 键、
  四分类恰好覆盖健康/烂鳃/出血/水霉、每个训练类别都有 `severity_weight`）
- **融合层 / 调度层**（114 项）：单模态权重归一化；四条协同规则独立触发条件；
  置信度加权；定级阈值边界；模态失败上报；护栏补调；防重复调用；工具异常不扩散；
  LLM 不可用降级；建议输出 JSON 解析（含 markdown 围栏）；
  **模型写错风险等级时整份输出作废并留痕**；
  **等级下限不被单模态稀释**（水质高危 + 健康照片仍判高危）；
  `fused_score >= weighted_score` 的单调性；下限生效时 `LEVEL_FLOOR` 证据/告警齐全；
  病害检出时不得建议"按常规量喂料"、观察窗口要跟着风险来源走；
  **可信度说明三句式齐全、"不确定"为空时必须明说、同一份事实同时推导"不确定"与"复核"**；
  英文路径下"可信度素材"必须是英文且不能混进中文病名；模型漏写这一栏时回填模板句；
  `FISHERIES_DISABLE_KNOWLEDGE=1` 时跳过检索
- **回译层**（144 项，D12）：语言按后端自动判定且可被入参/环境变量覆盖；
  未知后端默认中文（默认英文等于默认让模型用非母语作答）；
  词表回译最长优先（`dissolved oxygen` 先于 `oxygen`）且**整句英文一律拒绝**而不是硬译；
  名词短语不加句号（`氨氮。` 是病句）；**`risk_level` / `alerts` 一个字都不过翻译**；
  译不出来的字段退回规则模板同名字段（**绝不留英文**）；动作条目逐条补位；
  每一次退回都进 `notes`；翻译模型超时/401/改数字都不抛异常只降级；
  没配翻译模型时复用调度角色（本地后端免 key；provider 配了但没 key 时挡掉注定失败的调用）；
  端到端：英文提示词 + 英文 JSON → 中文报告，且回译状态不跨 run 泄漏
- **评测层**（181 项，M8 外部验证子包自身，不跑真实数据集也能测）：
  标签映射三套口径都在合法等级内且未知取值**报错而不是猜**；五级目录与序号顺序一致；
  混淆矩阵的 accuracy / 逐类 P-R-F1 / 宏平均；**类别极不平衡时高准确率不换来高 κ**
  （永远猜多数类的 κ ≈ 0）；单类别时 κ 返回 0.0 而非 NaN（NaN 进不了 JSON）；
  Spearman 只看单调性（非线性但单调仍是 1.0）、无方差时返回 0；
  按站点分组按日期排序；趋势窗口跳过历史不足的记录；对照口径三点数值相同、等间隔；
  `water_quality_index` 由标签决定时判"有刻度"、无关时判"没刻度"；
  几乎全空的列被丢弃而不是参与排名；
  引擎外推 = `末点 + 最小二乘斜率 × 一步`（**刻意调用线上那个函数，而不是重写一遍**）；
  外推更差时改善率为负、两者都完美时不除零；
  工具拒绝的窗口被记成失败样本而不是中断整轮；`limit` 提前停止；
  报告在缺检查项时写"（本次未运行）"而不是静默省略（**静默省略会让第 4 节的数字看起来可读**）；
  **第 8 节（图像模态）在场与缺席两种形态都渲染**，mAP 的弱口径 caveat 必须跟着数字一起出现，
  「病害 → 健康」的漏检数要按混淆矩阵算对，指标文件缺字段也不能崩
- **RAG 层**（100 项，用 `HashEmbeddings` 假向量，不加载 90 MB 模型）：
  front-matter 解析与非法值拒收；标题链构造；表格整体保留不切；列表原子性；
  整节粘合；`section_role=meta` 被谓词排除（且只认"标题主语是本项目"，
  不误伤顺带提到本项目的领域小节）；jieba 领域词不被切碎；
  RRF 融合确定性；软加权；标题去重；轮转预算不让低优先级用途整块消失；
  清单指纹不一致时检出；`store` 拒绝维度/模型不匹配的旧库
- **接口层**（104 项 = `test_api.py` 79 + `test_api_rag.py` 25，`TestClient` 不启动真实服务）：
  `_Strict` 拒收多余字段返回 422；空 readings；两模态都缺；图片不可达；
  工具名未知；上传图片的扩展名/大小/MIME 校验；
  **带 `timestamp` 的读数不因 datetime 序列化而 500**；
  知识库不可用时 `/assess` 仍 200 而 `/chat` 返回 503；
  `use_knowledge=false` 时确实跳过检索；并发请求下 `run_lock` 不串台；
  **`/api/v1/health` 申报的工具后端必须同时满足"有权重"与"装了 ultralytics"**，
  两种条件各自缺失时的措辞不同（只查 `find_spec` 会在没权重时谎报 `yolov8-cls`），
  且据此判断的 `degraded` 状态要跟着变
- **演示页接入层**（69 项，不 import gradio）：
  真实数据集四种列名写法都能认（含 `DISOLVED OXYGEN` 拼写错、`Lenght` 拼写错）；
  **模糊匹配会认错列，所以只做精确别名匹配**；四种时间格式解析；
  `-127`/`-999` 哨兵值按缺失处理并提醒；负值不当真实读数；
  必填格缺失时丢整行而不是留 `None` 键（留 `None` 会让整个请求被校验拒绝）；
  丢到不足 3 点必须报错而不是让工具层抛异常；
  超长表格截尾保留最新数据；乱序时间重排；**补插时间戳只动缺失行、真实点的时刻不被推移**；
  量纲超出物理量程时提醒核对列映射
- **数据集层**（74 项，不依赖本机真的装了数据）：
  注册表每个键都有非空 `entry`、`label`、`note`；名字不重复；
  解析顺序四级（`FISHERIES_DATA_ROOT` → `data/datasets/` → 历史目录 → kagglehub 缓存）逐级生效且优先级可测；
  **`FISHERIES_DATA_ROOT` 只是优先级提示而不是强制覆盖**（指错时仍能回落到历史目录）；
  `required` 子项缺失时 `require()` 抛错**且报错消息里必须带下载命令**（报错要能直接抄去跑）；
  `default_path()` 找不到时返回 `None` 而**不抛异常**（模块级默认值不能把 import 搞崩）；
  **回归守卫**：`src/`、`scripts/`、`tests/` 里不得再出现 `Agent\数据集` 或 `Path(r"D:\` 这类绝对路径字面量；
  文档与脚本不得再要求调用方先设 `PYTHONPATH`（探针要自己把 `src` 插进 `sys.path`）；
  `fetch_datasets.py` 的 `--list/--status/--where/--json` 在数据缺失时也不报错
- **控制台层**（27 项，用假流模拟 GBK / UTF-8 / 无 `reconfigure` 的控制台）：
  `install()` **只把 `errors` 放宽为 `replace`，绝不动 `encoding`**（单方面改成 UTF-8 会让字节
  与终端对不上、中文变乱码，那正是被替换掉的旧写法）；
  **编码锁**：实测 `import ultralytics` 会把 `sys.stdout` 从 gbk 改成 utf-8，于是同一个流
  前半段 GBK、后半段 UTF-8——终端上加载模型之后整段乱码且**退出码为 0**；`install()`
  记住初次见到的编码，可重复调用来扳回，且不会把被改成的 UTF-8 误记成"真实编码"；
  没有 `fileno()` 的流（`StringIO` / 测试替身）不做编码锁，只放宽 `errors`；
  `✔`/`✘`/`⚠` 在 GBK 下降级为**等宽**的 `[OK]`/`[--]`/`[!!]`（承载信息的符号不能退化成 `?`）；
  **回归守卫**：`scripts/` 下不得再出现 `reconfigure(encoding=` 旧写法；
  所有打印非 GBK 符号的脚本必须调用 `_console.install()`
- **启动器层**（35 项，管的是双击那一刻能不能看清、听不见噪声）：
  **`.bat` 里不许出现任何非 ASCII 字节**——cmd.exe 按控制台代码页分块解码批处理文件，
  一个中文字符就会让它把后面的行**从中间切入当成命令执行**，而退出码仍是 0、服务照样起来，
  只有黑窗口里多出 `'�' is not recognized as an internal or external command` 这种噪声；
  隔离实验的判据是**等长 ASCII 变体**（2036 B 对 2036 B）就干净，所以是「内容」不是「长度」，
  加 BOM 或调 `chcp` 的顺序都无效；
  于是中文名 `.bat`（`启动演示页.bat` 等）只做转发，实际逻辑在纯 ASCII 的 `run-*.bat`，
  所有中文提示改由 `src/fisheries/launcher.py` 与 `scripts/selftest.py` 打印——
  这样中文走的是 Python 的编码兜底（上一条），不再经过 cmd 的批处理解析器

---

## 10. M8 外部验证：池塘实测数据上的两组验证

复现：`\.\.venv\Scripts\python.exe scripts\evaluate_water_rules.py`（约 25 秒，读 11 个 CSV / 96 MB），
完整报告在 [`docs/M8_评测报告.md`](docs/M8_评测报告.md)，原始指标 JSON 在 `docs/_m8_metrics.json`。

| | 外部验证一 | 外部验证二 |
|---|---|---|
| 数据 | `aquaponics_pond`（11 口塘 / 1,114,970 行） | 同一份数据的溶氧序列 |
| 真值形态 | **按行业阈值现算的**三档标签 | **真实发生的**未来溶氧读数 |
| 验证对象 | 引擎的**阈值定级** | 引擎的**趋势外推** |
| 独立性 | ⚠ **不独立**——标签与引擎是同一套阈值 | ✅ 独立——真值不是算出来的 |
| 结果 | 准确率 72.2%、κ 0.573、漏报 0.0% | 引擎**劣于**持久性基线 14.5% |

### 10.1 换数据的原因：上一版的两把尺子量程不对

上一版的两组数据（`china_water_pollution_data.csv` 与河川溶氧宽表）都是**地表水断面监测**，
而引擎是按**池塘养殖**阈值设计的——同一份水在两套口径下本来就会得出不同结论。
上一版的两条结论（「污染标签与所有测量列统计独立」与「河川溶氧外推劣于基线」）
都指向同一个问题：**不是引擎不行，是尺子量错了对象。**

现在两组验证跑在同一份数据上：`Sensor Based Aquaponics Fish Pond Datasets`
（11 个 CSV、1,114,970 行、2021-06-18 ～ 2021-12-30、原始采样间隔中位数 20 秒）。
它是**真·时序**，每条塘有上万条连续读数——所以连上一版「复制凑长度」的瞬时口径妥协也不需要了。
两条旧数据集在 `datasets.py` 里**保留但标注「已停用」**，停用理由写在 `note` 里。

### 10.2 外部验证一：两套阈值口径的对照（**标签不独立**）

在 **3,183** 个不重叠的 3 小时窗口上：准确率 **72.2%**、Cohen's κ **0.573**、Spearman ρ **0.786**、
漏报率 **0.0%**、误报率 19.8%、红线召回 1123/1127（99.6%）、主控因子一致率 1644/1907（86.2%）。

**但这些数字不是「引擎准确率」。** 标签是我们在报告 2.3 节按 `water_risk.py` **同一套阈值**
算出来的：标签取**最差指标**（最坏情况），引擎取**加权评分 + 时间持续性**（平均 + 趋势）。
所以这组测的是**两套口径的分歧**，不是与事实的符合度。

它仍有确切价值：把「引擎会怎么偏离最坏情况口径」量化了，且逐塘分解与主控因子一致率
能指出分歧出在哪个指标上。**报告里不出现「准确率 X%、可用于生产」这种话。**

### 10.3 外部验证二：溶氧外推 vs 持久性基线（真值独立）

在 **8,198** 个样本上（目标点互不重叠、回看窗口之间有重叠），用过去 18 个点（3 小时）预测 **1 小时之后**的溶氧——
真值是**真实发生的读数**：

| n | 引擎 MAE | 基线 MAE | 改善 | 方向判对 |
|---|---|---|---|---|
| 8,198 | **1.943** | **1.696** | **-14.5%** | 44.1% |

结论是负面的但可量化：**线性外推不如「什么都不做」**。
`water_risk.py` 的 `_predict_series()` 就是留给 LSTM 的替换点。
（引擎的**实时超限判定**——溶氧 < 3.0 预警、< 1.5 泛塘、氨氮 > 0.5 高危——不依赖外推，
所以这条不影响那部分逻辑。）

### 10.4 三个必须写在数字前面的量纲判断

数据列名标着 `Ammonia(g/ml)` / `Dissolved Oxygen(g/ml)` / `Nitrate(g/ml)`，字面量与物理量级矛盾：

1. **氨氮实际是 µg/L。** 依据不是猜的：同一份数据里每条鱼的体长体重都在记录，而**每条塘的鱼都长得很好**
   （IoTPond7 体重 8.0 → 998.5 g，**+12,360%**）。按 0.55～60 g/ml 读相当于致死红线的百万倍，
   鱼不可能一边泡着一边增重 120 倍；÷1000 后中位数落在 0.0006～0.06 mg/L，全在适宜带——与增重事实一致。
   **IoTPond8 是唯一标了 `(mg/L)` 的文件，它不换算**，数值量级恰好是其余文件的 1/1000，这是最强旁证。
2. **溶氧不换算**（中位数 3.3～18.3，只能是 mg/L）。
3. **丢弃哨兵与越界值**：温度 < 0 ℃、pH 不在 0～14、溶氧 > 20 mg/L、氨氮换算后 > 50 mg/L。
   温度 −127、pH −3.137 是缺测码；溶氧最大 44.98 mg/L（25 ℃ 饱和约 8.4）是物理不可能。
   本次共清洗 **384,372** 个这样的读数。

换算系数与量程表写在 [`src/fisheries/evaluation/ponds.py`](src/fisheries/evaluation/ponds.py)
的 `UNIT_DIVISORS` 与 `RANGES` 里，是**可改常量**——改掉它们，报告里所有数字都会跟着变。

### 10.5 一个真实盲区，和一条不依赖自建阈值的检查

- **亚硝酸盐填不进去。** 这份数据只有硝酸盐、没有亚硝酸盐，引擎的一级指标 `nitrite_n`
  在这份数据上无法取值，权重被重新归一化。真实塘口亚硝酸盐超标时本评测看不到。
- **合理性检查：标签指向与实际产出方向一致。** 11 口塘的「高危样本占比 vs 鱼体重增长」
  Spearman ρ = **-0.582**——高危占比高的塘，鱼确实长得慢。这条检验**不依赖**我们自建的阈值，
  是唯一一条能说明「标签没有荒谬到反着来」的证据（`labels.growth_coherence`）。

### 10.6 评测代码本身也是被测对象

`src/fisheries/evaluation/` 只做评测，不参与线上决策链路，`tests/unit/test_evaluation.py`
钉住它**自己**算得对不对——阈值分档、混淆矩阵与 κ、Spearman、滑窗切分、
**量纲归一与量程清洗的先后顺序**、生长自洽性。

理由是：评测代码算错比引擎算错更隐蔽——报告里的数字会看起来很有信息量。
这次就抓到一个真实缺陷：`ponds.py` 的 `cell()` 一度**先按量程清洗、后做单位换算**，
而 `RANGES` 写的是**归一化之后**的量的量程，于是合法的 µg/L 氨氮/硝酸盐读数被静默当成坏值丢掉，
「清洗掉 1,246,392 个越界值」里有一大半其实是好数据。
（调换顺序后：越界值降到 384,372，聚合点从 40,121 升到 49,394，κ 从 0.555 升到 0.573。）

---

## 11. 后续迭代

| 模块 | 状态 |
|---|---|
| M0 工程骨架与数据契约 | ✅ |
| M1 数据清洗与特征工程 | ⏳ 待定（依赖 Q1–Q9 缺陷处理与量纲确认） |
| M2 水质时序风险预测（规则基线） | ✅ 规则版；LSTM 待接入 |
| M3 图像识别工具封装 | ✅ 已训练 YOLOv8n-cls 四分类（留出集 Top-1 93.0%）；仅四类、无病灶定位框 |
| M4 多源融合与风险决策 | ✅ |
| M5 RAG 知识库（混合检索 + 定向检索 + 引证） | ✅ 语料为自编种子语料；标准编号与水质数值已逐条核对现行标准原文 |
| M6 Agent 调度与函数调用 | ✅ 已接入 RAG |
| M7 后端 FastAPI（10 个端点） | ✅ 内存态会话缓冲；持久化与鉴权待做 |
| M8 测试评测文档 | ✅ 两组外部验证 + 939 项单测；集成测试仍缺（见下） |
| M9 Web 演示页（Gradio） | ✅ 表格接入 / 双模态 / 分栏报告；未做多轮对话与历史对比 |
| M10 中文输出（D12 英文 prompt + 回译） | ✅ 提示词双语 / 四级回译后端（本地离线模型 + 分块 + 术语后订正）/ 语言自动判定；见 [`docs/M10_回译层说明.md`](docs/M10_回译层说明.md) |

**建议的下一步**（按性价比排序）：

1. **M8 集成测试**——目前"端到端通过"指的是 8 个 `scripts/demo_*.py` 跑完不报错，
   脚本退出码不是断言，进不了 CI。把 demo 里的固定样例搬进 `tests/fixtures/`
   写成带断言的用例，是回归保护的第一件事；
2. **病害识别补到 7 类 / 补框标注**——现在是 4 分类且 mAP@0.5 走的是弱监督全图框
   口径（**不衡量病灶定位**）。要真做检测需要先有框标注：本机三份图集
   （`Fish Disease Dataset` / `Freshwater…` / `SB-FishDisease`）**一个框都没有**，
   且去重后只有 931 张不同照片，规模也偏小。这条比换模型更关键——**标注决定上限**；
3. **换掉趋势外推**——第 10 节的第二组验证已量化出线性外推劣于持久性基线 **14.5%**
   （真值独立，1,114,970 行池塘实测数据上跑的），`water_risk.py` 的 `_predict_series()`
   就是替换点。用**同一套协议**复测，才能说清 LSTM 是否真的更好（现在的数据不支持这个结论）；
4. **M1 数据清洗**——把 `evaluation/ponds.py` 里那套列名别名 / 单位换算 / 量程清洗
   从评测层提成正式的数据管线，让 Agent 能直接吃原始 CSV，而不是靠调用方手工构造 `readings`。
   `webui/table.py` 是界面侧的最小可用切片，缺的是把它和 `ponds.py` 合成同一条管线并出数据质量报告。
   注意 `ponds.py` 目前**丢掉了**一条信息：它只取引擎用得到的四个字段，
   `turbidity` / `nitrate` / `population` / `fish_length` 读过就扔——提成管线时要保留；
5. **M7 补强**——会话缓冲目前是进程内存态，重启即丢；演示够用，
   若要做成"塘口长期监测"需要落到 SQLite/PostgreSQL，并加上鉴权与限流。

> 语料核验已完成一轮（2026-10-15）：**GB 11607-1989 至今现行**（`GB 11607-2022` 不存在），
> `NY 5051/5052/5071/5072` 已废止且无直接替代（改用 `NY/T 755-2022`），9 篇语料全部 `verified: true`。
> **仍待项目负责人推动的事**：① 向设备厂商确认池塘数据的氨氮量纲（本项目按 µg/L 推断，见 §10.4）；
> ② 索取 SC/T 系列病害检疫规程的完整原文（本次只核到标准编号、名称与状态）。

---

## 12. 数据与合规

### 12.1 工程不自带数据集

**仓库里只有代码，没有数据。** 数据集（本机实测共 1.37 GB）放在工程目录之外，
`src/fisheries/datasets.py` 是「数据集名字 → 磁盘位置」的**唯一真相来源**。
改造前那种把 `D:\设置\项目\Agent\数据集\...` 硬编码进十几个文件的做法已经清零
（`tests/unit/test_datasets.py` 里有回归守卫，再写回去会测挂）。

`datasets.py` **只用标准库**；`kagglehub` 是可选依赖、延迟 import，
不装它也能正常跑——只是不能用 `--fetch` 下载缺的数据集。

查找顺序（先命中先赢，每一级都由测试钉住）：

| # | 位置 | 说明 |
|---|---|---|
| 1 | 环境变量 `FISHERIES_DATA_ROOT` | 指向数据集**总目录**，优先级最高 |
| 2 | `<工程>/data/datasets/<entry>` | 想跟工程放一起就用这个 |
| 3 | `<工程上级>/数据集/<entry>` | 改造前的历史目录，**改造后照样找得到** |
| 4 | `data/dataset_index.json` | 逐条手工映射 |
| 5 | `~/.cache/kagglehub/datasets/<owner>/<slug>/versions/*` | kagglehub 的缓存，**只读不联网** |

```powershell
.\.venv\Scripts\python.exe scripts\fetch_datasets.py --list     # 注册表里有什么
.\.venv\Scripts\python.exe scripts\fetch_datasets.py --status   # 本机有哪些（默认动作）
.\.venv\Scripts\python.exe scripts\fetch_datasets.py --where aquaponics_pond   # 只打印路径
.\.venv\Scripts\python.exe scripts\fetch_datasets.py --status --size --json   # 机器可读
.\.venv\Scripts\python.exe scripts\fetch_datasets.py --fetch all        # 下载缺的（需联网）
```

**缺数据不会让项目起不来**：`require()` 只在真正要用到该数据集时才抛错，
且报错消息里直接带上下载命令；`default_path()` 找不到时返回 `None`，
所以「没挂数据集的机器」上模块仍然 import 得进去，对应用例 `skip` 而不是失败。

### 12.2 瘦身：1.37 GB 里能砍掉 1.07 GB

体检结论（`--status --size` 可复现）——最大的那一份**没有任何代码在用**：

| 目录 | 体积 | 代码是否引用 | 处置建议 |
|---|---|---|---|
| `archive/`（`1..5` 共 2055 张 PNG） | **1,072.4 MB** | ❌ `used_by=()` | **可整目录删除**，删完剩 ~323 MB |
| `Fish Disease Dataset` | 179.6 MB | ✅ 病害识别演示 / 测试 / **训练数据源** | 保留；`Train_Augmented`（≈21,358 张）无代码读取，可只留 `Test/` |
| `Sensor Based Aquaponics Fish Pond Datasets` | 96.0 MB | ✅ **M8 两组外部验证的主数据源** | 保留；`ponds.py` 要读全部 11 个 CSV（单文件只有一口塘，切不出跨塘对比） |
| `Freshwater Fish Disease Aquaculture in south asia` | 24.8 MB | ⚠ 仅作去重比对 | 与上一份同源（重建后净贡献 0 张），可整目录删除 |
| `SB-FishDisease`（7 类 / 460 文件） | 10.6 MB | ⚠ 仅作去重比对 | 与上一份同源（净贡献 0 张），可整目录删除 |
| 其余（`Water Quality` / `river_do` / `china_water_pollution` / `Fish.csv` / 15N / ndjson） | ≈ 1.2 MB | 部分是 | 全留，加起来不到 2 MB |
| 合计 | **1,395.7 MB** | | 删掉 3 项同源/未用的可降到 **~180 MB** |

> 图集三份（`Fish Disease Dataset` / `Freshwater…` / `SB-FishDisease`）去重后
> **只有 931 张真正不同的照片**，后两份一张都没多贡献——判断依据是文件名主干
> 而非 md5（同一张图常以 `.jpg` / `.jpeg` / `.png` 三份并存，字节不同）。
> 详见 [docs/模型训练说明.md](docs/模型训练说明.md)。

工程**内部**还有两处可重建的产物：`data/derived/`（22.6 MB，训练/评测用派生数据）、
`models/pretrained/`（5.3 MB，预训练骨干）。两者都能用
`scripts\build_yolo_dataset.py --force` 与 `scripts\train_yolo_fish.py` 重新生成，
删掉不影响推理与测试；唯独 `models/yolov8-fish.pt`（2.98 MB）删掉后
图像模态会退回 mock 兜底。

若想把数据彻底移出工程：整个 `数据集\` 挪到别的盘，然后设
`set FISHERIES_DATA_ROOT=E:\你的新位置`（或写进 `data/dataset_index.json`），代码一行不用改。

### 12.3 数据来源与许可证

数据集清单、7 个病害类别中英对照、时序数据的 9 类质量缺陷（Q1–Q9）
与许可证说明见 [docs/数据集说明.md](docs/数据集说明.md)。

> ⚠ **阈值标准待核对**：代码中的阈值表为**工程默认值 v0.1**，
> 上线前需与 GB 11607、NY 5051 等现行标准的原文逐条核对，
> 并由有资质的水产技术人员复核。本项目输出**不构成用药指令**。
