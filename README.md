# ProjectFisheries

**面向中国淡水池塘养殖的智能监测与决策 Agent。**

把池塘的**水质时序读数**（水温 / pH / 溶氧 / 氨氮 / 亚硝酸盐）和**鱼体照片**一起交给它，
得到一份能直接执行的处置报告：风险等级、告警、分优先级的措施，以及每条结论的依据出处。

|  |  |
|---|---|
| **输入** | ① 水质时序表格（CSV / TSV / XLSX，或直接粘贴文本）② 鱼体照片（可选） |
| **输出** | 风险等级（正常 / 预警 / 高危）+ 告警 + 水质调控与病害处置建议 + 证据链 + 引用出处 + 完整推理链路 |
| **离线可用** | 不配 API key、不联网也能跑通全链路（内置确定性离线模型 + 规则模板建议） |
| **两个工具** | 水质时序风险预测、鱼体病害图像识别（烂鳃 / 细菌性出血 / 水霉 / 健康） |
| **三种入口** | Web 前端（与接口同源托管在 `/app`）、HTTP 接口（FastAPI，14 个端点）、命令行 |

---

## 目录

1. [它解决什么问题](#1-它解决什么问题)
2. [使用流程](#2-使用流程)
3. [快速开始](#3-快速开始)
4. [输入数据怎么准备](#4-输入数据怎么准备)
5. [内置水质模板](#5-内置水质模板)
6. [输出怎么看](#6-输出怎么看)
7. [接入真实大模型](#7-接入真实大模型)
8. [HTTP 接口](#8-http-接口)
9. [当作库调用](#9-当作库调用)
10. [工程结构](#10-工程结构)
11. [测试](#11-测试)
12. [已知限制与边界](#12-已知限制与边界)
13. [数据与合规](#13-数据与合规)
14. [文档索引](#14-文档索引)

---

## 1. 它解决什么问题

### 1.1 面向谁

面向**淡水池塘养殖的日常巡塘与风险研判**。典型使用者有三类：

- **养殖户 / 巡塘人员**：早上看一眼报告，知道今天该先开增氧机还是先泼药；
- **技术服务人员**：把塘口的传感器数据导出来，快速筛出哪些塘口要优先出诊；
- **系统集成方**：把本项目的 HTTP 接口挂到自己已有的监控平台上。

### 1.2 它做什么

一次完整评估做四件事：

1. **读懂你给的表格**——列名怎么写都能认，单位量纲会核查，坏值会挡掉；
2. **判水位**——按阈值 + 时间趋势算出水质风险等级，并指出主控因子；
3. **看病鱼**——识别烂鳃 / 细菌性出血 / 水霉三类高发疾病；
4. **合起来给结论**——单看水质不严重、单看鱼病也不严重，但同时出现就危险，融合层负责把这件事判出来，
   并给出**能照着做的措施**（开增氧机、停料、换水、隔离……）和**依据出处**。

### 1.3 它不做什么

说清楚边界比罗列功能更重要：

- **不构成用药指令**。给出的是措施方向与动作顺序，剂量与药品种类必须由有资质的水产技术人员决定。
- **不替代人工复核**。报告里的 `confidence_note` 会明说这次结论有多硬、哪里不确定、要不要复核。
- **不做长期数据存储**。接口层的塘口会话缓冲是进程内存态，重启即丢（见 [§12](#12-已知限制与边界)）。
- **阈值不是国标全文**。代码里的分档是**工程默认值 v0.1**，以现行标准为基准做了一轮核验，
  但警戒线 / 危险线 / 红线是本项目在标准之上扩展的，不是标准原文（见 [§13.3](#133-标准与阈值口径)）。

---

## 2. 使用流程

### 2.1 一张图看懂

```
                  ┌─────────────────── 你的输入 ───────────────────┐
                  │  水质表格（≥3 条读数）        鱼体照片（可选）  │
                  └───────────────────────┬────────────────────────┘
                                          ▼
   ① 接入与清洗   列名识别 → 时间轴排序 → 哨兵值/坏值剔除 → 量纲核查 → 缺格丢行
                                          ▼
   ② 工具执行     水质工具：阈值 + 趋势 + 外推  →  风险等级 / 恶化方向 / 倒计时
                  图片工具：病害分类            →  病害类型 / 置信度 / 严重度
                                          ▼
   ③ 护栏         必调的工具漏调了吗？漏了由调度层补齐，并留痕（不隐瞒）
                                          ▼
   ④ 融合定级     加权 + 多源协同加成 + 等级下限（确定性规则，不交给模型）
                                          ▼
   ⑤ 定向检索     按融合结论决定查哪几类资料：应急 / 水质 / 病害 / 投料
                                          ▼
   ⑥ 生成报告     措辞由模型写，数字只能引用不能改 → 报告
                  ┌──────────────────────────────────────────────────────┐
                  │ 风险等级 · 告警 · 水质调控建议 · 病害处置建议         │
                  │ 证据链 · 引用出处 · 可信度说明 · 完整推理链路         │
                  └──────────────────────────────────────────────────────┘
```

**这条链路最重要的一个约束：定级不交给大模型。**
风险等级、融合分、告警全部由确定性规则算出，模型只负责把结论翻译成养殖户看得懂的话。
所以同一份数据永远得出同一个等级，被问「为什么判高危」时能给出规则 ID 和数值依据。

### 2.2 第一次上手：四步

假设你刚把仓库克隆下来，想先看看它长什么样：

```bash
# 第 1 步：装环境（见 §3.2 的完整安装命令）
# 第 2 步：跑一遍全链路，确认能起来（不需要任何 API key）
python scripts/demo_agent.py

# 第 3 步：起服务，用鼠标点一遍（前端页面与接口是同一个服务）
python -m fisheries.api                # 然后打开 http://127.0.0.1:8000/app/

# 第 4 步：页面上选一个内置模板 → 点「用该模板填充」→ 点「运行评估」
```

第 4 步用模板是为了让你**不用先去找数据**。内置 12 个水质场景（见 [§5](#5-内置水质模板)），
覆盖正常 / 预警 / 高危三档，选一个就能看见完整报告。

### 2.3 日常使用：选一个入口

| 你的场景 | 用哪个入口 | 怎么做 |
|---|---|---|
| 想快速看一眼，不想敲命令 | **Web 前端** | 打开 `/app/` → 选模板或上传表格 + 图片 → 点「运行评估」 |
| 要把结果接进自己的系统 | **HTTP 接口** | `POST /api/v1/assess`（见 [§8](#8-http-接口)） |
| 要接传感器做定时巡检 | **HTTP 接口 + 会话缓冲** | 先 `POST /api/v1/ingest/sensor` 追加读数，再 `POST /api/v1/assess` |
| 写脚本批量跑 | **当作库调用** | `from fisheries.agent.orchestrator import ...`（见 [§9](#9-当作库调用)） |
| 只想问知识库，不跑评估 | **`POST /api/v1/chat`** | 直接问答，不经 Agent |

### 2.4 Web 前端有哪几页

页面是一套纯原生 HTML/CSS/JS（无构建步骤、无 CDN、断网可演示），由接口进程同源托管在 `/app`：

| 视图 | 干什么 |
|---|---|
| **总览** | 后端状态、塘口历史风险趋势、演示模式横幅 |
| **智能评估** | 选内置模板或上传表格 + 鱼体照片 → 一键出完整报告（**主入口**） |
| **知识问答** | 直接问知识库，看到召回原文片段与出处 |
| **数据与标准** | 当前生效的阈值表、品种档位、红线，以及两个工具的调用契约 |

页面故意**只读后端**：等级、分数、告警全部来自接口，前端一个都不重算。
页面自身的说明见 [web/README.md](web/README.md)。

---

## 3. 快速开始

### 3.1 环境要求

- **Python ≥ 3.10**（开发与测试环境为 3.13）
- 操作系统不限，命令示例以 Windows PowerShell 为主
- **不需要 GPU、不需要 API key、不需要联网**即可跑通全链路
- 需要联网的只有三件事：首次安装依赖、首次下载中文向量模型（约 91 MB）、按需下载数据集

### 3.2 安装

#### 一键配置（推荐）

下载或克隆本仓库后，**双击 `一键配置环境.bat`** 即可。
直接双击 `启动接口服务.bat` / `运行自检.bat` 也行：第一次发现还没有 `.venv` 时，
它们会先自动跑一遍同一条配置流程再继续启动——所以「下载 → 双击 → 使用」不用先读文档。
它会依次做完七件事，重复运行是安全的（已经做过的步骤会跳过）：

1. 检查 Python 版本（需 ≥ 3.10），挡掉 Windows 商店版的 python 转发器；
2. 建 `.venv`；
3. 升级 pip / setuptools / wheel；
4. 装依赖 `.[dev,rag,api]`；
5. 导入数据集（可选，最大约 1.4 GB；没配 Kaggle 凭证时走国内镜像，见 §3.5）；
6. 构建知识库向量库（**向量库不入库**，所以全新克隆必须做一次），
7. 验证依赖齐全。

最后它会把「下一步双击哪个 .bat、打开哪个地址」打印出来。

也可以走命令行，带参数：

```bash
python scripts/setup_env.py                     # 和双击等价
python scripts/setup_env.py --mirror tsinghua   # 用国内镜像装依赖
python scripts/setup_env.py --skip-kb           # 先不建向量库，只想看看页面
python scripts/setup_env.py --skip-datasets     # 不要数据集（省 1.4 GB 与几分钟）
python scripts/setup_env.py --dataset-mirror https://你的地址/fish-datasets
python scripts/setup_env.py --recreate          # 删掉 .venv 重建
python scripts/setup_env.py --extras api,rag,dev --python "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"   # 换成自己的解释器
```

| 参数 | 作用 |
|---|---|
| `--mirror {official,tsinghua,aliyun,ustc}` | 换 PyPI 源，默认 `official` |
| `--hf-mirror` | 下向量模型时直接用 `hf-mirror.com`（默认源失败会自动切一次） |
| `--skip-kb` | 跳过向量库（知识问答会不可用，其余功能正常） |
| `--rebuild-kb` | 向量库已存在也重建 |
| `--skip-datasets` | 跳过第 5 步，不导数据集 |
| `--dataset-mirror URL` | 数据集从国内镜像下载（写进 `FISHERIES_DATASET_MIRROR`，见 §3.5） |
| `--recreate` | 删掉 `.venv` 重建（环境搞坏了用这个） |
| `--extras` | 自选 extras，默认 `dev,rag,api` |
| `--skip-install` / `--skip-verify` | 只跑其中几步 |

> 第 5 步与第 6 步失败**都不影响环境可用性**，脚本会照常走完并给出补做的命令：
> 数据集缺了就没有 M8 评测与病害图像演示；向量库缺了只有「知识问答」不可用。
> 退出码只看前四步与第 7 步。


> 装依赖失败多半是网络。加 `--mirror tsinghua` 重试；仍然失败就看脚本打印的
> `[失败]` 那一行，它会指出是哪一步。
>
> 第 5 步要下约 91 MB 的中文向量模型。`huggingface.co` 在国内经常直连超时，
> 脚本会在默认源失败后**自动换 `hf-mirror.com` 重试一次**；也可以一开始就加
> `--hf-mirror` 跳过那 40 秒的等待。这一步失败不影响页面与接口，只是「知识问答」
> 不可用，之后可以单独补建。

#### 手动安装（不想用脚本时）

```bash
# 建虚拟环境
python -m venv .venv

# 激活
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # macOS / Linux

# 安装本工程（含测试与 lint 依赖）
python -m pip install -U pip
python -m pip install -e ".[dev]"

# 运行所需的其余依赖（检索 + 接口）
python -m pip install -e ".[rag,api]"
```

按需加装：

| 想要什么 | 命令 |
|---|---|
| Web 前端 + HTTP 接口 | `python -m pip install -e ".[api]"` |
| 图像识别（加载训练好的权重） | `python -m pip install -e ".[vision]"` |
| 自己训练病害识别模型 | 见 [docs/模型训练说明.md](docs/模型训练说明.md)（torch 需指定官方索引） |
| 本地离线回译模型 | `python -m pip install -e ".[translate]"` 后跑 `python scripts/fetch_translate_model.py` |
| 下载数据集 | `python -m pip install -e ".[datasets]"` |

> **只跑工具层、不想装检索依赖？** 设 `FISHERIES_DISABLE_KNOWLEDGE=1`，
> Agent 会跳过检索并降级为「无依据建议」，不会启动失败。

### 3.3 首次运行会下载什么

仓库里**只有代码、语料和一份权重校验文件（`models/checksums.md5`），没有权重本体，也没有数据集**（见 [§13](#13-数据与合规)）。
第一次跑的时候有两件事要留意：

| 产物 | 大小 | 怎么来 |
|---|---|---|
| `models/fastembed/` 中文向量模型 | ≈ 91 MB | **首次检索时自动下载**。离线环境请先联网跑一次，或预先放进该目录 |
| `models/yolov8-fish.pt` 病害识别权重 | ≈ 3 MB | 仓库**不附带**，走外部托管：`python scripts/fetch_datasets.py --fetch-weights fish_disease` 一条命令拉回来并自动核 md5（见 [§3.6](#36-模型权重托管校验与缺失时会发生什么)）。也可以自己训练，见 [docs/模型训练说明.md](docs/模型训练说明.md) |
| `models/yolov8-fish-lesion-seg.pt` 病灶定位权重 | ≈ 6 MB | 同上，`--fetch-weights fish_lesion`；或 `python scripts/train_lesion_seg.py` 训练产出（见 §12.4）。缺失时只是没有病灶数量/面积，病害结论照旧 |
| `models/yolov8-species.pt` 鱼种分类权重 | ≈ 3 MB | 同上，`--fetch-weights fish_species`；或 `python scripts/train_yolo_species.py` 训练产出（见 §12.5）。缺失时鱼种模块直接隐藏，水质按默认品种（鲤科）判定 |
| `models/translate/` 离线回译模型 | ≈ 315 MB | 可选，`python scripts/fetch_translate_model.py` |

**向量库不随仓库分发**（`knowledge/chroma/` 在 `.gitignore` 里）：它是约 2 MB 的
二进制产物，每次构建内容都会变，进仓库只会给每次提交带上一块 review 不了的 delta。
克隆下来**先构建一次**，之后只有语料改动了才需要重建：

```bash
python scripts/build_knowledge_base.py           # 首次构建（会下载向量模型）
python scripts/build_knowledge_base.py --check   # 只校验一致性，不重建
```

### 3.4 两个常用命令（Windows 上双击对应 .bat 即可）

```bash
# ① 起 HTTP 服务 + Web 前端（页面在 /app/，Swagger 在 /docs）
python -m fisheries.api                            # http://127.0.0.1:8000/app/

# ② 环境体检（不开服务，逐项检查依赖、模型、知识库、数据集）
python scripts/selftest.py
```

对应的双击入口是 `启动接口服务.bat` 与 `运行自检.bat`（都是纯 ASCII 的薄壳，
只负责找到 `.venv` 并转发，中文提示都由 Python 打印）。

命令行 demo（每一个都能离线跑，输出是确定性的）：

| 命令 | 看什么 |
|---|---|
| `python scripts/demo_agent.py` | 全链路，6 个场景 |
| `python scripts/demo_tools.py` | 只跑两个工具，6 个场景 |
| `python scripts/demo_cases.py` | 融合层边界用例（等级下限的直接证据） |
| `python scripts/demo_rag.py` | 检索质量，4 个场景，含注入提示词的原文 |
| `python scripts/demo_prompt.py` | 系统提示词与护栏，3 个场景 |
| `python scripts/demo_api.py` | 完整 HTTP 链路，6 个场景（进程内，不用起服务） |
| `python scripts/demo_translate.py` | 回译层，4 个场景 |

### 3.5 数据集：没有 Kaggle 凭证怎么办

`data/datasets/` 里那 1.4 GB 数据**不随仓库分发**（见 [§13](#13-数据与合规)）。
注册表里 10 份可从 Kaggle 拿，另外 2 份（`fish_disease_cleaned`、`fish_species_images`）
本来就没进工程。正常路径是：

```bash
python -m pip install -e ".[datasets]"        # 装 kagglehub
# 凭证放 %USERPROFILE%\.kaggle\kaggle.json
python scripts/fetch_datasets.py --fetch all
```

**没有 Kaggle 账号也不用卡在这里**，数据集可以走自己搭的静态镜像，不需要任何凭证。
`fetch` 的顺序是「先 kaggle，失败自动切镜像」，两条路都不通才会报错，报错里会把
两条路的具体原因和三个办法都列出来。

搭镜像只有三步：

```bash
# ① 找一台已经把数据导好的机器，把 data/datasets 里的目录打成 zip
python scripts/pack_dataset_mirror.py --all          # 产出 dist/mirror/*.zip + manifest.json
python scripts/pack_dataset_mirror.py --name fish_disease --out dist\share

# ② 把 dist/mirror 整个目录传到任意静态托管（对象存储、公司内网、Nginx 都行）
#    上传后应该能直接访问 https://你的地址/fish-datasets/manifest.json

# ③ 换台机器，把地址填进环境变量，然后正常下载
set FISHERIES_DATASET_MIRROR=https://你的地址/fish-datasets
python scripts/fetch_datasets.py --fetch all
# 或者一步到位：一键配置环境.bat --dataset-mirror https://你的地址/fish-datasets
```

几点约定：

* **`manifest.json` 是唯一的校验来源**。里面记了每个 zip 的 `sha256` 与文件数，
  `fetch` 下载完先校验再解压，对不上就丢弃并报错——所以注册表**一行都不用改**。
  想更严可以把 `archive_url` 与 `sha256` 写回 `src/fisheries/datasets.py` 的 spec，
  那样连 manifest 被换掉也拦得住（注册表优先于 manifest）。
* **打包是字节可复现的**：同一份数据重复打包得到同一个 `sha256`（zip 内时间戳固定
  为 1980-01-01），校验值可以放心抄进文档。
* 单个数据集也可以只用本地路径或 `file://` 当源，离线演示够用：
  `set FISHERIES_DATASET_MIRROR=dist\share`（相对工程根目录）。
* 镜像里没有的（比如那 2 份本地本来就没有的），`--fetch` 会明确说「两条路都不通」，
  不会静默跳过。

### 3.6 模型权重：托管、校验与缺失时会发生什么

图像链路要用三个权重：病害识别 `models/yolov8-fish.pt`（≈ 3 MB）、病灶定位
`models/yolov8-fish-lesion-seg.pt`（≈ 7 MB）、鱼种分类 `models/yolov8-species.pt`（≈ 3 MB）。
它们**不提交 git**——二进制大文件会让每次提交都带上一块 review 不了的 delta，而权重只会在重训时变。
仓库里放的是**`models/checksums.md5`**（标准 `md5sum` 格式，三行哈希 + 两行注释），本体走外部托管。

```bash
python scripts/fetch_datasets.py --fetch-weights all            # 三个都拉，下载完自动校验
python scripts/fetch_datasets.py --fetch-weights fish_disease   # 只拉一个
python scripts/fetch_datasets.py --weights                      # 只看状态，不下载
python scripts/fetch_datasets.py --weights --verify             # 重新算一遍哈希
```

镜像地址取 `FISHERIES_WEIGHTS_MIRROR`；没配就回退到数据集的 `FISHERIES_DATASET_MIRROR`
（权重放在该地址下的 `weights/` 子目录）。**一键配置不会自动下权重**（装完还没配镜像时必然失败），
但它会在结尾按「缺哪些 + 一条命令」提示你，`运行自检.bat` 里也会逐项列状态。自己搭镜像：

```bash
# 在一台权重已就绪的机器上，把权重原样复制（不压缩）到 dist/mirror/weights/
python scripts/pack_dataset_mirror.py --weights --skip-datasets --out dist\share
# 把 dist\share\weights\ 整个传上静态托管，然后换台机器指过去
set FISHERIES_WEIGHTS_MIRROR=https://你的地址/fish-weights
python scripts/fetch_datasets.py --fetch-weights all
```

校验是自动的：下载完比对 `checksums.md5` 里的 md5、spec 里的 sha256 与文件大小，对不上
就删掉重下一次，两次都不对才报错；`--verify` 手动复核。离线机器也可以自己核：

```bash
md5sum -c models/checksums.md5                                    # Linux / Git Bash
certutil -hashfile models\yolov8-fish.pt MD5                      # Windows cmd
```

**权重缺失不会静默降级**，会在第一次真正用到时打一条 WARNING（同一条只打一次）：

```text
WARNING fisheries.weights: fish_disease | 权重缺失，将启用 mock 模拟模式：病害识别权重（models/yolov8-fish.pt）
```

三个入口都会说：工具层第一次落兜底时、API 启动的 lifespan 里、`python scripts/selftest.py`
的「检查可选依赖」那一步（逐项列「就绪 / 缺失」，缺失的给获取命令）。

**鱼种识别是唯一例外**：它缺权重时不会 mock 一个品种——猜错品种会直接换错水质阈值档位，
比不显示更糟——所以它的提示是「权重缺失，鱼种识别模块将隐藏」，页面上那一块直接不出现，
水质按默认品种（鲤科）判定，下拉框仍然可以手动选。

#### 申报用：权重清单

| 文件 | 大小（B） | md5 | sha256 | 用途 | 缺失后果 |
|---|---|---|---|---|---|
| `models/yolov8-fish.pt` | 2,980,669 | `9c93b4614b8e879dfcd33d854495b572` | `95a9bb0d62e9032ed515f4d43f61b4733fdf53ae443b6875bdbe17684ba96da7` | 鱼病分类（YOLOv8-cls 微调） | 图像工具落 mock，响应标 `degraded_mode=true` |
| `models/yolov8-fish-lesion-seg.pt` | 7,053,685 | `93ce235e828481c003e1f577ab0f7149` | `d814df5155324b65dc62df3c718a6e43a99ee92f01ce9342a9562e15fa1e4e10` | 病灶分割，给病灶数量与面积 | 病害结论照旧，只是没有位置与面积 |
| `models/yolov8-species.pt` | 2,976,829 | `f8578ae043c7feb0fd5c637e704972db` | `8518796b626b5b29e7355b1d1508011f7a914e0bbda268f2918dbe6532497f79` | 鱼种分类（鲤科 / 罗非鱼 / 虾蟹） | 鱼种模块隐藏、不猜品种，水质按默认档位（鲤科）判定 |

托管位置：`FISHERIES_WEIGHTS_MIRROR` 指向的 `<地址>/weights/`，三个文件同名存放；
校验值同时写在仓库内的 `models/checksums.md5` 里，评委拿到文件后可当场核 md5。

另外两个官方预训练权重（`models/pretrained/yolov8n-cls.pt`、`yolov8n-seg.pt`）只在重训时
用到，`pinned=False`、不钉哈希，`--fetch-weights` 会跳过它们；确实需要时按 spec 里的
`upstream` 从 Ultralytics 官方 release 下载。

---

## 4. 输入数据怎么准备

### 4.1 表格要有什么

**必需列**（四列缺一不可）：

| 列 | 含义 | 单位 |
|---|---|---|
| `water_temp` | 水温 | ℃ |
| `ph` | pH | — |
| `dissolved_oxygen` | 溶解氧 | mg/L |
| `ammonia_n` | 氨氮 | mg/L |

**可选列**：

| 列 | 含义 | 说明 |
|---|---|---|
| `timestamp` | 采样时间 | 强烈建议带上，见 §4.3 |
| `nitrite_n` | 亚硝酸盐 | 缺失时该指标整列剔除、其余权重自动归一化 |

**条数**：至少 **3 条**。24 条以上（一天一点一小时）才能给出可靠的趋势判断；
不足 24 点时报告会明说「趋势偏弱」。

### 4.2 列名怎么写都行

真实数据导出工具的列名千奇百怪，接入层有一张**精确别名表**（不做模糊匹配，
因为 `Total_Phosphorus_mg_L` 里含 `ph`、`Nitrate` 与 `Nitrite` 只差三个字母，模糊匹配会认错列）。
下面这些写法都能识别：

| 指标 | 可识别的列名 |
|---|---|
| 水温 | `water_temp`、`water_temp_c`、`水温`、`水温(C)`、`temperature` |
| pH | `ph`、`pH值`、`酸碱度` |
| 溶解氧 | `dissolved_oxygen`、`dissolved_oxygen_mg_l`、`dissolved oxygen`、`DISOLVED OXYGEN`（拼写错的也认）、`溶氧`、`溶氧(mg/L)` |
| 氨氮 | `ammonia_n`、`ammonia_n_mg_l`、`氨氮`、`氨氮(mg/L)` |
| 亚硝酸盐 | `nitrite_n`、`nitrite_n_mg_l`、`亚硝酸盐`、`亚硝酸盐(mg/L)` |
| 时间 | `timestamp`、`采集时间`、`时间`、`datetime` |

表头大小写不敏感，前后空格会被忽略，带 BOM 的表头也能认。

> ⚠ 温度列**不要**写 `水温(℃)`——单位后缀里 `℃` 不在识别集内，会被当成未知列忽略。
> 写 `水温(C)` 或干脆写 `水温`。

### 4.3 时间戳

- 支持 `2026-07-20 08:00:00`、`2026-07-20T08:00:00`、`2026/07/20 08:00` 等常见写法；
- 带时区偏移的（`2026-07-20T00:00:00Z`、`+08:00`）会**换算到本地时区**，不会被静默抹掉；
- 乱序的会自动重排，并给出提醒；
- **个别行缺时间戳会被按相邻已知点线性内插补齐**，不会推移真实点的时刻。
  但缺得太多（比如只填了首尾两条）趋势外推的可信度会下降，请尽量完整。

### 4.4 单位与坏值

- 单位请按 §4.1 的表填。量纲明显超出物理量程时会给出核对提醒，
  例如溶氧写成 `6000`（看起来像 µg/L）会提示检查列映射。
- 下列值会被**按缺失处理**而不是当成真实读数：
  - 设备哨兵值：`-127`、`-999`、`N/A`、`--`、空字符串；
  - 非有限值：`inf`、`nan`（导出工具算不出数时会写 `inf`，这一格不是空的，会单独提醒你回原始数据核对）；
  - 明显不可能的负值。
- **必填列缺格的整行会被丢弃**并告诉你丢了几行——留个空值会让整份数据在工具层被整体拒收，
  用户看到的会是「上传成功、分析失败」，那比丢几行更难排查。

---

## 5. 内置水质模板

不想准备数据也能立刻试。仓库自带 **12 个水质场景模板**，每个都是 24 小时的完整读数，
放在 `templates/` 下：`<场景名>.csv`。

| 场景 | 养殖品种 | 预期结论 | 看的是什么 |
|---|---|---|---|
| 夏季正常塘 | 鲤科 | 正常 | 基准模板：对比「我的塘口比正常差多少」 |
| 虾蟹塘正常 | 虾蟹 | 正常 | 同一份水换品种档位，结论可能不同（这份数据鲤科会判预警） |
| 黎明前缺氧 | 鲤科 | **高危** | 夜间耗氧，凌晨跌破危险线并触发外推红线 |
| 溶氧破红线 | 鲤科 | **高危** | 泛塘临界，溶氧直接击穿红线 |
| 高温高 pH 氨氮累积 | 鲤科 | **高危** | pH 高、水温高，非离子氨折算后超 0.020 mg/L |
| 氨氮缓慢爬升 | 鲤科 | **高危** | 总氨氮 0.38 没破线，pH 8.25/28.5℃ 折算非离子氨 0.043（限值 2 倍） |
| 亚硝酸盐超标 | 鲤科 | **高危** | 亚硝酸盐趋势快速上行 |
| 午后 pH 偏高 | 鲤科 | **高危** | 碱度大，pH 日变化幅度超限 |
| 冬季冰封期 | 鲤科 | **高危** | 低温 + 冰封，水温长期越界 |
| 溶氧偏低 | 鲤科 | 预警 | 尚未破线但持续低于适宜下限 |
| 暴雨降温后 | 鲤科 | 预警 | 温度与 pH 剧烈波动 |
| 早春水温偏低 | 鲤科 | 预警 | 低温期管理 |

**三种用法**：

```bash
# ① 前端页面：选模板 → 点「用该模板填充」→ 点「运行评估」
#    （点了模板会同时填好数据、并把品种档位切到模板自带的品种）

# ② 接口：先取目录，再取 CSV
curl http://127.0.0.1:8812/api/v1/config/templates
curl -O -J http://127.0.0.1:8812/api/v1/config/templates/dawn_hypoxia/csv

# ③ 直接把 templates/<key>.csv 当普通表格上传或粘贴
```

模板的 12 个 CSV 与 `templates/README.md` 都由代码生成，改了模板定义后重新生成：

```bash
python scripts/build_water_templates.py           # 重新生成
python scripts/build_water_templates.py --check   # 校验磁盘与代码是否一致
python scripts/build_water_templates.py --list    # 只打印目录表
```

> 模板里的「预期结论」是**被单元测试钉住的**：`tests/unit/test_water_templates.py` 会真的调用
> 规则引擎跑一遍每个模板，等级对不上就测试失败。所以模板同时起到了回归用例的作用。

---

## 6. 输出怎么看

一次评估返回的东西分七块，按「先看结论、再看依据」的顺序排：

| # | 内容 | 看什么 |
|---|---|---|
| 顶部 | **风险等级** + 融合分 + 一句话结论 | 先看这个是正常 / 预警 / 高危 |
| ① | **告警** | 按 `critical`（红）/ `warning`（橙）/ `info`（蓝）三档配色 |
| ② | **处置建议** | **水质调控**与**病害处置**两张表分开列，含优先级、对象、动作、时机 |
| ③ | **数据接入质量** | 列名映射、被忽略的列、采样间隔及其来源、缺失格数、量纲提醒 |
| ④ | **图像识别** | 病害类型、置信度、完整概率分布、是否降级模式 |
| ⑤ | **融合依据** | 加权分 / 等级下限 / 协同加成 / 有效权重 / 触发了哪些下限 |
| ⑥ | **知识库引用** | 命中资料按用途分组，每条带标题链与溯源标记 |
| ⑦ | **推理链路** | 五步链路逐条展开，页脚给耗时与降级原因 |

### 6.1 风险等级怎么定的

不是单纯按分数分档，而是取**「分数档位」与若干「等级下限」的最高档**。原因是加权平均
隐含「两个模态在测同一个量、可以互相抵消」，而这里不成立——
`S_水质` 衡量的是**环境危害**（溶氧 1.9，今晚就可能泛塘），`S_图像` 衡量的是**已见病征**。
「水质高危 + 照片上鱼看着还行」不等于风险减半，只是**症状还没出来**。

会抬高等级的下限：

| 下限 | 触发条件 | 结果 |
|---|---|---|
| 红线 | 水质工具命中红线 | 高危 |
| 高致病性病害 | 病害置信度 ≥ 0.70 | 高危 |
| 水质自身判决 | 水质工具判高危 / 预警 | 高危 / 预警 |
| 检出病害 | 非健康 ∧ 严重度 ≥ 0.70 ∧ 置信度 ≥ 0.40 | 预警 |

下限一旦生效，融合分会被抬到对应档线，并把**纯公式值**、**抬升值**、**触发原因**一并返回——
否则会出现「高危 / 32.3 分」这种自相矛盾的报告，按分数筛的调用方会漏报。

此外，**水质与病害同时出问题会额外加成**——单看水质不严重、单看鱼病也不严重，
但同时出现就危险，这正是多源融合的价值所在：

| 触发条件 | 加成 |
|---|---|
| 水质 ≥ 预警 **且** 检出细菌性病害 | +12 |
| 氨氮或亚硝酸盐超警戒 **且** 检出任何病害 | +8 |
| 溶氧低位或预测下降 **且** 检出烂鳃 | +15 |
| 水温 15–25℃ **且** 检出水霉 | +10 |

### 6.2 可信度说明

每份报告的 `confidence_note` 都是固定三句式，由**规则算事实、模板给措辞**：

```
依据：水质 24 点 + 图片检出烂鳃（把握 0.86）。不确定：结论含 12.0 小时外推。
复核：建议尽快补测一次水质。
```

- 「不确定」为空时会明写「无明显不确定项」，「复核」为空时写「无需人工复核」——不留空；
- 同一个事实同时决定两栏（图像分析失败 → 前面说"这份结论没有图像证据"，后面说"请人工重做图像分析"）；
- **同一条结论在「没接大模型」和「接了大模型」两条路径上结构一致**，只有措辞丰富度不同。

---

## 7. 接入真实大模型

### 7.1 为什么拆成两个角色

「决定调哪个工具」是整条链路最不能出错的一步，而**水产领域微调模型通常没做过工具调用训练**。
所以本项目把大模型拆成两个角色，可以用**不同的模型**，互不影响：

- **调度角色**：必须可靠地输出函数调用。推荐中文工具调用成熟的服务（DeepSeek / Qwen 等）。
- **建议角色**：把已经算好的结构化结论翻译成养殖户看得懂的建议。这里适合用领域模型。

### 7.2 配置

全部通过环境变量配置，**代码一行不用改**：

```bash
# 调度角色
export FISHERIES_LLM_SCHEDULER_PROVIDER=deepseek
export FISHERIES_LLM_SCHEDULER_API_KEY=sk-xxxxxxxx

# 建议角色（示例：本地 Ollama 跑的领域模型）
export FISHERIES_LLM_ADVISOR_PROVIDER=aqua-7b-ollama
export FISHERIES_LLM_ADVISOR_BASE_URL=http://localhost:11434/v1
export FISHERIES_LLM_ADVISOR_MODEL=aqua-7b
```

Windows PowerShell 用 `$env:名字 = "值"` 的写法。

部署到 Windows 服务器时不用手改配置：`scripts\deploy_windows.ps1` 支持 `-LlmProvider deepseek -LlmApiKey sk-xxx`（可选 `-LlmModel` / `-LlmBaseUrl`），它会把这几项写进 `run-server.bat` 并持久化到 `C:\fisheries\llm.txt`，重跑部署自动复用（`-NoLlm` 可关掉）。验证方法：`POST /api/v1/chat` 的响应里 `answer_is_raw_retrieval` 从 `true` 变 `false`。

内置预置后端：

| provider | 说明 |
|---|---|
| `mock` | 离线确定性模型（**默认**） |
| `deepseek` | DeepSeek 官方 API |
| `openai` | OpenAI 官方 API |
| `dashscope` | 阿里云百炼（Qwen） |
| `aqua-7b-ollama` / `aqua-7b-vllm` | 本地部署的水产领域模型 |

**没配任何 key 时**自动走 `mock`，整条链路（选工具 → 执行 → 护栏 → 融合 → 检索 → 生成建议）
与接真实模型时完全一致，只是措辞来自规则模板。

### 7.3 中文输出：提示词语言与回译层

部分优秀的水产领域模型是**纯英文微调**的，让它直接写中文质量没有保证。本项目的做法是
**让它用母语（英文）思考，再用一个中文模型把结果整句回译**：

```
水质/图像数据 ──► 英文提示词 ──► 领域模型用英文写建议 ──► 回译层 ──► 中文报告
```

语言按后端自动判定（英文底座走英文，中文模型直接走中文），也可以强制：

```bash
export FISHERIES_PROMPT_LANG=en          # zh / en
export FISHERIES_TRANSLATOR_MODE=auto    # auto / local / glossary / llm / bypass
```

回译层按可用性自动降级，**四级后端**：

| 后端 | 何时启用 | 能力 |
|---|---|---|
| `local` | 有本地离线模型时优先（零 key、离线、确定性） | 整句翻译，分块 + 术语后订正 |
| `llm` | 没有本地模型但有可用的翻译模型 | 整句翻译，质量最好 |
| `glossary` | 兜底（零依赖） | 短句与指标名能直译，自由长句一律拒绝并退回模板 |
| `bypass` | 显式指定 | 不回译，保留英文——只用于排查提示词本身 |

两条硬约束：**① `risk_level` 与 `alerts` 一个字都不过翻译**（它们是护栏要比对的标识符）；
**② 译不出来就退回规则模板的中文表述，绝不保留英文**，且每一次退回都写进 `degraded_reasons`。

> 完整的实测数据、复现命令与已知限制见 [docs/M10_回译层说明.md](docs/M10_回译层说明.md)。

---

## 8. HTTP 接口

```bash
# 方式 A（不需要装包）：--app-dir 让 uvicorn 去 src/ 里找 fisheries 包
python -m uvicorn --app-dir src fisheries.api.app:app --host 127.0.0.1 --port 8812

# 方式 B（已经 pip install -e . 之后）：直接用模块入口
python -m fisheries.api.app

# Swagger 文档： http://127.0.0.1:8812/docs
```

> 注意别直接敲 `python -m uvicorn fisheries.api.app:app`：只有装过 editable 包
> （`pip install -e .`）或设置过 `PYTHONPATH=src` 时才找得到 `fisheries`，否则报
> `ModuleNotFoundError: No module named 'fisheries'`。仓库自带的 `启动接口服务.bat`
> 已经处理好了这两件事。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/v1/health` | 健康检查（含知识库状态、降级原因、各工具后端是否可用） |
| POST | `/api/v1/assess` | **主接口**：水质时序 +（可选）图片 → 风险等级 / 告警 / 建议 / 依据 |
| POST | `/api/v1/tools/call` | 跳过 Agent 直调工具，便于分步演示与调试 |
| POST | `/api/v1/ingest/sensor` | 传感器读数追加到塘口会话缓冲 |
| GET | `/api/v1/ponds/{pond_id}/readings` | 取回缓冲的读数 |
| POST | `/api/v1/ingest/image` | 上传鱼体图片，返回 `upload_id` 与即时识别结果 |
| POST | `/api/v1/chat` | 知识库问答（RAG 检索，不经过 Agent） |
| GET | `/api/v1/ponds/{pond_id}/history` | 该塘口的历史评估记录 |
| GET | `/api/v1/config/thresholds` | 当前生效的阈值表与品种档位（前端画仪表盘用） |
| GET | `/api/v1/config/tools` | 导出函数调用 schema（供前端或其它 Agent 复用） |
| GET | `/api/v1/species` | 鱼种识别是否可用（前端据此显示或隐藏鱼种模块） |
| POST | `/api/v1/species/detect` | 上传鱼体图片判品种档位；不可用时返回 503 并说明原因 |
| GET | `/api/v1/config/templates` | 12 个内置水质模板目录 |
| GET | `/api/v1/config/templates/{key}/csv` | 下载某个模板的 CSV |

最小调用示例：

```bash
curl -X POST http://127.0.0.1:8812/api/v1/assess \
  -H "Content-Type: application/json" \
  -d '{
        "pond_id": "P-A",
        "readings": [
          {"timestamp":"2026-07-20T08:00:00+08:00","water_temp":28.0,"ph":8.05,
           "dissolved_oxygen":6.2,"ammonia_n":0.15,"nitrite_n":0.05},
          {"timestamp":"2026-07-20T09:00:00+08:00","water_temp":28.4,"ph":8.11,
           "dissolved_oxygen":5.8,"ammonia_n":0.17,"nitrite_n":0.06},
          {"timestamp":"2026-07-20T10:00:00+08:00","water_temp":28.9,"ph":8.20,
           "dissolved_oxygen":5.1,"ammonia_n":0.19,"nitrite_n":0.07}
        ]
      }'
```

**接口层的六条设计约束**：

- 接口层不重新实现业务逻辑，只做四件事：校验入参、调用已实现的工具 / Agent、转成稳定契约、把异常转成明确状态码。
- 契约用 `extra="forbid"`——请求里多写一个字段返回 **422 而不是静默忽略**，字段拼错会立刻报错。
- **图片默认只接受 `upload_id`**，不接受服务器任意路径。只有设了 `FISHERIES_API_ALLOW_LOCAL_PATH=1`
  才开放 `image_path`（那是任意文件读取面，公网必须关闭）。
- **会话数据按客户端隔离**：读数缓冲、评估历史、上传记录都带 `session_id`
  （请求头 `X-Session-Id` > cookie `fisheries_session` > 首次访问自动签发；头不合法直接 400），
  落在 `data/uploads/api_state.sqlite3`（`FISHERIES_API_STATE_DB` 可改）。
  两个浏览器用同一个塘号也串不了台，进程重启数据还在。
- **公网部署必须开口令**：设了 `FISHERIES_API_TOKEN` 后，除 `/api/v1/health`、静态页
  `/app/` 与登录/登出路由外，所有接口都要带 `Authorization: Bearer <token>`；
  浏览器首次访问会被 303 跳到 `/login`，登录后下发 `fisheries_auth` cookie（30 天，
  值是口令的 SHA-256 而非口令本身）。没设这个变量=完全开放，只适合本机或内网。
  改口令：`& C:\fisheries\app\scripts\deploy_windows.ps1 -Token "新口令"`——它会同时写进 `C:\fisheries\token.txt`，之后不带 `-Token` 的部署沿用它，浏览器里旧的 `fisheries_auth` cookie 立刻失效（需要重新登录一次）。口令必须是可打印 ASCII；少于 8 位脚本只告警不拦，但公网端口**没有登录限流**，短口令能被暴力猜（见 §12.6）。
- 知识库不可用时 `/assess` 仍返回 200，只在 `degraded_reasons` 里写明「未引用规范依据」；
  只有 `/chat` 本身是知识库功能，会返回 503。

> 要交给前端同学重做页面，看 [docs/前端对接说明.md](docs/前端对接说明.md)：
> 接口清单、字段表、已实现功能清单，以及「哪些东西绝不能由前端自己算」。
> 机器可读的契约在 [docs/openapi.json](docs/openapi.json)，
> 真实响应样例在 [docs/fixtures/_api_samples.json](docs/fixtures/_api_samples.json)。

---

## 9. 当作库调用

不经过 HTTP，直接在 Python 里调两个工具或整条 Agent：

```python
from fisheries.tools.water_risk import water_risk_predict
from fisheries.tools.fish_disease import fish_disease_detect

# 工具① 水质时序风险预测
result = water_risk_predict(
    readings=[
        {"timestamp": "2026-07-20T08:00:00", "water_temp": 28.0, "ph": 8.05,
         "dissolved_oxygen": 6.2, "ammonia_n": 0.15, "nitrite_n": 0.05},
        # … 至少 3 条
    ],
    pond_id="P-A",
    interval_minutes=60,
    horizon_hours=72,
    species="鲤科",          # 鲤科 / 罗非鱼 / 虾蟹
)

print(result["risk_level"], result["risk_score"], result["rule_hits"])

# 工具② 鱼体病害识别
print(fish_disease_detect("photo.jpg"))
```

两个工具的完整签名：

```python
water_risk_predict(readings: list[dict], pond_id: str = "unknown",
                   interval_minutes: int = 60, horizon_hours: int = 72,
                   species: str = "鲤科") -> dict

fish_disease_detect(image_path: str, conf_threshold: float = 0.25,
                    top_k: int = 4, weights_path: str | None = None,
                    force_mock: bool = False) -> dict
```

`fish_disease_detect` 的病害类型是 **烂鳃 / 出血 / 水霉 / 健康**，另有第 5 个安全兜底值
**「其他」**——底层数据集是 7 类（含寄生虫病、病毒性白尾病），强行塞进 4 类等于把病鱼判成健康鱼，
生产环境是安全事故。原始 7 分类判定无损保留在 `raw_class` 字段里。

---

## 10. 工程结构

```
.
├── pyproject.toml
├── README.md                # 本文件
├── 使用说明.txt             # 面向非技术使用者（双击 .bat 之前先看这个）
├── 一键配置环境.bat         # 一键配置环境（建 .venv + 装依赖 + 建向量库 + 验证）
├── 启动接口服务.bat         # 启动接口服务（同时托管 Web 前端）
├── 运行自检.bat             # 环境体检
│                            # ↑ 根目录只留这三个中文入口；缺 .venv 时自动先跑配置
├── docs/                    # 设计与交付文档（见 §14）
│   └── fixtures/            # 探针留档与真实响应样例（离线快照，断网也能看）
├── knowledge/
│   ├── README.md            # 语料可信度声明与组织约定（★ 先读这个）
│   ├── corpus/              # 9 篇种子语料
│   └── chroma/              # 向量库产物（gitignore，构建脚本重建）
├── templates/               # 12 个内置水质场景模板（可重建）
├── web/                     # 前端页面（纯原生 HTML/CSS/JS，由接口同源托管在 /app）
├── examples/
│   └── pond_demo_water.csv  # 演示用 24 小时水质读数
├── data/                    # 运行期产物（不进版本库；datasets/ 是可选的数据集落点）
├── models/                  # 模型权重（不进版本库；只有 checksums.md5 入库，见 §3.6）
├── scripts/                 # 全部 CLI：一键配置 / demo / 建库 / 建模板 / 取数据 / 训练 / 体检
│   ├── build_lesion_dataset.py  # 清洗真实多边形标注 → fish_lesion_seg 数据集
│   ├── train_lesion_seg.py      # 训练 + 评估病灶分割模型（见 §12.4）
│   ├── train_yolo_species.py    # 训练 + 评估鱼种三档分类模型（见 §12.5）
│   ├── pack_dataset_mirror.py   # 把已导好的数据集 / 权重打成可上传的镜像包（见 §3.5、§3.6）
│   ├── sync_test_counts.py      # 文档里的测试数量 = pytest 收集结果（见 §11）
│   ├── bat/                 # 启动器的纯 ASCII 实现（中文入口只做转发）
│   │   ├── setup.bat        #   配置环境（run-*.bat 缺 .venv 时自动调用）
│   │   ├── run-api.bat      #   启动接口服务 + 托管网页
│   │   └── run-selftest.bat #   环境体检
│   └── make_groundtruth_sample.py / evaluate_groundtruth.py  # 人工真值集抽样与评测（见 §12.2）
├── src/fisheries/
│   ├── datasets.py          # 「数据集名字 → 磁盘位置」的唯一真相来源
│   ├── weights.py           # 权重注册表 / md5 校验 / 外托管下载 / 缺权重时的显式提示
│   ├── console.py           # 控制台编码兜底
│   ├── launcher.py          # 启动器要显示的全部中文（.bat 是纯 ASCII）
│   ├── water_templates.py   # 12 个内置模板的定义与 CSV 生成
│   ├── tools/               # 工具层：给大模型调用的「手」
│   ├── agent/               # Agent 层：调度、护栏、融合、回译
│   ├── rag/                 # 检索层：加载 / 分块 / 向量库 / 混合检索
│   ├── api/                 # 接口层：FastAPI 契约、路由与前端静态托管
│   │   └── store.py         #   会话数据（读数 / 历史 / 上传）落 SQLite
│   ├── tables.py            # 表格接入层：列名别名 / 时间轴 / 缺失与量纲
│   └── evaluation/          # 外部验证（只做评测，不参与线上链路）
└── tests/
    ├── unit/                # 单元测试（模型/检索/数据集用替身，见 §11.1）
    └── integration/         # 端到端集成测试（真权重 + 真知识库，见 §11.2）
```

> **为什么根目录只有三个中文名 `.bat`？** `cmd.exe` 按控制台代码页分块解码批处理文件，
> 一个非 ASCII 字节就可能让它把后面的行从中间切入当成命令执行——而**退出码仍是 0、服务照样起来**，
> 只有黑窗口里多出一行噪声。所以根目录的中文名 `.bat` 只负责转发，实际逻辑在纯 ASCII 的
> `scripts/bat/*.bat`（它们先 `cd` 回工程根，发现没有 `.venv` 时自动跑一遍 `setup.bat` 再启动），
> 所有中文提示改由 Python 打印。`tests/unit/test_launchers.py` 会守住这三条约束。

---

## 11. 测试

测试分**两层**：`tests/unit/` 是单元测试（模型、检索、数据集全用替身，秒级到分钟级）；
`tests/integration/` 是端到端集成测试（真权重、真知识库、真路由串起来跑，缺料就 `skip`）。
两层分开跑、分开统计。

```bash
python -m pytest -m "not integration"        # 单元测试：1361 项
python -m pytest tests/integration           # 集成测试：13 项，10 秒内跑完
python -m pytest                             # 两层都跑
python -m pytest tests/unit/test_agent.py -v
```

> ⚠ 统计行被吞时加 `-o addopts=""`：`pyproject.toml` 里已有 `addopts="-q"`，
> 命令行再叠一个 `-q` 会变成 `-qq`，pytest 就不打印汇总行了。

> 上面这些「N 项」不是手写的：`python scripts/sync_test_counts.py` 拿 `pytest --collect-only`
> 的结果重写 README / 使用说明 / 技术方案设计 / 前端对接说明四份文档，`--check` 只比对不落盘
> （`scripts/selftest.py` 第 4 步会调它，数字过期时自检直接报红）。

### 11.1 单元测试（1361 项）

| 测试文件 | 项数 | 覆盖重点 |
|---|---|---|
| `test_tools.py` | 95 | 阈值端点、红线、非离子氨换算（Emerson 公式 vs 表A1/A2）、缺失权重重归一化、NaN 拒绝、混合基线四条约束、图像确定性 |
| `test_fish_species.py` | 42 | 品种别名映射、可用性探测（缺权重 / 缺 ultralytics / 可用）、无权重时不猜、错误码、不在 `TOOL_HANDLERS` 里 |
| `test_agent.py` | 117 | 单模态权重归一化、协同规则、护栏补调、等级下限不被稀释、输出契约 |
| `test_translate.py` | 144 | 语言判定、词表回译、逐字段回译、降级路径、端到端 |
| `test_evaluation.py` | 181 | 评测代码自身的正确性（阈值分档、κ、Spearman、滑窗、量纲归一顺序） |
| `test_rag.py` | 105 | front-matter、分块、RRF 融合、清单一致性守卫 |
| `test_api.py` | 111 | 契约校验、错误状态码、上传校验、并发、鱼种端点（503 / 415 / 422 与 upload_id 复用） |
| `test_api_auth.py` | 32 | 访问口令：未配口令照常开放、配了之后 401 + `WWW-Authenticate`、Bearer 大小写、登录/登出与 cookie（值是口令的 SHA-256）、导航跳转、`next` 只收站内路径 |
| `test_api_rag.py` | 34 | 接口层 × 真实知识库的一致性、溯源标记 |
| `test_tables.py` | 51 | 表格接入、列名别名、时间轴、缺失与量纲 |
| `test_web_contract.py` | 26 | 前端静态托管、`dict.js` 枚举码 vs 后端源码、`api.js` 路径 vs OpenAPI |
| `test_water_templates.py` | 68 | 模板预期等级 vs 规则引擎、CSV 往返、提交物与代码一致 |
| `test_datasets.py` | 87 | 数据集注册表、解析优先级、绝对路径回归守卫 |
| `test_console.py` | 27 | 编码兜底、符号降级、编码锁 |
| `test_launchers.py` | 38 | `.bat` 纯 ASCII 守卫、转发、横幅文案 |
| `test_setup_env.py` | 39 | 一键配置：版本门、extras 与 `pyproject` 一致、镜像与 pip 命令拼装 |
| `test_fish_lesion.py` | 15 | 病灶检测解析、面积占比、门控默认关、下游把「没鱼」当无证据 |
| `test_groundtruth.py` | 13 | 人工真值集：分层抽样可复现、清单往返、非法标注被拒、不足条数只显示待标注 |
| `test_dataset_mirror.py` | 19 | 镜像打包字节可复现、kaggle 失败自动切镜像、sha256 不符拒绝、zip 目录穿越 |
| `test_deploy_script.py` | 27 | 服务器部署脚本的静态守卫：大模型参数还在、启动脚本仍写四行 LLM 环境变量、`llm.txt` 重跑复用、`-Token` 改口令会落盘且拒绝非 ASCII、脚本保持纯 ASCII |
| `test_build_kb_script.py` | 3 | `build_knowledge_base.py --rebuild`：参数真的存在、会删旧库、`--check` 下不删 |
| `test_test_layers.py` | 2 | 分层守卫：`integration` marker 已注册、每个集成模块都带 marker、`conftest.py` 在位 |
| `test_doc_counts.py` | 14 | 文档里的测试数量 == `pytest --collect-only` 的结果、明细合计 == 标题、不再出现会随环境变的「通过/跳过」口径 |
| `test_session_store.py` | 29 | 会话标识的合法性、读数按会话/塘号隔离、`replace`、超上限丢最旧、历史与上传同样隔离、换实例后数据还在（模拟重启）、惰性建库 |
| `test_weights.py` | 42 | 权重注册表与哈希形状、md5/sha256 校验五情形、下载与缓存、坏包拒绝、镜像回退、`mock_notice` 文案与只喊一次、`.gitignore` 放行 `checksums.md5`、命令行三条子命令 |

**数据集缺失不会让测试失败**：对应用例会 `skip`，脚本会打印一行「该数据集未导入」而不是崩掉。

### 11.2 端到端集成测试（13 项）

这层不 mock 任何路由函数，断言的是**真模型 / 真检索 / 真表格解析器串起来之后**的
HTTP 状态码、响应字段和风险标签——单元测试把每一环都换成替身，恰恰测不到这里。

| 测试文件 | 项数 | 覆盖的 P0 链路 |
|---|---|---|
| `test_p0_disease_image.py` | 2 | 上传真实烂鳃图片 → 200 + 契约字段 + `degraded_mode=False` + 类别/置信度/模型版本；同一张图喂 `/assess` → 等级 ≥ 预警、`VISION_*` 证据、病害处置动作带四字段 |
| `test_p0_water_series.py` | 3 | 模板 CSV → 真解析器 → 评估：正常塘维持「正常」并给投影；缺氧模板「高危」+ 跌破 1.5 mg/L 的预计小时数；同一份水虾蟹「正常」而鲤科「预警」 |
| `test_p0_fusion.py` | 2 | 双模态融合分比水质单模态高 10 分以上、`water`/`vision`/`fusion` 三类证据齐全、协同规则有名字（`SYN_LOW_DO_WITH_GILL`）、两种处置动作都在 |
| `test_p0_knowledge_chat.py` | 3 | 答案正文带「【文档 > 章节】」引文、backend 含 chroma + bm25 + 向量维度、chunk 字段齐全且分数降序；溶氧 / 烂鳃查询命中对应文档 |
| `test_p0_species_gate.py` | 3 | 可用性接口显式说明（缺权重时给补法）、图片识别出「鲤科」并带置信度与模型版本、识别出的品种与手选等价且区别于虾蟹 |

**缺料就 `skip`**：没训权重、没装 `ultralytics`、没导数据集、没建知识库索引，集成测试会
跳过并在 reason 里写清补的命令，不会伪装成通过。

### 11.3 代码风格：一条 `ruff check`

`pyproject.toml` 里有 `[tool.ruff]`（`target-version = "py310"`、`line-length = 100`、
`src = ["src", "tests", "scripts"]`），规则集取的是「基础 + 有实害的」那一档：

```bash
python -m ruff check                 # 全仓检查，当前是 All checks passed!
python -m ruff check --fix           # 能安全自动修的自动修（改完照例再跑一遍测试）
python -m ruff check --statistics    # 按规则看还剩多少
```

选中的规则：`E4/E7/E9/F/W`（语法、未定义名、import 用不上）、`I`（import 顺序）、
`UP`（老写法升级）、`B`（bugbear：可变默认值、`zip` 不写 `strict=True` 等）、
`SIM`（能化简的写法）、`RUF`（ruff 自家的坑）、`PERF`（明显的性能写法）、`C4`（推导式）、
`PYI`（`.pyi` 一致性）、`PLW`（pylint 里有实害的几条）、`BLE`（`except Exception` 必须写清理由）。

刻意**没选**的两组：`S`（flake8-bandit）会给 1200 多条测试的每个 `assert` 报「硬编码口令」，
噪声远大于收益；`D`（pydocstyle）会逼每个私有函数补 docstring。
`ignore` 只放 `RUF001/RUF002/RUF003`（中文全角标点与同形字是刻意的），`tests/**` 额外豁免 `B017`。

`BLE001` 不是一律禁止：确实要兜底的 15 处写了 `# noqa: BLE001 - <为什么可以兜>`，
另外 3 处按实际异常类型收窄（`agent/prompts.py` → `ImportError`、
`agent/translate.py` → `(ImportError, AttributeError)`、`scripts/selftest.py` → `ImportError`）。


---

## 12. 已知限制与边界

这一节列的是**当前版本真实的短板**，不是「未来计划」的漂亮话。

### 12.1 趋势外推：好 9.8%，但赚的是平滑不是趋势

在 11 口塘、1,114,970 行真实时序数据上做过一组真值独立的验证：
用过去 18 个点预测 1 小时后的溶氧，当前实现（混合基线）的 **MAE 1.529**，
而「直接用当前值当作预测」的持久性基线是 **1.696**——**优于基线 9.8%**。
三个 history 窗口分别好 9.80% / 9.82% / 9.83%，11 口塘逐塘都不劣于基线。

但这个数字要拆开看，否则会高估引擎：

| 做法（history=18） | MAE | 相对持久性 |
|---|---|---|
| 锚点=最近 3 点均值，斜率修正 ×0.05 | **1.529** | **+9.8%** |
| 只用最近 3 点均值（不修正） | 1.525 | +10.1% |
| 全窗口斜率，α=0.1 | 1.708 | −0.7% |
| 全窗口斜率，α=0.2 | 1.723 | −1.6% |
| 全窗口斜率，α=0.3 | 1.740 | −2.6% |
| 持久性（直接用当前值） | 1.696 | 0 |

结论很直白：**收益几乎全部来自「锚点换成最近 3 点均值」这个平滑动作**，
斜率项在这批数据上贡献约 0.3 个百分点（1.529 vs 1.525），
而任何满量程的线性外推都劣于「什么都不做」——因为溶氧的短期趋势信噪比太低，
外推放大的是噪声。所以阻尼系数取 0.05 而不是 1，且修正量小于 0.02 mg/L 时直接回退纯持久性。

两条刻意保留的分叉：

- **告警路径不走阻尼**：`projected_extreme`（区间极值）与 `hours_to_threshold`
  （还有几小时触线）仍按原始回归速率计算。数值外推可以保守，预警不能变慢。
- **方向判定没有改善**：方向准确率仍是 44.1%，因为它用的是全窗口斜率的符号。
  这块要真做好，得换模型，不是调参数。

`water_risk.py` 的 `_predict_series()` 仍是留给替换的接口点（接 LSTM 只改这一处）。
好消息是**实时超限判定不依赖外推**（溶氧 < 3.0 预警、< 1.5 泛塘、氨氮 > 0.5 高危），
所以这条不影响那部分逻辑。详见 [docs/M8_评测报告.md](docs/M8_评测报告.md)。


### 12.2 两个「准确率」：规则一致性 71.9%，人工标签待标注

**别再把 72.2%（现在是 71.9%）当「准确率」引用。** 这两个指标回答的不是同一个问题，
项目里从这一版起分开写：

| | 规则一致性准确率 | 人工标签准确率 |
|---|---|---|
| 真值 | 离线阈值脚本的判定（与引擎同一套阈值） | **人逐条看读数判的等级** |
| 独立性 | ❌ 不独立 | ✅ 独立 |
| 量的是什么 | 两套实现是否一致——证明**代码没写错** | 引擎判得**对不对** |
| 现在的值 | **71.9%**（3,183 窗口、κ 0.570、漏报 0.0%） | **待标注**（100 条清单已备好） |

**① 规则一致性准确率 71.9%**：在 3,183 个窗口上，引擎的等级与离线阈值脚本逐条比，
一致率 71.9%、κ 0.570、漏报率 0.0%。**但这组标签不是独立真值**——它是按与引擎同一套
阈值现算的，测的是**两套口径（最坏情况决定论 vs 加权评分）的分歧**，不是与事实的符合度。
所以报告里**不出现「准确率 X%、可用于生产」这种话**。

> 氨氮改成按非离子氨换算后，这组数字从 72.2% / κ 0.573 变成 71.9% / κ 0.570：
> 有 7 个窗口被抬到了更高一档（3 个 `正常→预警` 来自 pH 换算后的软性警告，
> 4 个 `预警→高危` 来自非离子氨实测超限）。**漏报率仍是 0.0%**，
> 换来的是一条原先整段沉默的判据——这 0.3 个百分点是知情选择，不是退化。

唯一一条相对独立的旁证：11 口塘的「高危样本占比 vs 鱼体重增长」Spearman ρ = −0.582，
高危占比高的塘鱼确实长得慢——说明标签没有荒谬到反着来。

**② 人工标签准确率：现在是「待标注」**。这是唯一能当准确率引用的数字，做法是：

```bash
python scripts/make_groundtruth_sample.py          # 分层抽 100 个窗口到 data/eval_groundtruth/
# 打开 data/eval_groundtruth/labels.csv，逐条填 human_level（正常/预警/高危）+ annotator
python scripts/evaluate_groundtruth.py             # 标满 20 条才出指标 → docs/_m8_groundtruth.json
```

- 抽样**按溶氧分层、同塘至少隔 24 小时**，固定 `--seed 42` 可复现；
- 数据集与训练集完全隔离（不许放 `data/derived/`、`data/datasets/` 下，脚本会报错）；
- **不足 20 条不出任何指标**，状态记 `pending_labels`，退出码 2——
  宁可显示「待标注」，也不拿 5 条样本算一个漂亮数字；
- 清单里的等级必须由人来填，脚本不替人做判断。这是这个指标唯一诚实的来源。

### 12.3 亚硝酸盐在这份外部数据里是空的

池塘数据只有硝酸盐、没有亚硝酸盐，引擎的一级指标 `nitrite_n` 在这份数据上无法取值，
评测时权重被重新归一化。**真实塘口亚硝酸盐超标的情况，这组评测看不到。**

### 12.4 图像识别：病灶能定位了，但「这不是鱼」还判不了

**病害分类**是 4 分类（烂鳃 / 出血 / 水霉 / 健康），留出集 Top-1 93.0%。

**病灶定位**（新增，`models/yolov8-fish-lesion-seg.pt`）从一批真实多边形标注训练而来：
203 张鱼图去重后 201 张（2 组完全重复已剔除），标注同时含两种格式——404 条实例多边形
（`class x1 y1 … xn yn`）与 30 条轴对齐框（`class cx cy w h`），清洗脚本统一成多边形。
两类：`0: fish` / `1: lesion`。训练 YOLOv8n-seg，80 轮 640px，单卡 RTX 5070 Laptop 约 25 分钟。

| 留出集（38 张 / 82 个实例） | box | mask |
|---|---|---|
| mAP@0.5 | **0.969** | 0.969 |
| mAP@0.5:0.95 | **0.789** | 0.778 |

按类拆：`fish` P 0.967 / R 0.974 / AP50 **0.990**；`lesion` P 0.864 / R 0.907 / AP50 **0.948** /
AP50-95 0.667。真实鱼图上的表现也说得通：`1074.jpg` 病灶占鱼体 8.6%（肉眼可见侧腹溃烂）、
`1076.jpg` 6.0%、`1079.jpg` 3 处合计 45.5%。工具返回 `lesion_count` / `lesion_area_ratio` /
`lesions`（框坐标），界面写「病灶定位：N 处，合计约占鱼体面积 X%」。

**但「这是不是鱼」仍然判不了**，这是这份数据的天花板：201 张训练图**全是鱼**，
一张非鱼负样本都没有，检测器根本没学过「背景长什么样」。实测那张设计海报照样得到
**conf=0.830 的鱼框**，而 38 张真实鱼里最低的也有 **0.815**（中位 0.929）——
**没有任何阈值能把两者分开**。所以门控 `fish_disease_detect(..., enable_gate=True)`
虽然实现了（检测不到鱼框就返回「未检测到鱼」，`is_fish=False`，四个结论字段全空），
但**默认关闭**：打开它只会给出一个假的「确认是鱼」。默认路径下 `is_fish` 恒为 `None`。

也试过两条零成本替代，都不成立：

1. **ImageNet 分类器看原图**：海报「水生动物类概率」合计 0.0023，看着能拦；但对 40 张真实鱼图
   标定后中位数只有 0.34，阈值 0.2 误拒 48%、0.5 误拒 60%；
2. **ImageNet 分类器看检测框裁图**：海报 0.0076，真实鱼 p25 已到 0.0095、中位 0.0914，
   阈值 0.02 就误拒 37%。

两条都只能靠重训，不能靠现成权重。

**要真修，缺的只有数据**：几百张非鱼图片即可——海报、截图、风景、室内都行。放进源目录
（`images/` 里同名文件 + `labels/` 里**空标签文件**；YOLO 里空标签就是「背景图」），然后：

```bash
python scripts/build_lesion_dataset.py --force   # 重新清洗 + 划分
python scripts/train_lesion_seg.py --epochs 80   # 重新训练 + 评估
```

训练完成后把 `enable_gate` 的默认值改回 `True` 就有意义了。数据集怎么清洗、怎么划分、
每张图的 sha256 记在哪里，见 `data/derived/fish_lesion_seg/info.json`。

### 12.5 鱼种识别：只分三个粗档位，100% 是因为任务本身粗

新增 `models/yolov8-species.pt`（YOLOv8n-cls），三分类：**鲤科 / 罗非鱼 / 虾蟹**。
数据来自孟加拉淡水鱼图像集 BFF-15（5,577 张、15 个当地鱼种），按别名归到三个档位：
鲤科 2,569 张（GRASS CARP / KATLA / MRIGAL / MIRROR CARP / PUTI / RUI / SILVER CARP 共 7 类）、
罗非鱼 383 张（数据集拼作 `TELAPIA`）、虾蟹 403 张（`ShrimpImages` 里的健康虾）；
另有 500 张增强虾图**只进训练集**。8:2 划分 → **3,186 训练 / 669 验证**，
60 轮上限、早停于第 39 轮（best 在第 19 轮），验证集 Top-1 **100%**。

这个 100% 不能当成「识别很准」来宣传：三个档位是「银灰侧扁的鲤科 / 罗非鱼 / 带壳的虾」，
差异大到颜色直方图都能分；真正难的是**同档位内认到种**，那需要分种标注，而 BFF-15 的
15 个类里有 7 类不属于本工程的档位，被 `--drop-unmapped` 丢掉了
（TENGRA / PANGAS / KOI / PABDA / SHOL / MAGUR / ILISH，共 2,625 张）。
所以界面只说「鲤科 / 罗非鱼 / 虾蟹」，从不说具体鱼种；置信度低于 0.55 时明说「不确定，
请自己在下拉框里选」，不替用户猜。三个档位正好对上阈值表的三套档案，
「粗」不影响它唯一的用途——决定用哪套水质阈值。

还有两个边界得写明：**蟹类没有训练图像**（找到的蟹类数据集只有 CSV，没有图），
所以「虾蟹」档实际上只有虾；**训练图与验证图同源**，同一条鱼的不同增强可能两边都出现，
100% 里有多少是「记住了原图」无法排除——它只服务粗档位，不值得为它重新组织划分。
真要提升，缺的是分种标注与蟹类图片。

### 12.6 会话缓冲已落 SQLite，但还没做限流

接口层的塘口读数缓冲、历史记录与上传登记都按会话隔离，并落在
`data/uploads/api_state.sqlite3`，**进程重启不丢**；访问口令见 §8（`FISHERIES_API_TOKEN`）。
仍然缺的是**限流**与多副本：单进程 + SQLite 适合一台机器长期跑，不适合横向扩展。

### 12.7 工程不自带数据集

仓库里只有代码与语料。数据集（约 1.37 GB）放在工程目录之外，
`src/fisheries/datasets.py` 是「名字 → 磁盘位置」的唯一真相来源，
查找顺序见 [§13.1](#131-数据集是可选的)。缺失时模块仍然 import 得进去，对应用例 `skip`。

### 12.8 首次运行需要联网下载向量模型

向量库与**向量化模型都不随仓库分发**（模型约 91 MB）。克隆后第一次构建知识库时会
自动下载向量模型；纯离线部署请预先联网跑一次，或把模型预置到 `models/fastembed/`。

### 12.9 换向量化后端必须重建向量库

向量库里记着「这份库是哪个 embedder 建的」，服务只接受与 `FISHERIES_RAG_EMBEDDER` 一致的那份，
不一致会直接报 `StoreMismatchError`（hash 是 256 维伪向量，fastembed 是 512 维语义向量），
所以切换后端时**旧库必须删掉重建**，不能只改环境变量：

```powershell
# Windows 服务器上（部署脚本会先停服务、删旧库、重建、重启）
& C:\fisheries\app\scripts\deploy_windows.ps1 -Embedder fastembed
& C:\fisheries\app\scripts\deploy_windows.ps1 -Embedder fastembed -RebuildKb   # 就是想在原后端上重建
& C:\fisheries\app\scripts\deploy_windows.ps1 -Embedder hash                   # 离线机器退回伪向量
```

脚本把选定的后端写进 `C:\fisheries\embedder.txt` 并同步进 `run-server.bat` 的
`FISHERIES_RAG_EMBEDDER`，重跑部署自动沿用。**不传 `-Embedder` 时沿用 `embedder.txt`，两者都没有就用 `hash`**
——`fastembed` 要下载 91 MB 的 ONNX 模型，国内机器上 `huggingface.co` 与 `hf-mirror.com` 都可能超时
（实测三次重试共 2 分半后失败），所以只在显式要求时才试；要换语义后端就明确写 `-Embedder fastembed`，
并把模型预置到 `C:\fisheries\app\models\fastembed\`（断网可用）。本机（非服务器）等价命令是
`python scripts\setup_env.py --rebuild-kb`，或直接 `python scripts\build_knowledge_base.py --embedder fastembed`。

### 12.10 视觉能力要「运行时 + 权重」两半都有

鱼种识别和病灶分割是**可选**能力，默认部署只装 `dev,rag,api`：`ultralytics` 不在里面，
权重也不进 git（训练产物，见 §13.4）。两半缺任一半，`/api/v1/species/detect` 与
`/api/v1/ingest/image` 都会回 503 并在 `message` 里说清缺什么，前端也会把这两个模块隐藏。

补齐的方法（两条命令，`-Vision` 装运行时，`-WeightsFrom` 拷权重）：

```powershell
# 1) 服务器上：装 ultralytics（会带 torch/torchvision，约 1 GB，清华源几分钟）
& C:\fisheries\app\scripts\deploy_windows.ps1 -Vision

# 2) 同时把本机训练好的权重带上去。
#    先把 RDP 客户端的「本地资源 → 详细信息 → 驱动器」勾上，
#    会话里就能用 \\tsclient\C 或 \\tsclient\D 访问本机磁盘：
& C:\fisheries\app\scripts\deploy_windows.ps1 -Vision -WeightsFrom "\\tsclient\D\设置\项目\Agent\project\models"
```

两个参数都是幂等的，重跑不会重复装；`-WeightsFrom` 只拷 `*.pt` 到 `C:\fisheries\app\models\`，
来源不存在时只打一行 `WARN` 而不是整体失败（很好理解：有人没开驱动器重定向）。
脚本结尾会打印 `VISION     N weight file(s) in …`，N 大于 0 就说明权重到位。

验证（`/api/v1/species` 是公开探测接口，不暴露绝对路径，只回 `models/yolov8-species.pt` 这种相对路径）：

```powershell
curl.exe http://127.0.0.1:8000/api/v1/species     # available=true, reason=""
curl.exe http://127.0.0.1:8000/api/v1/health      # fish_species_detect / fish_disease_detect 不再是 mock
```

权重文件也可以手动拖进远程桌面（驱动器/剪贴板重定向开着时直接复制粘贴即可），
放到 `C:\fisheries\app\models\` 后 `schtasks /End /TN FisheriesAPI; schtasks /Run /TN FisheriesAPI` 重启即可生效
（`/species` 每次请求都重新探测权重，但模型加载在进程内，所以还是要重启）。

---

### 12.11 服务器更新代码：用 SHA 归档，别用 main.zip

`https://ghfast.top/https://github.com/ZaKo333/ProjectFisheries/archive/refs/heads/main.zip`
这类**分支归档会被镜像缓存**：镜像给你一份旧的 `main.zip`，`tar` 照常解包成功，
部署脚本照样打印 `DEPLOY_OK`——于是线上跑的是旧代码，而你以为已经更新了（这个坑真的踩到过：
`/api/v1/species` 里还是绝对路径、`/openapi.json` 里没有新 schema）。

按提交 SHA 取归档就没有这个问题：归档内容对某个 SHA 是**不可变**的，镜像缓存再多也只会给到同一份。

```powershell
# 一行到位：按 SHA 拉代码（脚本会校验解包出的根目录名以短 SHA 结尾）+ 重启
& C:\fisheries\app\scripts\deploy_windows.ps1 -CodeSha <完整 40 位 SHA>
```

第一次从「旧脚本」切到「带 -CodeSha 的新脚本」时，先用显式命令把新脚本取下来：

```powershell
Remove-Item C:\fisheries\src.zip -ErrorAction SilentlyContinue
curl.exe -L --fail -o C:\fisheries\src.zip https://ghfast.top/https://github.com/ZaKo333/ProjectFisheries/archive/<完整 SHA>.zip
tar.exe -xf C:\fisheries\src.zip -C C:\fisheries\app --strip-components 1
```

两个细节：① 先删旧 zip——`curl` 失败而 `tar` 成功，等于把旧包又解了一遍；
② 命令用 `;` 连起来时，`curl` 的失败**不会**阻止后面的 `tar`，务必看 `curl` 的返回码或
直接把两条写成 `if ($LASTEXITCODE -eq 0) { … }`。之后 `-CodeSha` 分支用 `robocopy /E`
覆盖代码（**不用 `/MIR`**，否则 `app\models` 与 `app\data` 会被顺手删掉）。

---

## 13. 数据与合规

### 13.1 数据集是可选的

**仓库里只有代码，没有数据。** 数据集放在工程目录之外，脚本负责找到它们：

```bash
python scripts/fetch_datasets.py --list                  # 注册表里有什么
python scripts/fetch_datasets.py --status                # 本地有哪些（默认动作）
python scripts/fetch_datasets.py --where aquaponics_pond # 只打印路径
python scripts/fetch_datasets.py --fetch aquaponics_pond # 下载缺的（需联网）
```

`--fetch` 走两条路：先 kaggle，失败自动切国内静态镜像，**两条路都不需要同时具备**。
没有 Kaggle 账号时怎么搭自己的镜像见 [§3.5](#35-数据集没有-kaggle-凭证怎么办)。

查找顺序（先命中先赢，每一级都由测试钉住）。下表里 `<工程>` = 本仓库在你机器上的位置，`<工程上级>` = 它的上一级目录：

| # | 位置 | 说明 |
|---|---|---|
| 1 | 环境变量 `FISHERIES_DATA_ROOT` | 指向数据集**总目录**，优先级最高 |
| 2 | `<工程>/data/datasets/<entry>` | 想跟工程放一起就用这个 |
| 3 | `<工程上级>/数据集/<entry>` | 改造前的历史目录，照样找得到 |
| 4 | `data/dataset_index.json` | 逐条手工映射 |
| 5 | `~/.cache/kagglehub/datasets/<owner>/<slug>/versions/*` | kagglehub 缓存，只读不联网 |

**缺数据不会让项目起不来**：`require()` 只在真正用到该数据集时才抛错，且报错消息里直接带下载命令；
`default_path()` 找不到时返回 `None`，所以没挂数据集的机器上模块仍然 import 得进去。

想把数据彻底移出工程：把数据集总目录挪到别处，设 `FISHERIES_DATA_ROOT` 指向它（或写进
`data/dataset_index.json`），代码一行不用改。

数据集清单、病害类别中英对照、时序数据的质量缺陷清单与许可证说明见
[docs/数据集说明.md](docs/数据集说明.md)。

### 13.2 语料与知识库

知识库语料是**自编的种子语料**，不是标准原文的转录——原始数据集里没有任何规范文档。
9 篇语料全部带 `verified: true`，检索结果的提示块会带上依据与核验日期：

```text
【淡水池塘养殖水质指标阈值与分级 > 2. 溶解氧（最关键的指标）】（依据 GB 11607-1989；核验日期 2026-10-15）
```

核验覆盖的范围**只有「引用的标准编号与水质数值是否正确」**，工程分级（警戒线 / 危险线）仍是本项目自定的。
语料的可信度分级与组织约定见 [knowledge/README.md](knowledge/README.md)。

改完语料要重建向量库：

```bash
python scripts/build_knowledge_base.py           # 重建
python scripts/build_knowledge_base.py --check   # 只校验一致性
```

### 13.3 标准与阈值口径

- 水质数值部分对到 **GB 11607-1989《渔业水质标准》**（现行）官方原文；
- **氨氮口径**：传感器输入按**总氨氮（TAN）**处理；标准限的是**非离子氨（NH₃）≤ 0.020 mg/L**
  （表1 序号 19），两者每次运行时按 Emerson 公式换算：
  `占比 = 1 / (1 + 10^(pKa − pH))`、`pKa = 0.09018 + 2729.92 / (T + 273.15)`。
  **告警阈值用的是换算后的非离子氨（毒性氨），不是总氨氮**——
  `> 0.020` 触发红线 `UNIONIZED_AMMONIA_REDLINE`，`0.010 – 0.020` 触发
  `UNIONIZED_AMMONIA_RISK`；总氨氮自己的 0.2 / 0.5 / 1.0 三档是另一条工程分档线。
  公式不是本项目自创：`pKa` 的温度关系式出自 **Emerson 等（1975）**，
  *J. Fish. Res. Board Can.* 32(12): 2379–2383，其数值结果就是 GB 11607-1989
  **附录 A 表 A1**（非离子氨百分比表）；实现与表 A1 / 表 A2 已逐点核过
  （`pH 8.0 / 25℃` 算得 5.37%，表 A2 反算 `0.02 / 0.37 = 5.4%`）；
  最近一次读数对应的换算值在响应里是 `unionized_ammonia`（含 `fraction_pct` /
  `free_ammonia_n` / `limit_ratio` / 公式与出处原文）；
- 部分旧行业标准（如无公害食品系列的养殖用水水质、渔药使用准则）已废止且无直接替代，
  语料里已删除引用，渔药方向改挂现行的《绿色食品 渔药使用准则》；
- **没有名为《淡水养殖水质标准》的现行标准**：淡水养殖用水水质曾经的标准是
  `NY 5051-2001`《无公害食品 淡水养殖用水水质》，已于 **2025-01-17 废止且无直接替代**；
  目前**唯一现行的国家标准仍是 `GB 11607-1989`《渔业水质标准》**，本项目全部水质限值
  与氨氮口径都回指到它（含表1 序号 19 与附录 A 表 A1/A2）；
- **适宜档对照标准，警戒 / 危险 / 红线是在标准之上的工程扩展**，不是标准原文。

> ⚠ 代码中的阈值表为**工程默认值 v0.1**，正式使用前须由有资质的水产技术人员复核。
> 本项目输出**不构成用药指令**。

### 13.4 模型权重同样外托管

权重不上仓库（二进制、每次提交带不可 review 的 delta），但**可获取性受约束**：仓库内的
`models/checksums.md5` 钉住 md5，§3.6 的清单给出三个文件的大小、md5、sha256、托管位置与
下载命令。评委拿到文件后 `md5sum -c models/checksums.md5` 即可确认没被损坏或替换；
拿不到权重时程序会打 `权重缺失，将启用 mock 模拟模式` 并继续跑完其他能力，不会假装有结果
（鱼种权重是例外：缺了就把鱼种模块藏起来，不猜品种）。

---

## 14. 文档索引

| 文档 | 什么时候看 |
|---|---|
| [使用说明.txt](使用说明.txt) | 不想敲命令，想双击 `.bat` 直接跑 |
| [docs/技术方案设计.md](docs/技术方案设计.md) | 想知道每个设计决策的理由（融合定级、等级下限、护栏、检索策略、接口契约） |
| [docs/前端对接说明.md](docs/前端对接说明.md) | 要把界面交给别人重做 |
| [docs/数据集说明.md](docs/数据集说明.md) | 数据集清单、质量缺陷、许可证 |
| [docs/模型训练说明.md](docs/模型训练说明.md) | 复现病害识别模型的训练与评估 |
| [docs/M10_回译层说明.md](docs/M10_回译层说明.md) | 看回译层的实测数据与已知限制 |
| [docs/M8_评测报告.md](docs/M8_评测报告.md) | 看两组外部验证的完整结果（自动生成，勿手改） |
| [knowledge/README.md](knowledge/README.md) | 看语料可信度分级与组织约定 |
| [docs/openapi.json](docs/openapi.json) | 机器可读的接口契约 |
| [docs/fixtures/_api_samples.json](docs/fixtures/_api_samples.json) | 各端点的真实响应样例 |

---

## 许可与致谢

工程代码与自编语料为本项目成果。所引用的第三方数据集、预训练模型与开源库的许可证信息
见 [docs/数据集说明.md](docs/数据集说明.md)。
