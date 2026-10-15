# 水产养殖智能监测与决策 Agent · 前端

把「水质传感器时序 + 鱼体图片」两路输入变成一份面向养殖户的中文处置报告。

纯原生 HTML / CSS / JS，**零依赖、无构建步骤、断网可用**。它由后端进程同源托管，
不需要另外起静态服务，也不需要配 CORS 或基地址。

---

## 一、怎么跑

只需要一条命令 —— 页面和接口是**同一个服务**：

```bat
双击「启动接口服务.bat」
```

或者手工：

```bash
python -m fisheries.api          # 默认 http://127.0.0.1:8000
```

然后打开横幅里给出的地址：

```
http://127.0.0.1:8000/app/       ← 前端页面
http://127.0.0.1:8000/docs       ← Swagger
```

`http://127.0.0.1:8000/` 本身返回的是机器可读的服务目录 JSON（调用方和评测脚本要的是
JSON，不是 HTML），所以前端特意挂在 `/app` 而不是根路径。

**为什么同源托管**：跨源请求会被演示机上的代理和安全软件拦掉预检；同源之后不用配 CORS、
不用手改基地址、不用同时开两个终端。少一个环节就少一个现场故障点。

### 也支持直接双击打开

双击 `web/index.html`（`file://`）也能用，此时页面默认连 `http://127.0.0.1:8000`。
后端确实在别的机器或别的端口时，点左下角「设置」填地址即可。留空表示与本页面同源。

---

## 二、目录结构

```
web/
├── index.html          单页应用入口（4 个视图走 hash 路由）
├── css/app.css         全部样式（风险等级配色、图表、时间轴、模态框）
└── js/
    ├── dict.js         枚举字典：规则码 / 红线码 / 融合层 code / 病害标签 / trace kind / 错误码
    ├── charts.js       零依赖 SVG 图表：趋势折线(含外推虚线)、仪表盘、条形图、历史趋势
    ├── markdown.js     轻量 Markdown 渲染（先转义 HTML 再套 Markdown，安全）
    ├── parse.js        CSV/TSV/XLSX/粘贴文本解析 + 列名模糊识别 + 前端预校验
    ├── ui.js           通用渲染 + 完整评估报告渲染器（不吃 null/undefined）
    ├── api.js          接口客户端：同源基地址、normalizeError(三种 detail)、超时
    └── app.js          路由 + 四个页面 + 交互逻辑
```

加载顺序即依赖顺序（普通 `<script>`，不是 module），改动顺序会直接报错。

---

## 三、页面结构

| 页面 | 路由 | 要点 |
|---|---|---|
| 总览（默认） | `#/home` | 后端状态卡（真实/模拟工具、知识库、服务端提示）、历史风险趋势（纵轴 0–100、40/70 线）、演示模式横幅 |
| 智能评估 | `#/evaluate` | **水质 + 病害 + 融合三合一**：内置模板一键填充 / 上传 CSV / 粘贴文本 + 鱼体图片 + 补充问题，点「运行评估」出完整报告（趋势/仪表盘/严重度/规则命中/溶氧 ETA + 病害识别/置信度 + 告警/动作/融合算式/证据链/trace/引用/知识块/原始 JSON） |
| 知识问答 | `#/qa` | RAG 问答、k 滑块、分类/病害筛选、检索原文拼接标注、引用由 chunks 拼出 |
| 数据与标准 | `#/standards` | 品种阈值档案、权重、红线、action_thresholds、免责声明、工具契约 |

---

## 四、内置水质模板

「智能评估」页顶部有一个模板下拉框，数据来自 `GET /api/v1/config/templates`：

- **12 个场景**，覆盖正常 / 预警 / 高危三档，品种横跨鲤科与虾蟹。
- 点「用该模板填充」会一次做完三件事：拉取该模板的 CSV（`.../{key}/csv`）、填进粘贴框、
  **把品种下拉切到模板自带的品种**。
  只换数据不换品种等于拿虾蟹塘的数字按鲤科的档位判级，结果会差一档。
- 每个模板都带 `expected_risk`，`tests/unit/test_water_templates.py` 会真的调
  `water_risk_predict` 跑一遍，把模板数据和规则引擎同时钉住。

模板故意用了 5 种表头风格（`water_temp` / `water_temp_c` / `TEMPERATURE` / `水温` /
`水温(C)`），用来证明列名别名层是有效的 —— 前端 `parse.js` 与服务端 `fisheries/tables.py`
各有一份别名表，**改一处必须同步另一处**。

---

## 五、已遵守的硬约束

- **等级/分数只读后端**：`risk_level`、`fused_score` 直接展示，前端不重算、不自行定级；
  融合算式只做排版，`level_floors` / `score_floor` 被抬升时明示。
  `tests/unit/test_web_contract.py` 会扫描前端源码，确保没有 `risk_level =` / `fused_score =` 这类赋值。
- **三种 detail 形状**统一归一化（字符串 / 数组 / 对象），全部中文提示，不白屏。
- **不显示服务器路径**：图片用本地 objectURL，`stored_path` 从不渲染。
- **演示模式如实披露**：`degraded_mode=true` 顶部挂横幅，`llm_backends` 一并展示。
- **告警为空不等于正常**：`alerts=[]` 时显示「本次没有触发告警」，风险等级独立可见。
- **水质/病害动作分开**：`report.water_actions` 与 `report.disease_actions` 分两栏。
- **空值兜底**：`water` / `vision` / `report` 为 null、缺字段一律显示「—」或隐藏区块，
  绝不出现 `undefined` / `NaN` / `[object Object]`。
- **引用「未核对标准原文」警示色**；`answer_is_raw_retrieval=true` 标注原文拼接。
- **不做任何用药/准确率承诺**：免责声明原样显示，外推明确标注「仅 12 小时有参考意义」。

---

## 六、契约测试

前端没有构建步骤，也就没有类型系统替我们检查「前端认的枚举码」和「后端真会发的码」
是否一致。曾经就是这么错的：`dict.js` 的 `REDLINES` 里列了两个后端永远不发的键
（`PH_LOW_REDLINE` / `PH_HIGH_REDLINE` —— 它们只是后端阈值表的字典键），同时漏了三个真会发
的码，结果那三个码在界面上直接显示英文。

`tests/unit/test_web_contract.py` 现在把这些关系钉住，真值**取自后端源码**而不是在前端再抄一份：

- `REDLINES` 的键 == `water_risk.py::_check_redlines` 里 `hits.append(...)` 的全部码；
- `RULE_HITS` == 红线码 ∪ `_check_normal_rules` 的全部码；
- `TRACE_KINDS` == `TraceStep.kind` 的 `Literal`；
- `DISEASE_GROUPS` == `fish_disease.DISEASE_LABELS`；
- `TOOL_ERRORS` 覆盖图像工具、水质工具、调度层三层错误码；
- `api.js` 调的每个路径都在 OpenAPI 里；
- `/app/` 能取到页面与全部静态资源，不存在的资源 404，且静态挂载没有把 `/docs` 与 `/api/v1/*` 吃掉。

跑法：

```bash
python -m pytest tests/unit/test_web_contract.py -q
```
