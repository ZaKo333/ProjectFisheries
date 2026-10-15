# 环境探测脚本（一次性诊断用）

这些脚本不是项目运行链路的一部分，是**开发期用来确认第三方库真实行为**的诊断工具。
保留它们的原因：技术方案里的若干结论（量纲、切分块数、向量模型加载耗时、
LangChain API 形态）都是靠它们实测出来的，评委或队友可以复跑核对。

| 脚本 | 用途 | 当时得出的结论 |
|---|---|---|
| `_probe_chunking.py` | 打印 8 篇语料的切分统计（块数、字数分布、含表格块数、标题链抽样） | 语料改动用它先看块数变化，再决定要不要重建向量库 |
| `_probe_fastembed_cache.py` | 验证 fastembed 能否从**项目内**缓存离线加载模型 | 项目内缓存加载 0.18 秒，维度 512，中文检索排序正确 |
| `_probe_fastembed_cfg.py` | 读取 fastembed 支持的模型清单与模型描述 | `bge-small-zh-v1.5` 的 `query_prefix` / `passage_prefix` 均为空，**不需要加指令前缀** |
| `_probe_rag_deps.py` | 打印 RAG 相关依赖的安装状态与版本 | langchain / chromadb / fastembed / jieba / rank_bm25 是否齐备 |
| `_probe_webui_callback.py` | 直调 Gradio 真正的回调 `app._run(...)`（演示脚本走的是 `WebSession.run`，绕过了按钮绑的函数） | 8 个输出的个数与顺序必须和 `outputs=` 严格一致；三条错误路径都返回完整 8 元组；组件数 42 |
| `_probe_d12.py` | 打印 D12 的语言判定与回译后端（含环境变量覆盖、各 provider 的默认语言） | 未知后端默认中文；`aqua-7b-*` 判英文；`FISHERIES_PROMPT_LANG` 可覆盖 |
| `_probe_confidence.py` | 把「可信度说明」在四种输入 × 中英两语下的成品句打印出来 | 三句式齐全、"不确定"为空时明说、标点不重复（`etc..` / `。。`） |

输出留档在 `docs/_probe_*.txt`，便于评审时不用重跑就能核对上述结论。

运行方式（在 `project/` 目录下）：

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe scripts\probes\_probe_chunking.py
```

> 想看**正式**的切分与建库报告，用 `scripts/build_knowledge_base.py --dry-run`；
> 想看检索质量，用 `scripts/demo_rag.py`。这两个才是交付物。
