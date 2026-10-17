#!/usr/bin/env python
"""下载回译层用的**离线翻译模型**到 ``models/translate/<名字>/``。

为什么要有这个脚本
====================
D12 回译层的收益本来是"让纯英文的 AQUA-7B 产出中文结论"，但离线部署时
没有一个**译得动整句**的翻译模型：词表后端超过 6 个英文词就整句放弃
（见 ``GlossaryTranslator``），于是 AQUA-7B 的输出被整条丢掉、退回规则模板，
D12 只剩名义上的收益。这个脚本负责把"轻量、能离线跑、专门做中英翻译"的
模型权重放进工程，让 :class:`fisheries.agent.translate.LocalTranslator` 有东西可加载。

模型从哪来
============
本机实测 ``huggingface.co`` 不可达、``hf-mirror.com`` 的 API 返回 403，
``modelscope.cn`` 通。所以**首选 ModelScope**，HF 镜像作为备用源。
两个源都是公开直链，不需要 token。

用法::

    python scripts/fetch_translate_model.py --list
    python scripts/fetch_translate_model.py                 # 默认 opus-mt-en-zh
    python scripts/fetch_translate_model.py --model qwen2.5-0.5b-instruct

约定
======
- 目标目录默认 ``<仓库根>/models/translate/<源名>``，与 ``models/yolov8-fish.pt``
  同级；模型不进仓库（``.gitignore`` 已排除 ``models/``），换机器重新跑一次即可。
- 已存在且大小一致的文件**跳过**，所以中断后重跑是安全的。
- 下载失败**不抛栈**，逐文件报告并返回非零退出码。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "models" / "translate"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from fisheries import console  # noqa: E402 - 必须在 sys.path 调整之后

#: 模型登记表：``本地目录名 -> (ModelScope id, HuggingFace id, 一句话说明)``。
#:
#: 只收**小号**模型。回译是每条自由文本字段都要跑一次的路径，塞一个 7B 进去
#: 会让一次演示从 3 秒变成 3 分钟，而译文质量对"养殖户看得懂"这个目标早已过剩。
MODELS: dict[str, tuple[str, str, str]] = {
    "opus-mt-en-zh": (
        "Helsinki-NLP/opus-mt-en-zh",
        "Helsinki-NLP/opus-mt-en-zh",
        "专用英中 NMT（Marian，约 77M 参数）。句子级翻译质量最好、CPU 上最快，"
        "默认选它。它只做翻译、不会自作主张补事实。",
    ),
    "qwen2.5-0.5b-instruct": (
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen 小号通用 instruct 模型（约 0.5B）。译风更像人话、能认领域词，"
        "但它是通用模型，偶发改写数字——数值校验会兜住。",
    ),
}

MS_FILES = "https://www.modelscope.cn/api/v1/models/{mid}/repo/files?Revision=master&Root="
MS_RAW = ("https://www.modelscope.cn/api/v1/models/{mid}/repo"
          "?Revision=master&FilePath={path}")
HF_RAW = "https://hf-mirror.com/{mid}/resolve/main/{path}"

#: 权重文件名按优先级排；命中第一个就算"模型可加载"。
WEIGHT_NAMES = (
    "model.safetensors", "pytorch_model.bin", "model.onnx",
    "onnx/model.onnx", "onnx/model_quantized.onnx",
)
#: 判定"这个目录是一个可用的本地翻译模型"时必须同时具备的东西。
REQUIRED_ANY_WEIGHT = WEIGHT_NAMES
REQUIRED_FILES = ("config.json",)

#: 明显不需要的仓库文件（训练脚本、文档、其他框架的权重）。
SKIP_SUFFIXES = (".md", ".gitattributes", ".txt", ".jsonl")
SKIP_PREFIXES = ("tf_model", "flax_model", "rust_model", "run_", "train")


def _fetch_json(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "projectfisheries-fetch"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def list_modelscope_files(model_id: str) -> list[tuple[str, int]]:
    """列出 ModelScope 仓库里的文件 ``(相对路径, 字节数)``。"""
    data = _fetch_json(MS_FILES.format(mid=model_id))
    files = (data.get("Data") or {}).get("Files") or []
    out: list[tuple[str, int]] = []
    for item in files:
        path = str(item.get("Path") or "").strip()
        if not path or item.get("Type") == "tree":
            continue
        out.append((path, int(item.get("Size") or 0)))
    return out


def wanted(path: str) -> bool:
    """过滤掉训练脚本、说明文档等与推理无关的文件。"""
    name = path.rsplit("/", 1)[-1]
    if name.startswith(SKIP_PREFIXES):
        return False
    return not (name.endswith(SKIP_SUFFIXES) and name not in ("vocab.txt",))


def download(url: str, dest: Path, *, expect: int = 0, force: bool = False) -> str:
    """下单个文件。返回 ``"skip"`` / ``"ok"`` / ``"fail"``。"""
    if dest.exists() and not force:
        if not expect or dest.stat().st_size == expect:
            print(f"  [skip] {dest.relative_to(PROJECT_ROOT)}（已存在）")
            return "skip"
        print(f"  [重下] {dest.name}：本地 {dest.stat().st_size:,} B "
              f"≠ 远端 {expect:,} B")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "projectfisheries-fetch"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
            total = 0
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)
                total += len(block)
        if expect and total != expect:
            print(f"  [fail] {dest.name}：下到 {total:,} B，期望 {expect:,} B")
            tmp.unlink(missing_ok=True)
            return "fail"
        tmp.replace(dest)
        print(f"  [ok]   {dest.relative_to(PROJECT_ROOT)}  {total:,} B")
        return "ok"
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        print(f"  [fail] {dest.name}：{type(exc).__name__}: {exc}")
        tmp.unlink(missing_ok=True)
        return "fail"


def has_weight(target: Path) -> bool:
    return any((target / name).exists() for name in REQUIRED_ANY_WEIGHT)


def usable(target: Path) -> bool:
    """目标目录是否已经是一个能加载的本地模型。"""
    return target.is_dir() and all((target / f).exists() for f in REQUIRED_FILES) \
        and has_weight(target)


def fetch(name: str, dest_root: Path, *, force: bool = False) -> int:
    if name not in MODELS:
        print(f"未知模型 {name!r}；可选：{', '.join(MODELS)}")
        return 2
    ms_id, hf_id, blurb = MODELS[name]
    target = dest_root / name
    print(f"模型 {name}：{blurb}")
    print(f"来源 {ms_id} → {target}")

    try:
        entries = list_modelscope_files(ms_id)
    except Exception as exc:                       # noqa: BLE001 - 网络原因很多
        print(f"  ModelScope 列目录失败（{type(exc).__name__}: {exc}），改用 HF 镜像")
        entries = []

    todo = [(p, s) for p, s in entries if wanted(p)]
    if not todo:
        # 备用源：HF 镜像没有列目录接口，直接按已知文件名抓。
        print("  按已知文件名从 HF 镜像抓取")
        todo = [(p, 0) for p in
                ("config.json", "vocab.json", "tokenizer_config.json",
                 "source.spm", "target.spm", "pytorch_model.bin")
                if not p.startswith("tf_")]

    counts = {"ok": 0, "skip": 0, "fail": 0}
    for path, size in todo:
        url = (MS_RAW.format(mid=ms_id, path=urllib.parse.quote(path))
               if entries else HF_RAW.format(mid=hf_id, path=path))
        counts[download(url, target / path, expect=size, force=force)] += 1

    print(f"完成：新增 {counts['ok']}、跳过 {counts['skip']}、失败 {counts['fail']}")
    if counts["fail"]:
        return 1
    if not usable(target):
        print(f"{console.warn()} {target} 仍不是一个可用模型（缺 config.json 或权重文件）")
        return 1
    print(f"{console.mark(True)} 可用：{target}")
    print("   启用方式：set FISHERIES_TRANSLATOR_MODE=local")
    return 0


def main(argv: list[str] | None = None) -> int:
    console.install()          # 控制台编不出 ✔ / ⚠ 时降级为 ?，而不是抛异常
    parser = argparse.ArgumentParser(description="下载离线翻译模型到 models/translate/")
    parser.add_argument("--model", default="opus-mt-en-zh", help="模型名（见 --list）")
    parser.add_argument("--dest", default=str(DEFAULT_DEST), help="目标根目录")
    parser.add_argument("--force", action="store_true", help="忽略已存在的文件")
    parser.add_argument("--list", action="store_true", help="只列出可选模型")
    args = parser.parse_args(argv)

    if args.list:
        for key, (ms, hf, blurb) in MODELS.items():
            mark = console.mark(usable(Path(args.dest) / key))
            print(f"{mark} {key:24} {blurb}")
            print(f"   {ms}  |  {hf}")
        return 0
    return fetch(args.model, Path(args.dest), force=args.force)


if __name__ == "__main__":                          # pragma: no cover
    sys.exit(main())
