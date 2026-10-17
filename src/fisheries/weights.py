"""模型权重的清单、下载与校验。

**权重不进 git，但「权重是什么」必须进 git。** 仓库里存不下几个 GB 的二进制
（`models/` 整体在 `.gitignore` 里），所以这里换一种记法：

* 本模块是**权重名单的唯一真相来源**：每个权重干什么、多大、md5 与 sha256 是多少、
  缺了以后系统会退化成什么样。
* `models/checksums.md5` 是给**人**看的同内容副本（标准 md5sum 格式，评委可以用
  `certutil -hashfile` 或 `md5sum -c` 自己核），由 :func:`render_checksums` 生成，
  `tests/unit/test_weights.py` 会钉住两边一致。
* :func:`fetch` 从镜像站下载并**同时校验 md5 与 sha256**，任一不符就删掉半成品并报错。

下载地址只由一个环境变量决定，默认复用数据集的镜像站：

    set FISHERIES_WEIGHTS_MIRROR=https://你的地址/fish-datasets/weights

**缺权重的行为是「明确降级」，不是静默 mock**：:func:`warn_mock` 会打一行
「权重缺失，将启用 mock 模拟模式」，工具层、接口启动、自检脚本都会调用它。
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

#: 给人核的校验文件（**提交进 git**，`.gitignore` 里给了例外）。
CHECKSUMS_FILE = MODELS_DIR / "checksums.md5"

#: 权重镜像站根地址；没配时退回数据集镜像站（同一个站点两种资产）。
MIRROR_BASE_ENV = "FISHERIES_WEIGHTS_MIRROR"
FALLBACK_MIRROR_ENV = "FISHERIES_DATASET_MIRROR"

#: 镜像站里权重放在这个子目录下。
MIRROR_SUBDIR = "weights"

#: 降级时打印的那句话。工具层、接口、自检共用一个措辞，便于 grep。
MOCK_NOTICE = "权重缺失，将启用 mock 模拟模式"

_CHUNK = 1 << 20
_LOG = logging.getLogger("fisheries.weights")
_WARNED: set[str] = set()


# --------------------------------------------------------------------------- #
# 一、清单
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class WeightSpec:
    """一个权重文件的档案。"""

    name: str
    file: str
    label: str
    purpose: str
    fallback: str
    md5: str = ""
    sha256: str = ""
    size: int = 0
    upstream: str = ""
    pinned: bool = True
    #: 缺失时怎么提示。留空 = 走 :data:`MOCK_NOTICE`（真的退化成 mock）；
    #: 少数模块缺权重时**不做 mock**，而是把自己藏起来（鱼种识别就是这样：
    #: 硬猜一个品种只会换错阈值档位，比不显示更糟），那就把话说清楚。
    missing_notice: str = ""

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / self.file

    @property
    def filename(self) -> str:
        return Path(self.file).name

    @property
    def hosted(self) -> bool:
        """是不是本项目自己训出来的（需要我们自己托管）。"""
        return not self.upstream

    def status(self) -> dict[str, Any]:
        exists = self.path.is_file()
        actual = self.path.stat().st_size if exists else 0
        return {
            "name": self.name,
            "file": self.file,
            "label": self.label,
            "exists": exists,
            "bytes": actual,
            "expected_bytes": self.size,
            "size_ok": exists and (not self.size or actual == self.size),
            "pinned": self.pinned,
        }


WEIGHTS: dict[str, WeightSpec] = {
    "fish_disease": WeightSpec(
        name="fish_disease",
        file="models/yolov8-fish.pt",
        label="鱼病四分类权重",
        purpose="回答「是哪一种病」（烂鳃 / 出血 / 水霉 / 健康）",
        fallback="病害结论退化为 mock（输出带 degraded_mode=true），其余功能不受影响",
        md5="9c93b4614b8e879dfcd33d854495b572",
        sha256="95a9bb0d62e9032ed515f4d43f61b4733fdf53ae443b6875bdbe17684ba96da7",
        size=2980669,
    ),
    "fish_lesion": WeightSpec(
        name="fish_lesion",
        file="models/yolov8-fish-lesion-seg.pt",
        label="病灶分割权重",
        purpose="定位病灶位置与面积（鱼体框 + lesion 掩膜）",
        fallback="没有病灶数量与面积占比，病害分类结论照旧",
        md5="93ce235e828481c003e1f577ab0f7149",
        sha256="d814df5155324b65dc62df3c718a6e43a99ee92f01ce9342a9562e15fa1e4e10",
        size=7053685,
    ),
    "fish_species": WeightSpec(
        name="fish_species",
        file="models/yolov8-species.pt",
        label="鱼种分类权重",
        purpose="从鱼体图片判断品种档位（鲤科 / 罗非鱼 / 虾蟹），决定水质阈值",
        fallback="鱼种模块隐藏，水质按默认品种（鲤科）判定；不猜测替代结果",
        md5="f8578ae043c7feb0fd5c637e704972db",
        sha256="8518796b626b5b29e7355b1d1508011f7a914e0bbda268f2918dbe6532497f79",
        size=2976829,
        missing_notice="权重缺失，鱼种识别模块将隐藏",
    ),
    "yolov8n-cls": WeightSpec(
        name="yolov8n-cls",
        file="models/pretrained/yolov8n-cls.pt",
        label="YOLOv8n 分类预训练权重（官方）",
        purpose="重训病害分类模型的起点（`scripts/train_yolo_fish.py`）",
        fallback="只是不能重训；已有模型仍可推理",
        upstream="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt",
        pinned=False,
    ),
    "yolov8n-seg": WeightSpec(
        name="yolov8n-seg",
        file="models/pretrained/yolov8n-seg.pt",
        label="YOLOv8n 分割预训练权重（官方）",
        purpose="重训病灶分割模型的起点（`scripts/train_lesion_seg.py`）",
        fallback="只是不能重训；已有模型仍可推理",
        upstream="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt",
        pinned=False,
    ),
}


def spec(name: str) -> WeightSpec:
    try:
        return WEIGHTS[name]
    except KeyError:
        raise KeyError(
            f"未知权重 {name!r}；已知的：{'、'.join(sorted(WEIGHTS))}"
        ) from None


def find(name: str) -> Path | None:
    """本地已就绪就返回绝对路径，否则 ``None``。"""
    path = spec(name).path
    return path if path.is_file() else None


def local_path(name: str) -> Path:
    return spec(name).path


def missing(names: list[str] | tuple[str, ...] | None = None) -> list[str]:
    return [n for n in (names or WEIGHTS) if find(n) is None]


def status(names: list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    return [spec(n).status() for n in (names or WEIGHTS)]


# --------------------------------------------------------------------------- #
# 二、校验
# --------------------------------------------------------------------------- #

def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_of(path: Path) -> str:
    return _digest(path, "md5")


def sha256_of(path: Path) -> str:
    return _digest(path, "sha256")


def verify(name: str) -> tuple[bool, str]:
    """校验本地文件，返回 ``(是否通过, 说明)``。

    没钉哈希的条目（官方预训练权重，上游会随版本换）只核对存在与大小。
    """
    item = spec(name)
    path = item.path
    if not path.is_file():
        return False, f"文件不存在：{item.file}"
    size = path.stat().st_size
    if item.size and size != item.size:
        return False, f"大小不符：期望 {item.size}，实际 {size}"
    if not item.pinned:
        return True, "大小正确（该条目未钉哈希）"
    actual_md5 = md5_of(path)
    if item.md5 and actual_md5 != item.md5:
        return False, f"md5 不符：期望 {item.md5}，实际 {actual_md5}"
    actual_sha = sha256_of(path)
    if item.sha256 and actual_sha != item.sha256:
        return False, f"sha256 不符：期望 {item.sha256}，实际 {actual_sha}"
    return True, "md5 与 sha256 均通过"


def read_checksums(path: Path | None = None) -> dict[str, str]:
    """读 ``models/checksums.md5``，返回 ``{相对路径: md5}``。"""
    target = path or CHECKSUMS_FILE
    if not target.is_file():
        return {}
    table: dict[str, str] = {}
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            table[parts[-1]] = parts[0]
    return table


def render_checksums() -> str:
    """生成 ``models/checksums.md5`` 的内容（标准 md5sum 格式，LF 行尾）。"""
    lines = [
        "# 模型权重的 md5 清单，由 fisheries.weights.render_checksums() 生成。",
        "# 权重文件本身不进 git（models/ 在 .gitignore 里），这张表进 git，",
        "# 用 `certutil -hashfile <文件> MD5` 或 `md5sum -c models/checksums.md5` 核对。",
        "# 重新生成：python scripts/fetch_datasets.py --write-checksums",
    ]
    for name in sorted(WEIGHTS):
        item = WEIGHTS[name]
        if item.pinned and item.md5:
            lines.append(f"{item.md5}  {item.file}")
        else:
            lines.append(f"# {item.file}  —— 官方上游权重，哈希随版本变化，不在此钉")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 三、下载
# --------------------------------------------------------------------------- #

class WeightError(RuntimeError):
    """权重相关的可预期错误（下载失败、校验不过）。"""


class WeightNotFoundError(WeightError):
    """没有可用的来源，或者两条来源都失败了。"""


def mirror_base() -> str:
    """镜像站根地址；两个环境变量都没配时返回空串。"""
    for key in (MIRROR_BASE_ENV, FALLBACK_MIRROR_ENV):
        value = os.environ.get(key, "").strip()
        if value:
            return value.rstrip("/")
    return ""


def url_for(name: str, base: str | None = None) -> str | None:
    """这个权重应该从哪下：先镜像，再上游官方地址。"""
    root = mirror_base() if base is None else base.rstrip("/")
    if root:
        return f"{root}/{MIRROR_SUBDIR}/{spec(name).filename}"
    return spec(name).upstream or None


def _is_local_source(url: str) -> bool:
    return url.startswith("file://") or "://" not in url


def _local_path_of(url: str) -> Path:
    if url.startswith("file://"):
        from urllib.request import url2pathname

        return Path(url2pathname(url[len("file://"):]))
    return Path(url)


def download(url: str, dest: Path) -> str:
    """把 ``url`` 下到 ``dest``（先写 ``.part`` 再改名），返回实际用的地址。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    part.unlink(missing_ok=True)
    if _is_local_source(url):
        source = _local_path_of(url)
        if not source.is_file():
            raise WeightNotFoundError(f"本地源不存在：{source}")
        shutil.copyfile(source, part)
    else:
        import urllib.request

        try:
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                part.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle, _CHUNK * 8)
        except Exception as exc:
            part.unlink(missing_ok=True)
            raise WeightNotFoundError(f"下载失败（{url}）：{exc}") from exc
    part.replace(dest)
    return url


def fetch(
    name: str,
    *,
    url: str | None = None,
    force: bool = False,
    base: str | None = None,
) -> Path:
    """下载并校验一个权重。

    Args:
        name: :data:`WEIGHTS` 里的键。
        url: 显式指定来源（跳过镜像/上游推导）。
        force: 已存在也重下。
        base: 显式指定镜像站根地址。

    Raises:
        WeightNotFoundError: 没有来源、下载失败、或校验不过。
    """
    item = spec(name)
    existing = item.path
    if existing.is_file() and not force:
        ok, detail = verify(name)
        if ok:
            return existing
        # 已有文件但校验不过：不静默使用，重下覆盖。
        _LOG.warning("权重校验不通过，将重新下载：%s（%s）", item.file, detail)

    source = url or url_for(name, base)
    if not source:
        raise WeightNotFoundError(
            f"{item.label}（{name}）没有可下载的来源：\n"
            f"  设环境变量 {MIRROR_BASE_ENV}=<镜像站根地址>，或直接用 --url 指定。\n"
            f"  镜像包用 python scripts/pack_dataset_mirror.py --weights 生成。"
        )

    download(source, existing)
    ok, detail = verify(name)
    if not ok:
        existing.unlink(missing_ok=True)
        raise WeightNotFoundError(
            f"{item.label}（{name}）下载完成但校验不过，已删除：{detail}\n"
            f"  来源：{source}"
        )
    return existing


def fetch_all(
    names: list[str] | tuple[str, ...] | None = None,
    *,
    force: bool = False,
) -> dict[str, str]:
    """批量下载，返回 ``{名字: 结果}``（成功是路径，失败是错误文本）。"""
    results: dict[str, str] = {}
    for name in names or WEIGHTS:
        try:
            results[name] = str(fetch(name, force=force))
        except WeightError as exc:  # noqa: PERF203 - 每个权重的失败要单独记账
            results[name] = f"失败：{' '.join(str(exc).split())}"
    return results


# --------------------------------------------------------------------------- #
# 四、降级提示（不许静默）
# --------------------------------------------------------------------------- #

def mock_notice(name: str | None = None, reason: str = "") -> str:
    """生成那句显式的降级提示。

    ``name`` 为 None 时只返回固定文案（可带一句原因），用于「注册表里没有这一项」
    或者调用方只知道「权重缺失」而不知道具体是哪个文件的场合。
    """
    if name is None:
        return MOCK_NOTICE + (f"：{reason}" if reason else "")
    item = spec(name)
    head = item.missing_notice or f"{MOCK_NOTICE}：{item.label}（{item.file}）"
    text = head
    if reason:
        text += f"，原因：{reason}"
    text += f"。{item.fallback}。"
    return text + f" 获取方式：python scripts/fetch_datasets.py --fetch-weights {name}"


def warn_once(component: str, text: str) -> str:
    """同一个 component 只打一次的 WARNING，返回原文本。

    「只打一次」是因为 :func:`fish_disease_detect` 可能被调用几百次，每张图刷一遍
    警告等于没警告。
    """
    if component not in _WARNED:
        _WARNED.add(component)
        _LOG.warning("%s | %s", component, text)
    return text


def warn_mock(component: str, name: str | None = None, reason: str = "") -> str:
    """打一次「降级到 mock」的警告（同一 component 只打一次），返回那句话。"""
    return warn_once(component, mock_notice(name, reason))


def reset_warnings() -> None:
    """测试用：清掉「已警告过」的记忆。"""
    _WARNED.clear()


__all__ = [
    "CHECKSUMS_FILE",
    "FALLBACK_MIRROR_ENV",
    "MIRROR_BASE_ENV",
    "MIRROR_SUBDIR",
    "MOCK_NOTICE",
    "MODELS_DIR",
    "PROJECT_ROOT",
    "WEIGHTS",
    "WeightError",
    "WeightNotFoundError",
    "WeightSpec",
    "download",
    "fetch",
    "fetch_all",
    "find",
    "local_path",
    "md5_of",
    "mirror_base",
    "missing",
    "mock_notice",
    "read_checksums",
    "render_checksums",
    "reset_warnings",
    "sha256_of",
    "spec",
    "status",
    "url_for",
    "verify",
    "warn_mock",
    "warn_once",
]
