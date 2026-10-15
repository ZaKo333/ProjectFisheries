"""数据集注册表与路径解析——把「数据在哪」从代码里彻底抽出来。

## 为什么需要这个模块

在此之前，数据集路径是**硬编码的绝对路径**，散落在至少 13 个文件里：

    <工程上级>\\数据集\\Fish Disease Dataset\\Test
    <工程上级>\\数据集\\Sensor Based Aquaponics Fish Pond Datasets
    <工程上级>\\数据集\\Dissolved oxygen prediction in river water

这带来三个问题：

1. **换台机器就全废。** 盘符不是 D:、目录不叫「数据集」，代码直接崩。
2. **工程被数据绑架。** 数据集 1.37 GB / 2.6 万个文件，和源码混在一起，
   既没法进版本控制，也没法打包分发。
3. **同一份数据存在多个副本。** `Fish Disease Dataset`（179 MB）与
   `Freshwater Fish Disease Aquaculture in south asia`（24.8 MB）内容高度重叠，
   `archive`（1.07 GB）又是一份 PNG 原图。没人知道哪份是哪份。

本模块把这三件事一次性解决：**数据集的「名字 → 位置」只在这里定义一次**，
其余代码一律通过 `require()` / `resolve()` 取路径。

## 解析顺序（先命中先返回）

1. 环境变量 ``FISHERIES_DATA_ROOT``——指向一个总目录，每个数据集是它下面的
   一个子路径（见 ``DatasetSpec.entry``）。部署时最常用这一条。
2. 索引文件 ``data/dataset_index.json``——``fetch`` 下载成功后自动记录，
   把 kagglehub 缓存里的绝对路径钉下来，避免每次去猜。
3. 工程内 ``data/datasets/<entry>``——本工程自带的导入位置。
4. 历史位置 ``<工程上级目录>/数据集/<entry>``——兼容改造前就摆好的那份数据，
   所以本次改造**不会让任何现有脚本失效**。
5. kagglehub 缓存 ``~/.cache/kagglehub/datasets/<owner>/<slug>/versions/*``
   ——只读扫描，不联网。已有缓存时无需再下载。

全都没有就抛 ``DatasetNotFoundError``，错误信息里直接给出可复制的下载命令——
**不要只说「找不到」**，那会让人去翻代码。

## 依赖边界

本模块**只用标准库**。``kagglehub`` 是可选依赖，只有真正要下载时才会 import；
没装的时候 ``fetch`` 会给出安装提示，而不是 ImportError 崩在导入期。
这样离线环境、纯推理部署都不需要装 kagglehub。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DATASETS",
    "DATA_ROOT_ENV",
    "DatasetNotFoundError",
    "DatasetSpec",
    "DatasetStatus",
    "default_path",
    "fetch",
    "find",
    "index_path",
    "kaggle_cache_root",
    "legacy_root",
    "list_datasets",
    "local_root",
    "require",
    "resolve",
    "status",
    "status_all",
]

# --------------------------------------------------------------------------
# 目录常量
# --------------------------------------------------------------------------

#: 工程根目录（``<root>/src/fisheries/datasets.py`` 往上三层）。
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: 改造前数据集所在的位置：``<工程上级>/数据集``。保留它是为了**向后兼容**——
#: 现有环境里数据已经摆在这儿，改造后必须照样找得到。
LEGACY_ROOT: Path = PROJECT_ROOT.parent / "数据集"

#: 改造后推荐的位置：数据导入到工程内的 ``data/datasets``，或者干脆不放工程里
#: （用 ``FISHERIES_DATA_ROOT`` 指到外部磁盘）。
LOCAL_ROOT: Path = PROJECT_ROOT / "data" / "datasets"

#: ``fetch`` 成功后把「名字 → 绝对路径」记在这里。
INDEX_PATH: Path = PROJECT_ROOT / "data" / "dataset_index.json"

#: 指定数据集总目录的环境变量。
DATA_ROOT_ENV: str = "FISHERIES_DATA_ROOT"

#: kagglehub 默认缓存位置（Windows/Linux 一致，都是 ``~/.cache/kagglehub``）。
KAGGLE_CACHE_ROOT: Path = Path.home() / ".cache" / "kagglehub" / "datasets"

#: 下载非 kaggle 直链时要走的前置镜像。``{url}`` 会被换成原始地址。
#:
#: 直连 ``raw.githubusercontent.com`` 在国内网络下经常被 RST，因此把镜像写成
#: 一条链：**按顺序试，第一个成功的就用**。空字符串表示「直连，不加前缀」，
#: 放最后当兵底。
ARCHIVE_URL_MIRRORS: tuple[str, ...] = (
    "https://ghproxy.net/{url}",
    "https://gh-proxy.com/{url}",
    "",
)


def legacy_root() -> Path:
    """历史数据集目录（``<工程上级>/数据集``）。"""
    return LEGACY_ROOT


def local_root() -> Path:
    """工程内导入目录（``<工程>/data/datasets``）。"""
    return LOCAL_ROOT


def index_path() -> Path:
    """索引文件路径。"""
    return INDEX_PATH


def kaggle_cache_root() -> Path:
    """kagglehub 缓存根目录。"""
    return KAGGLE_CACHE_ROOT


class DatasetNotFoundError(FileNotFoundError):
    """数据集既不在本地、也没有可用的下载来源。

    继承 ``FileNotFoundError``，所以原来 ``except FileNotFoundError`` 的调用方
    不用改一行代码就继续正常工作。
    """

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.name = name


# --------------------------------------------------------------------------
# 注册表
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """一个数据集的静态描述。

    ``entry`` 是**相对路径**（相对数据集根目录），可以指向一个目录也可以指向
    单个文件——历史布局里两种都有（``china_water_pollution_data.csv`` 是散在
    根目录下的文件，``Fish Disease Dataset`` 是目录）。
    """

    name: str
    entry: str
    kaggle_id: str = ""
    label: str = ""
    note: str = ""
    #: 必须存在才算「可用」的相对子路径（相对 ``entry``）。留空表示 entry 存在即可。
    required: tuple[str, ...] = ()
    #: 代码里哪些地方在用。仅供 ``--list`` 展示，不参与逻辑。
    used_by: tuple[str, ...] = ()
    #: Ultralytics 平台页面等外部入口。无 kaggle id 时至少有个出处。
    url: str = ""
    #: 已知体积（人写，仅用于提示，不做校验）。
    size_hint: str = ""
    #: 非 kaggle 来源的直链压缩包（目前用于 GitHub 上托管的 zip）。
    #: 有它就说明 ``fetch`` 走「下载 zip → 校验 → 解压」这条路，不需要 kagglehub。
    archive_url: str = ""
    #: ``archive_url`` 的 sha256（小写十六进制）。非空时下载后強制校验。
    sha256: str = ""

    @property
    def title(self) -> str:
        return self.label or self.name

    @property
    def fetchable(self) -> bool:
        """有没有可自动下载的来源。"""
        return bool(self.kaggle_id or self.archive_url)

    def download_command(self) -> str:
        """给人复制的下载命令。"""
        if not self.fetchable:
            return f"（该数据集需手动获取：{self.url or '无来源信息'}）"
        return f"python scripts/fetch_datasets.py --fetch {self.name}"

    def kaggle_snippet(self) -> str:
        """等价的原生 kagglehub 片段，方便想手动下的人。"""
        if not self.kaggle_id:
            return ""
        return (
            "import kagglehub\n"
            f'path = kagglehub.dataset_download("{self.kaggle_id}")'
        )


#: 数据集注册表。**这是「名字 → 位置」的唯一真相来源。**
#:
#: 顺序即 ``--list`` 的展示顺序，大体按「工程真正依赖的程度」排。
DATASETS: dict[str, DatasetSpec] = {
    "fish_disease": DatasetSpec(
        name="fish_disease",
        entry="Fish Disease Dataset",
        kaggle_id="subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia",
        label="鱼类病害图像（主用）",
        note=(
            "工程真正依赖的一份：Test/Train/Valid/Train_Augmented 四个划分，"
            "按病害类别分目录。演示脚本和单元测试都从这里取图。"
        ),
        required=("Test",),
        used_by=("scripts/demo_*.py", "tests/unit/test_agent.py", "前端演示图"),
        size_hint="179.6 MB / 23808 文件",
    ),
    "sb_fish_disease": DatasetSpec(
        name="sb_fish_disease",
        entry="SB-FishDisease",
        label="SB-FishDisease（鱼病图像，7 类）",
        note=(
            "GitHub 上托管的 SB-FishDisease：460 张图，按 7 个病害类别分目录，"
            "另有一份 SB-Fish-Disease.csv 清单。**只有图像级标签，没有检测框**。"
            "实测与 fish_disease 内容 99.8% 重复（457 张里 456 张同 md5），"
            "差别是它保留了 7 类原始目录名和官方 CSV——"
            "训练集构建拿它当「类别名与中英文对照」的权威来源。"
        ),
        required=("Healthy Fish",),
        used_by=("scripts/build_yolo_dataset.py（4 类训练集构建）",),
        url="https://github.com/SsekittoIsaac/SB-FishDisease",
        archive_url=(
            "https://raw.githubusercontent.com/SsekittoIsaac/"
            "SB-FishDisease/main/SB-FishDisease.zip"
        ),
        sha256="d4668cb145a35b8e16f4e3e73ff57cf0e41334bd809e0b4b10a11ac2d9b73870",
        size_hint="10.5 MB / 460 图",
    ),
    "china_water_pollution": DatasetSpec(
        name="china_water_pollution",
        entry="china_water_pollution_data.csv",
        kaggle_id="khushikyad001/china-water-pollution-monitoring-dataset",
        label="中国地表水断面监测（**已停用**）",
        note=(
            "单个 CSV，散在数据集根目录下（不是目录）。曾经是 M8 第一组外部验证的"
            "数据源，实测结论是负结果——它的 Pollution_Level 标签与全部测量列"
            "统计独立（连它自带的 Water_Quality_Index 都不相关），这把尺子没有刻度。"
            "**现已换成 aquaponics_pond**：地表水断面不是池塘养殖，"
            "而且没有可用的分级标签。注册表条目保留，仅为记录这份教训与备查。"
        ),
        size_hint="0.5 MB",
    ),
    "river_do": DatasetSpec(
        name="river_do",
        entry="Dissolved oxygen prediction in river water",
        kaggle_id="vbmokin/dissolved-oxygen-prediction-in-river-water",
        label="河川溶氧时序（**已停用**）",
        note=(
            "train.csv / test.csv，宽表 O2_1..O2_7。曾经是 M8 第二组外部验证的数据源，"
            "结论是线性外推 MAE 比持续性基线差 66%。**现已换成 aquaponics_pond 的"
            "溶氧序列**：河川的溶氧动力学（流动、复氧、点源污染）与静水塘不是一回事，"
            "而且原数据没有绝对时间单位（一个下标步 ≠ 一个已知时长）。"
            "注册表条目保留，仅为记录这份妥协与备查。"
        ),
        required=("train.csv",),
        size_hint="< 1 MB / 2 文件",
    ),
    "freshwater_fish_disease_asia": DatasetSpec(
        name="freshwater_fish_disease_asia",
        entry="Freshwater Fish Disease Aquaculture in south asia",
        kaggle_id="subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia",
        label="南亚淡水鱼病（检测标注版）",
        note=(
            "与 fish_disease 同源但切分方式不同，带检测标注。"
            "Ultralytics 平台上的同款（2,445 图 / 7 类 / 20.3 MB）可直接用于训练。"
        ),
        url="https://platform.ultralytics.com/allwet/datasets/"
        "freshwater-fish-disease-aquaculture-in-south-asia-2",
        size_hint="24.8 MB / 2451 文件",
    ),
    "aquaponics_pond": DatasetSpec(
        name="aquaponics_pond",
        entry="Sensor Based Aquaponics Fish Pond Datasets",
        kaggle_id="ogbuokiriblessing/sensor-based-aquaponics-fish-pond-datasets",
        label="鱼菜共生塘传感器时序（IoTpond*）——M8 两组验证的主数据源",
        note=(
            "11 个 IoTpond*.csv，共 1,114,970 行。列名至少有四种写法"
            "（Dissolved Oxygen(g/ml) / DISOLVED OXYGEN / PH / Ammonia(g/ml)），"
            "且用 -127 表示缺测——tables.py 的列名别名表就是照着这份数据做的。"
            "**单位有两个必须记住的坑**："
            "① 氨氮与硝酸盐标着 (g/ml)，但按 g/ml 读会得到「鱼在致死浓度里长胖 100 倍」"
            "的矛盾，故按 µg/L 除以 1000（唯一标 (mg/L) 的 IoTPond8 不除）；"
            "② 溶氧反之，标 (g/ml) 但只可能是 mg/L，不换算。"
            "M8 第一组（风险分级）与第二组（溶氧趋势）都用它。"
        ),
        used_by=(
            "src/fisheries/evaluation/ponds.py",
            "src/fisheries/evaluation/do_trend.py",
            "src/fisheries/tables.py（列名别名来源）",
        ),
        size_hint="96.0 MB / 11 文件",
    ),
    "water_potability": DatasetSpec(
        name="water_potability",
        entry="Water Quality",
        kaggle_id="adityakadiwal/water-potability",
        label="饮用水可饮用性",
        note="工程当前未使用，备选。",
        size_hint="0.5 MB / 1 文件",
    ),
    "fish_market": DatasetSpec(
        name="fish_market",
        entry="Fish.csv",
        kaggle_id="vipullrathod/fish-market",
        label="鱼类体尺-体重（Fish.csv）",
        note="工程当前未使用，备选（可用于生长模型）。",
        size_hint="< 0.1 MB",
    ),
    "soil_ammonia_15n": DatasetSpec(
        name="soil_ammonia_15n",
        entry="15N Abundance of Soil Ammonia Volatilization data.xlsx",
        kaggle_id="nikitamanaenkov/15n-abundance-of-soil-ammonia-volatilization",
        label="土壤氨挥发 15N 丰度",
        note="工程当前未使用，与养殖场景关系较弱。",
        size_hint="< 0.1 MB",
    ),
    "fish_disease_images": DatasetSpec(
        name="fish_disease_images",
        entry="archive",
        kaggle_id="moonburntcat/fish-disease-image-datasets",
        label="鱼病图像原始集（archive/1..5）",
        note=(
            "1.07 GB 的 PNG 原图，按 1..5 分五卷。**这是工程里最占地方的一份**，"
            "而代码并没有直接用它（代码用的是 fish_disease 的 Test/）。"
            "如果只是为了让工程变小，这一份是可以不导入的。"
        ),
        size_hint="1.07 GB / 2055 文件",
    ),
    "fish_disease_cleaned": DatasetSpec(
        name="fish_disease_cleaned",
        entry="cleaned-fish-disease-dataset",
        kaggle_id="rukesh17/cleaned-fish-disease-dataset",
        label="清洗过的鱼病图像集",
        note="工程当前未使用，备选。",
    ),
    "fish_species_images": DatasetSpec(
        name="fish_species_images",
        entry="fish-species-image-data",
        kaggle_id="sripaadsrinivasan/fish-species-image-data",
        label="鱼种图像数据",
        note="工程当前未使用，备选。",
    ),
}


# --------------------------------------------------------------------------
# 索引文件
# --------------------------------------------------------------------------


def _read_index() -> dict[str, str]:
    """读索引；坏了就当空的，不要让一个 JSON 语法错拖垮整条链路。"""
    try:
        raw = INDEX_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def _write_index(index: dict[str, str]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _record(name: str, path: Path) -> None:
    index = _read_index()
    index[name] = str(path)
    _write_index(index)


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------


def _roots() -> list[tuple[str, Path]]:
    """按优先级返回 ``(来源标签, 根目录)`` 列表。"""
    roots: list[tuple[str, Path]] = []
    env = os.environ.get(DATA_ROOT_ENV, "").strip()
    if env:
        roots.append((f"env:{DATA_ROOT_ENV}", Path(env)))
    roots.append(("工程 data/datasets", LOCAL_ROOT))
    roots.append(("历史目录 数据集/", LEGACY_ROOT))
    return roots


def _kaggle_cache_candidates(kaggle_id: str) -> list[Path]:
    """在 kagglehub 缓存里找这个 handle 已下载的版本目录，新的在前。

    只读、不联网：kagglehub 装没装都不影响这个函数。
    """
    if not kaggle_id or "/" not in kaggle_id:
        return []
    owner, slug = kaggle_id.split("/", 1)
    versions = KAGGLE_CACHE_ROOT / owner / slug / "versions"
    if not versions.is_dir():
        return []
    try:
        found = [p for p in versions.iterdir() if p.is_dir()]
    except OSError:
        return []
    # 版本号是数字目录名（1、2、3…），按数字降序；非数字的排后面。
    def sort_key(p: Path) -> tuple[int, str]:
        return (int(p.name), "") if p.name.isdigit() else (-1, p.name)

    return sorted(found, key=sort_key, reverse=True)


def _usable(base: Path, spec: DatasetSpec) -> bool:
    if not base.exists():
        return False
    return all((base / part).exists() for part in spec.required)


@dataclass(frozen=True)
class _Hit:
    path: Path
    source: str


def _locate(spec: DatasetSpec) -> _Hit | None:
    # 1) 环境变量总目录
    # 2) 工程 data/datasets
    # 3) 历史目录
    for label, root in _roots():
        candidate = root / spec.entry
        if _usable(candidate, spec):
            return _Hit(candidate, label)

    # 4) 索引文件（fetch 记下来的绝对路径）
    recorded = _read_index().get(spec.name)
    if recorded:
        candidate = Path(recorded)
        if _usable(candidate, spec):
            return _Hit(candidate, "索引 dataset_index.json")

    # 5) kagglehub 缓存
    for version_dir in _kaggle_cache_candidates(spec.kaggle_id):
        # kagglehub 下载的是一个数据集压缩包解出来的目录，里面可能是
        # 直接的 entry，也可能多套一层。两种都试。
        for candidate in (version_dir / spec.entry, version_dir):
            if _usable(candidate, spec):
                return _Hit(candidate, "kagglehub 缓存")
    return None


def find(name: str) -> Path | None:
    """找到就返回路径，找不到返回 ``None``（不抛异常）。"""
    spec = _spec(name)
    hit = _locate(spec)
    return hit.path if hit else None


def require(name: str) -> Path:
    """找到就返回路径；找不到抛 ``DatasetNotFoundError``，消息里带下载命令。"""
    spec = _spec(name)
    hit = _locate(spec)
    if hit is not None:
        return hit.path

    lines = [
        f"找不到数据集「{spec.title}」（{name}）。",
        "",
        "它不在以下任何一个位置：",
    ]
    for label, root in _roots():
        lines.append(f"  · {label}：{root / spec.entry}")
    lines.append(f"  · kagglehub 缓存：{KAGGLE_CACHE_ROOT / spec.kaggle_id}")
    lines += [
        "",
        "解决办法（任选其一）：",
        f"  1. 下载到默认位置：{spec.download_command()}",
        "  2. 已经有数据，指个路径就行（不用移动文件）：",
        f'       set {DATA_ROOT_ENV}=<你放数据集的目录>',
        f"     （该目录下应有 {spec.entry}）",
        "  3. 看看现在到底有哪些数据集：python scripts/fetch_datasets.py --status",
    ]
    if spec.url:
        lines.append(f"  参考：{spec.url}")
    raise DatasetNotFoundError(name, "\n".join(lines))


def resolve(name: str, *parts: str) -> Path:
    """取数据集里的某个具体文件/子目录，例如 ``resolve("fish_disease", "Test")``。"""
    base = require(name)
    return base.joinpath(*parts) if parts else base


def default_path(name: str) -> Path:
    """最佳努力定位：找得到就给真路径，找不到给一个「本该在那儿」的占位路径。

    专给**模块级默认值**用。import 期绝不该因为数据缺失就失败——否则一台
    没挂数据集的机器连 ``import fisheries.evaluation`` 都做不到，而那时它
    只是想做别的事。需要「找不到就报错」请用 :func:`require`。
    """
    found = find(name)
    if found is not None:
        return found
    spec = _spec(name)
    # 占位路径挑一个最像「用户会去放数据的地方」：历史目录优先。
    for root in (LEGACY_ROOT, LOCAL_ROOT):
        if (root / spec.entry).exists():
            return root / spec.entry
    return LOCAL_ROOT / spec.entry


def _spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        known = "、".join(sorted(DATASETS))
        raise KeyError(f"未注册的数据集：{name!r}。已注册的有：{known}") from None


# --------------------------------------------------------------------------
# 状态
# --------------------------------------------------------------------------


@dataclass
class DatasetStatus:
    spec: DatasetSpec
    path: Path | None = None
    source: str = ""
    size_bytes: int | None = None

    @property
    def available(self) -> bool:
        return self.path is not None

    @property
    def size_text(self) -> str:
        if self.size_bytes is None:
            return "—"
        mb = self.size_bytes / (1024 * 1024)
        return f"{mb:,.1f} MB" if mb >= 0.1 else f"{self.size_bytes / 1024:,.0f} KB"


def _dir_size(path: Path, cap_files: int = 200_000) -> int:
    """目录/文件体积。带文件数上限，避免在十万级文件上卡住。"""
    try:
        if path.is_file():
            return path.stat().st_size
    except OSError:
        return 0
    total = 0
    seen = 0
    for dirpath, _dirnames, filenames in os.walk(path, onerror=lambda _e: None):
        for fn in filenames:
            seen += 1
            if seen > cap_files:
                return total
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                continue
    return total


def status(name: str | None = None, *, with_size: bool = False) -> list[DatasetStatus]:
    """列出数据集状态。``with_size=True`` 时统计体积（会走一遍目录，慢）。"""
    names = [name] if name else list(DATASETS)
    out: list[DatasetStatus] = []
    for n in names:
        spec = DATASETS[n]
        hit = _locate(spec)
        st = DatasetStatus(
            spec=spec,
            path=hit.path if hit else None,
            source=hit.source if hit else "",
        )
        if with_size and st.path is not None:
            st.size_bytes = _dir_size(st.path)
        out.append(st)
    return out


def status_all(*, with_size: bool = False) -> list[DatasetStatus]:
    return status(None, with_size=with_size)


def list_datasets() -> list[DatasetSpec]:
    return list(DATASETS.values())


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------


def _sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> str:
    """下载 ``url`` 到 ``dest``，返回实际用的那个（镜像）地址。

    按 :data:`ARCHIVE_URL_MIRRORS` 顺序试：国内直连 GitHub raw 经常被 RST，
    先用代理镜像。先写 ``.part`` 再改名，中途断掉不会留下半截文件被当成好的。
    """
    import urllib.error
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last_error: Exception | None = None
    for template in ARCHIVE_URL_MIRRORS:
        candidate = template.format(url=url) if template else url
        try:
            request = urllib.request.Request(
                candidate, headers={"User-Agent": "fisheries-datasets/1.0"}
            )
            with urllib.request.urlopen(request, timeout=300) as response:
                with part.open("wb") as handle:
                    while True:
                        chunk = response.read(1 << 16)
                        if not chunk:
                            break
                        handle.write(chunk)
            part.replace(dest)
            return candidate
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_error = exc
            part.unlink(missing_ok=True)
            continue
    raise DatasetNotFoundError(
        "fetch",
        f"所有镜像都下载失败：{url}\n最后一条错误：{last_error}",
    )


def _flatten_to(root: Path, required: tuple[str, ...]) -> None:
    """把被多余目录套住的真实内容上提到 ``root``。

    上游打的 zip 经常是 ``<name>/<name>/<子目录>`` 这种双层嵌套。这里以
    ``required`` 里的第一个名字当锚点——哪个目录直接包含全部 ``required``，
    就把它的内容整份提到 ``root`` 下。
    """
    if not required or (root / required[0]).exists():
        return
    for anchor in sorted(root.rglob(required[0])):
        inner = anchor.parent
        if not all((inner / part).exists() for part in required):
            continue
        for child in list(inner.iterdir()):
            target = root / child.name
            if target.exists():
                continue
            child.rename(target)
        # 清掉搬空了的壳目录（由内向外删）。
        for leftover in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if leftover.is_dir() and not any(leftover.iterdir()):
                leftover.rmdir()
        return


def _extract_archive(archive: Path, dest: Path, spec: DatasetSpec) -> int:
    """解压 zip 到 ``dest``，返回写出的文件数。"""
    import shutil
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            # 防目录穿越：成员名里出现 .. 一律拒绝。
            relative = Path(info.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                continue
            out = dest / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written += 1
    _flatten_to(dest, spec.required)
    return written


def fetch(name: str) -> Path:
    """下载数据集，并把结果路径记进索引。

    两条路径：

    * 有 ``kaggle_id`` → 走 kagglehub（可选依赖，缺失时给安装提示）。
    * 有 ``archive_url`` → 直接下 zip，校验 ``sha256`` 后解压，**不需要 kagglehub**。

    **没有网络时不要调用它**——先试 :func:`find`，本地已有就不会走到这里。
    """
    spec = _spec(name)
    existing = find(name)
    if existing is not None:
        return existing

    if spec.archive_url:
        return _fetch_archive(spec)

    if not spec.kaggle_id:
        raise DatasetNotFoundError(
            name,
            f"数据集「{spec.title}」没有可自动下载的来源。\n"
            f"请手动获取：{spec.url or '（未记录来源）'}",
        )

    try:
        import kagglehub  # 延迟导入：离线部署不该被这个依赖卡住
    except ImportError as exc:  # pragma: no cover - 取决于环境
        raise DatasetNotFoundError(
            name,
            "需要 kagglehub 才能下载数据集，但当前环境没有安装。\n\n"
            "安装：\n"
            '    .venv\\Scripts\\python.exe -m pip install -e ".[datasets]"\n'
            "或者只装这一个包：\n"
            "    .venv\\Scripts\\python.exe -m pip install kagglehub\n\n"
            "不想装也可以：手动下好之后用 "
            f"{DATA_ROOT_ENV} 指向那个目录。",
        ) from exc

    downloaded = Path(kagglehub.dataset_download(spec.kaggle_id))

    # 下载目录里通常就是 entry，但有些包会多套一层，两个位置都找一下。
    for candidate in (downloaded / spec.entry, downloaded):
        if _usable(candidate, spec):
            _record(name, candidate)
            return candidate

    # 内容对不上（比如包结构变了）：记录也需要报出来，否则用户不知道为什么失败。
    raise DatasetNotFoundError(
        name,
        f"kagglehub 下载完成（{downloaded}），但里面找不到 {spec.entry!r}。\n"
        f"可能是上游包结构变了。请人工确认后把 {DATA_ROOT_ENV} 指过去。",
    )


def _fetch_archive(spec: DatasetSpec) -> Path:
    """下载 ``archive_url`` 形式的直链压缩包（目前是 GitHub 上的 zip）。"""
    import tempfile

    with tempfile.TemporaryDirectory(prefix=f"fisheries-{spec.name}-") as tmp:
        workspace = Path(tmp)
        archive = workspace / (spec.archive_url.rsplit("/", 1)[-1] or "archive.zip")
        used = _download(spec.archive_url, archive)

        if spec.sha256:
            actual = _sha256_of(archive)
            if actual != spec.sha256:
                raise DatasetNotFoundError(
                    spec.name,
                    f"下载到的文件 sha256 对不上，已丢弃。\n"
                    f"  来源：{used}\n"
                    f"  期望：{spec.sha256}\n"
                    f"  实际：{actual}\n"
                    f"要么上游更新了（请核对后更新注册表里的 sha256），"
                    f"要么下载被中间人改了。",
                )

        staging = workspace / "extracted"
        written = _extract_archive(archive, staging, spec)
        if written == 0:
            raise DatasetNotFoundError(
                spec.name, f"压缩包解出来是空的：{spec.archive_url}"
            )

        target = LOCAL_ROOT / spec.entry
        if target.exists():
            import shutil

            shutil.rmtree(target, ignore_errors=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.move(str(staging), str(target))

    _record(spec.name, target)
    if not _usable(target, spec):
        raise DatasetNotFoundError(
            spec.name,
            f"解压完成（{target}），但里面找不到 {list(spec.required)}。",
        )
    return target
