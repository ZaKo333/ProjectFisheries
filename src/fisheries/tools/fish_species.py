"""工具层：fish_species_detect —— 鱼种图像识别。

定位
----
品种在本系统里不是一个标签，而是**一整套阈值档案**：``虾蟹`` 的氨氮警戒指
0.3 mg/L，``鲤科`` 是 0.5 mg/L；温度适宜区间也不同（见
:data:`fisheries.tools.water_risk.SPECIES_PROFILES`）。因此"品种判错"会让后面
所有水质判定整体偏移，而不是只错一个字段。

这个模块把「YOLOv8-cls 鱼种分类模型」包装成标准工具函数，供接口层在用户上传
图片后自动推断品种，替换掉原来手工选下拉框的做法。

为什么这里**没有** mock 后端
---------------------------
:mod:`fisheries.tools.fish_disease` 有个基于文件哈希的确定性 mock：病害判错最多
让提示语难听一点，阈值不变。品种不同：mock 到 ``虾蟹`` 会悄悄把氨氮红线收紧一倍，
用户看到的是一份**看起来完全正常、实际按错档位算出来**的结论。这是安全事故，
不是"演示瑕疵"。

所以本模块的态度是：模型不在就说"没识别出来"，让上层退回默认档位并如实提示，
绝不随机编一个品种。:attr:`FishSpeciesOutput.detected` 因此是显式字段，
调用方必须看它，而不是看 ``species`` 是否有值。

标签体系
--------
对外只输出 :data:`SPECIES_LABELS` 三个值，直接派生自 ``SPECIES_PROFILES`` 的键，
避免"界面上能显示的品种"与"有阈值档案的品种"分叉。

底层模型（训练集）的类别名可以是中英文任意写法，靠 :data:`RAW_SPECIES_ALIASES`
做模糊匹配收敛到三档。**匹配不上的一律丢弃并单独记录**，不会硬塞进某一档：
数据集里出现 ``Catfish`` 就应当如实说"不是这三个品种"，而不是把它算成鲤科。
"""

from __future__ import annotations

import os
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from fisheries import console as _console

from .fish_disease import _sniff_image_format
from .water_risk import SPECIES_PROFILES

# --------------------------------------------------------------------------- #
# 一、标签体系
# --------------------------------------------------------------------------- #

#: 对外暴露的品种取值。**从阈值档案派生**，保证两者永不分叉。
SPECIES_LABELS: tuple[str, ...] = tuple(SPECIES_PROFILES)

#: 底层类别名 → 对外品种 的关键词表。
#:
#: 匹配方式是「归一化后的类名里是否含有关键词」，不分大小写、忽略 ``_``/``-``/空格。
#: 同一品种下按关键词长度降序尝试，``grass carp`` 会先于 ``carp`` 命中。
#: 收录的写法覆盖了常见 Kaggle 鱼类数据集的命名习惯（英文俗名、拉丁属名、中文）。
RAW_SPECIES_ALIASES: dict[str, tuple[str, ...]] = {
    "鲤科": (
        "cyprinidae", "cyprinid", "grass carp", "silver carp", "bighead carp",
        "common carp", "crucian", "goldfish", "tench", "rohu", "catla", "mrigal",
        "carp", "鲤", "鲫", "草鱼", "鲢", "鳙", "青鱼", "鳊",
    ),
    "罗非鱼": (
        "oreochromis", "tilapia", "nile tilapia", "罗非", "非洲鲫",
    ),
    "虾蟹": (
        "freshwater prawn", "giant river prawn", "macrobrachium",
        "crayfish", "crawfish", "lobster", "shrimp", "prawn",
        "crab", "虾", "蟹", "小龙虾", "对虾",
    ),
}

#: 置信度下界。低于它就不认定品种（宁可让上层用默认档位，也不按错档判水质）。
LOW_CONFIDENCE_THRESHOLD = 0.55

#: 权重目录（工程根下的 models/）。
_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models"
)

#: 权重文件名搜索顺序。``yolov8-species.pt`` 是 ``scripts/train_yolo_species.py`` 的产出。
_WEIGHTS_CANDIDATES: tuple[str, ...] = ("yolov8-species.pt", "fish_species_cls.pt")

#: 默认权重路径（对外文档与错误提示里引用的那一个）。
DEFAULT_WEIGHTS_PATH = os.path.join(_MODELS_DIR, _WEIGHTS_CANDIDATES[0])

#: 模型未就绪时 ``model_version`` 的取值。**不以 mock- 开头**——它不是 mock，
#: 而是"这件事没做"。
UNAVAILABLE_MODEL_VERSION = "unavailable"


# --------------------------------------------------------------------------- #
# 二、出参数据契约
# --------------------------------------------------------------------------- #

class FishSpeciesOutput(BaseModel):
    """`fish_species_detect` 的返回体。"""

    model_config = ConfigDict(protected_namespaces=())

    tool: str = Field(default="fish_species_detect", description="工具名")
    status: Literal["ok", "error"] = Field(default="ok", description="执行状态")
    image_path: str = Field(..., description="实际检测的图片路径")
    detected: bool = Field(
        default=False,
        description="是否真的从图像中认出了品种；false 时上层必须退回默认档位",
    )
    species: str | None = Field(
        default=None, description="品种：鲤科/罗非鱼/虾蟹；detected=false 时为 None"
    )
    confidence: float | None = Field(default=None, ge=0, le=1, description="该品种的置信度")
    is_low_confidence: bool = Field(
        default=False, description="置信度是否低于阈值；true 时不应据此切换阈值档案"
    )
    all_scores: dict[str, float] = Field(
        default_factory=dict, description="各品种的聚合概率（降序，最多 top_k 项）"
    )
    raw_class: str | None = Field(
        default=None, description="底层模型 top-1 的原始类别名，用于人工复核"
    )
    unmapped_classes: list[str] = Field(
        default_factory=list,
        description="模型输出了但无法归入三个品种档案的类别名（概率被丢弃）",
    )
    model_version: str = Field(..., description="模型版本标识；unavailable 表示模型未就绪")
    degraded_mode: bool = Field(..., description="true 表示没有可用的品种模型")
    image_meta: dict[str, Any] = Field(default_factory=dict, description="图片元信息")
    note: str = Field(default="", description="给 Agent / 用户看的一句话提示")
    error_code: str | None = Field(default=None, description="status=error 时的错误码")
    error_message: str | None = Field(default=None, description="status=error 时的错误说明")


# --------------------------------------------------------------------------- #
# 三、输入校验与后端探测
# --------------------------------------------------------------------------- #

def _is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def _normalize(raw: str) -> str:
    """类名归一化：小写，把 ``_``/``-`` 换成空格，压缩多余空白。"""
    text = raw.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def map_raw_species(raw: str) -> str | None:
    """把模型输出的原始类别名收敛到三个品种之一；无法判定返回 ``None``。

    Examples:
        >>> map_raw_species("Nile Tilapia")
        '罗非鱼'
        >>> map_raw_species("Grass_Carp")
        '鲤科'
        >>> map_raw_species("Catfish") is None
        True
    """
    key = _normalize(raw)
    if not key:
        return None
    for species, aliases in RAW_SPECIES_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if _normalize(alias) in key:
                return species
    return None


def resolve_weights(weights_path: str | None = None) -> str | None:
    """返回**实际会被使用**的权重文件绝对路径；没有可用权重时返回 ``None``。

    供接口层的 ``/health`` 与训练脚本共用，避免各处重复推断搜索顺序。
    """
    if weights_path:
        return weights_path if os.path.isfile(weights_path) else None
    for name in _WEIGHTS_CANDIDATES:
        candidate = os.path.join(_MODELS_DIR, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _resolve_backend(weights_path: str | None) -> str:
    """探测可用的推理后端，返回 ``"yolo"`` 或 ``"unavailable"``。

    找不到权重、或 ``ultralytics`` 没装，都落到 ``"unavailable"``——**不落 mock**，
    原因见模块开头。
    """
    if resolve_weights(weights_path) is None:
        return "unavailable"
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return "unavailable"
    # import ultralytics 会把 sys.stdout 编码改成 UTF-8；不改回来，之后打印的中文会乱码。
    _console.install()
    return "yolo"


# --------------------------------------------------------------------------- #
# 四、YOLO 后端
# --------------------------------------------------------------------------- #

def _yolo_predict_species(
    image_path: str, weights_path: str
) -> tuple[dict[str, float], str, list[str]]:
    """执行 YOLOv8-cls 推理，把底层类别概率聚合到三个品种档案。

    Returns:
        (品种 → 聚合概率, 底层 top-1 类别名, 无法归类的类别名列表)

    底层类别里出现三个档案之外的鱼种时**不抛异常**：概率被丢弃并记入第三个返回
    值。数据集多几个品种是正常情况，为此让整个请求失败没有意义；但把 ``Catfish``
    算成鲤科会更糟，那份水质结论会被当成真的。
    """
    from ultralytics import YOLO  # 延迟导入：未安装时不影响其他工具

    _console.install()  # ultralytics 会改 stdout 编码
    model = YOLO(weights_path)
    results = model.predict(image_path, verbose=False)[0]
    probs = results.probs
    if probs is None:
        raise RuntimeError("模型未返回分类概率，请确认使用的是 YOLOv8-cls 分类权重而非检测权重")

    names: dict[int, str] = results.names  # type: ignore[assignment]
    raw_scores: dict[str, float] = {
        names[i]: float(p) for i, p in enumerate(probs.data.tolist())
    }

    # ---- 多对一要「求和」，不是「取最大」 ----
    agg: dict[str, float] = {label: 0.0 for label in SPECIES_LABELS}
    unmapped: list[str] = []
    for raw_name, p in raw_scores.items():
        target = map_raw_species(raw_name)
        if target is None:
            unmapped.append(raw_name)
            continue
        agg[target] += p

    raw_top = max(raw_scores, key=lambda k: raw_scores[k])
    return agg, raw_top, unmapped


# --------------------------------------------------------------------------- #
# 五、对外工具函数
# --------------------------------------------------------------------------- #

def fish_species_detect(
    image_path: str,
    weights_path: str | None = None,
    top_k: int = 3,
    conf_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """【内部工具】从图片识别鱼种，用于自动选择阈值档案。

    入参
    ----
    image_path : str
        鱼体图片的本地绝对路径或 http(s) URL。
    weights_path : str | None
        自定义权重路径；None 时依次找 ``models/yolov8-species.pt``、
        ``models/fish_species_cls.pt``。
    top_k : int
        返回概率最高的前 K 个品种，默认 3（即全部）。
    conf_threshold : float
        认定品种的最低置信度，默认 0.55。低于该值时 ``detected=False``。

    返回
    ----
    dict
        可直接 JSON 序列化；失败时返回 ``status="error"`` 而非抛异常。

    Examples:
        >>> out = fish_species_detect("/path/to/tilapia.jpg")
        >>> out["status"]
        'ok'
        >>> out["detected"]  # 模型未训练时为 False，species 也是 None
        False
    """
    is_remote = _is_url(image_path)

    # ---- 1. 本地路径校验 ----
    if not is_remote:
        resolved = os.path.abspath(image_path)
        if not os.path.exists(resolved):
            return _error(image_path, "IMAGE_NOT_FOUND", f"图片不存在：{resolved}")
        if not os.path.isfile(resolved):
            return _error(image_path, "NOT_A_FILE", f"路径不是文件：{resolved}")
        fmt = _sniff_image_format(resolved)
        if fmt is None:
            return _error(
                image_path,
                "UNSUPPORTED_IMAGE",
                "无法识别的图片格式（支持 jpeg/png/webp/bmp/gif）。"
                "请确认文件未损坏，且不是被重命名过的其他类型文件。",
            )
        image_meta: dict[str, Any] = {
            "format": fmt,
            "size_bytes": os.path.getsize(resolved),
            "resolved_path": resolved,
            "is_remote": False,
        }
        display_path = resolved
    else:
        image_meta = {"format": "unknown", "size_bytes": None, "is_remote": True}
        display_path = image_path

    # ---- 2. 选择后端 ----
    backend = _resolve_backend(weights_path)
    weights = resolve_weights(weights_path)

    if backend != "yolo" or weights is None:
        return FishSpeciesOutput(
            tool="fish_species_detect",
            status="ok",
            image_path=display_path,
            detected=False,
            species=None,
            model_version=UNAVAILABLE_MODEL_VERSION,
            degraded_mode=True,
            image_meta=image_meta,
            note=(
                "品种识别模型未就绪，无法从图片判断品种。"
                "请按默认品种（鲤科）判定，或显式指定品种。"
                f"训练命令见 scripts/train_yolo_species.py（期望权重：{DEFAULT_WEIGHTS_PATH}）。"
            ),
        ).model_dump(mode="json")

    try:
        agg, raw_top, unmapped = _yolo_predict_species(display_path, weights)
        model_version = os.path.basename(weights)
        degraded = False
    except Exception as exc:  # noqa: BLE001 —— 模型推理失败时降级，不让接口整条挂掉
        return FishSpeciesOutput(
            tool="fish_species_detect",
            status="ok",
            image_path=display_path,
            detected=False,
            species=None,
            model_version=f"{UNAVAILABLE_MODEL_VERSION} (yolo failed: {type(exc).__name__})",
            degraded_mode=True,
            image_meta=image_meta,
            note=(
                f"品种识别模型调用失败（{type(exc).__name__}: {exc}），"
                "本次按默认品种（鲤科）判定。"
            ),
        ).model_dump(mode="json")

    # ---- 3. 汇总 ----
    total = sum(agg.values())
    scores = {k: (v / total if total > 0 else 0.0) for k, v in agg.items()}
    ref = scores if total > 0 else agg
    top_species = max(ref, key=lambda k: ref[k])
    confidence = float(ref[top_species])
    top_sorted = dict(
        sorted(ref.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_k)]
    )

    detected = total > 0 and confidence >= conf_threshold
    is_low_conf = total > 0 and confidence < conf_threshold

    if total <= 0:
        note = (
            f"模型判定为「{raw_top}」，但这不属于 鲤科/罗非鱼/虾蟹 三个品种档案，"
            "无法据此选择阈值，按默认品种（鲤科）判定。"
        )
    elif detected:
        note = f"从图片识别为{top_species}（置信度 {confidence:.2f}），已按该品种的阈值档案判定。"
    else:
        note = (
            f"品种识别置信度 {confidence:.2f} 低于阈值 {conf_threshold:.2f}，"
            "不足以切换阈值档案，按默认品种（鲤科）判定；可补拍清晰照片或手工指定品种。"
        )
    if unmapped:
        note += f" 另有 {len(unmapped)} 个类别未登记：{', '.join(unmapped[:5])}。"

    return FishSpeciesOutput(
        tool="fish_species_detect",
        status="ok",
        image_path=display_path,
        detected=detected,
        species=top_species if detected else None,
        confidence=round(confidence, 4),
        is_low_confidence=is_low_conf,
        all_scores={k: round(v, 4) for k, v in top_sorted.items()},
        raw_class=raw_top,
        unmapped_classes=unmapped,
        model_version=model_version,
        degraded_mode=degraded,
        image_meta=image_meta,
        note=note,
    ).model_dump(mode="json")


def _error(image_path: str, code: str, message: str) -> dict[str, Any]:
    """构造统一的错误返回体。"""
    return FishSpeciesOutput(
        tool="fish_species_detect",
        status="error",
        image_path=image_path,
        model_version=UNAVAILABLE_MODEL_VERSION,
        degraded_mode=True,
        error_code=code,
        error_message=message,
        note=f"品种识别失败（{code}）：{message}",
    ).model_dump(mode="json")


__all__ = [
    "DEFAULT_WEIGHTS_PATH",
    "LOW_CONFIDENCE_THRESHOLD",
    "RAW_SPECIES_ALIASES",
    "SPECIES_LABELS",
    "UNAVAILABLE_MODEL_VERSION",
    "FishSpeciesOutput",
    "fish_species_detect",
    "map_raw_species",
    "resolve_weights",
]
