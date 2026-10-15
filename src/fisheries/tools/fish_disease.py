"""工具 2：fish_disease_detect —— 鱼体病害图像识别。

定位
----
本模块把「YOLO 图像分类模型」包装成 Agent 可调用的标准工具函数。
对外只暴露 `fish_disease_detect`，内部按以下优先级选择执行后端：

    1. 真实模型：`ultralytics` 已安装且权重文件存在 → 跑 YOLOv8-cls 推理；
    2. Mock 后端：以上条件不满足 → 用"文件内容哈希"生成**确定性**假结果。

Mock 存在的意义
---------------
用户（项目负责人）负责模型训练，模型权重到位前 Agent 链路不能空转。
Mock 结果完全由图片路径+内容决定 → 同一张图每次结果一致 → 单元测试可断言、
演示可复现。同时 `degraded_mode=True` 与 `model_version="mock-v0.1"` 会
明确标注"这不是真实模型输出"，避免误当实验结论。

标签体系（本工具对外的 5 个取值）
---------------------------------
    烂鳃     ← Bacterial gill disease                  （细菌性烂鳃病）
    出血     ← Bacterial diseases - Aeromoniasis       （细菌性败血症）
             + Bacterial Red disease                   （细菌性赤皮病）
    水霉     ← Fungal diseases Saprolegniasis          （水霉病）
    健康     ← Healthy Fish
    其他     ← Parasitic diseases / Viral diseases White tail disease

关于「其他」：用户的接口约定只有 烂鳃/出血/水霉/健康 四类，但底层数据集是 7 类。
若强行把寄生虫病、病毒性白尾病塞进"健康"，等于把病鱼判成健康鱼——
这在生产环境是安全事故。因此这里保留第 5 个安全兜底值 "其他"，
同时用 raw_class 字段无损保留底层 7 分类的原始判定，供人工复核。
如需严格四类输出，把 ALLOW_OTHER_FALLBACK 置为 False 即可。
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from fisheries import console as _console

# --------------------------------------------------------------------------- #
# 一、标签体系与映射
# --------------------------------------------------------------------------- #

#: 对外暴露的病害类别（顺序即 mock 采样顺序）
DISEASE_LABELS: list[str] = ["烂鳃", "出血", "水霉", "健康", "其他"]

#: 是否允许"其他"兜底。置 False 时，寄生虫/病毒性会被归到 出血（最接近的细菌性病组）。
ALLOW_OTHER_FALLBACK: bool = True

#: 底层 7 分类 → 对外 5 类 的映射表。
#: 注意这是**多对一**映射，聚合时应当把同一目标类的原始概率**求和**，
#: 而不是取最大值——取最大值会系统性低估合并类别的概率。
RAW_CLASS_TO_LABEL: dict[str, str] = {
    # —— 上游 SB-FishDisease 原始 7 分类（图像分类口径）——
    "Bacterial gill disease": "烂鳃",
    "Bacterial diseases - Aeromoniasis": "出血",
    "Bacterial Red disease": "出血",
    "Fungal diseases Saprolegniasis": "水霉",
    "Healthy Fish": "健康",
    "Parasitic diseases": "其他",
    "Viral diseases White tail disease": "其他",
    # —— 本地四分类训练集口径（scripts/build_yolo_dataset.py 产出的目录名）——
    # 多模态演示只要求 健康 / 烂鳃 / 细菌性出血 / 水霉 四类，训练权重按这四个名字输出，
    # 因此必须在此登记；否则 _yolo_predict 会因为"模型输出的类别未登记"直接报错。
    "gill_rot": "烂鳃",
    "bacterial_hemorrhage": "出血",
    "saprolegniasis": "水霉",
    "healthy": "健康",
}

#: 类别元信息：分组、严重度权重（进入 M4 融合层用）、默认典型置信度下界
LABEL_META: dict[str, dict[str, Any]] = {
    "烂鳃": {"group": "细菌性", "severity_weight": 0.85},
    "出血": {"group": "细菌性", "severity_weight": 0.90},
    "水霉": {"group": "真菌性", "severity_weight": 0.70},
    "其他": {"group": "其他", "severity_weight": 0.75},
    "健康": {"group": "健康", "severity_weight": 0.00},
}

#: 支持的图片魔数（用于不依赖 Pillow 的轻量格式校验）
_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
]

#: 类别 → 分组中文名（拼提示语用）
LB_GROUP: dict[str, str] = {
    k: v["group"] for k, v in LABEL_META.items() if v["group"] != "健康"
}

MOCK_MODEL_VERSION = "mock-v0.1"

#: mock 后端的取值来源，取值 "hash" 或 "folder_hint"。
#:
#: - ``"hash"``（默认）：由「图片路径 + 文件大小」的 SHA-256 决定。与真实标签无关，
#:   因此它能诚实地暴露"模型其实在瞎猜"这件事，是单元测试与压力测试该用的模式。
#: - ``"folder_hint"``：若图片所在目录名或文件名能匹配到底层 7 分类中的某一类，
#:   就按该类输出（叠加确定性噪声）。**仅供模型尚未训练时的链路联调与答辩演示**，
#:   此时 `note` 会明确标注"标签来自文件名提示，非图像推理结果"。
#:
#: 之所以默认不是 "folder_hint"：如果默认就在读文件名，评审时一旦被问
#: "你怎么证明不是偷看标签"，这个实现无法自证清白。必须 opt-in 且显式标注才站得住。
MOCK_SOURCE_DEFAULT = "hash"

#: model_version 里用于标记"本结果来自文件名提示"的后缀
_MOCK_HINT_TAG = "+folder_hint"
LOW_CONFIDENCE_THRESHOLD = 0.60

#: 权重目录（工程根下的 models/）。
_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "models"
)

#: 权重文件名搜索顺序。``yolov8-fish.pt`` 是 ``scripts/train_yolo_fish.py`` 的产出；
#: ``fish_disease_cls.pt`` 是历史命名，保留以便旧权重继续可用。
_WEIGHTS_CANDIDATES: tuple[str, ...] = ("yolov8-fish.pt", "fish_disease_cls.pt")

#: 默认权重路径（对外文档与错误提示里引用的那一个）。
DEFAULT_WEIGHTS_PATH = os.path.join(_MODELS_DIR, _WEIGHTS_CANDIDATES[0])


# --------------------------------------------------------------------------- #
# 二、出参数据契约
# --------------------------------------------------------------------------- #

class FishDiseaseOutput(BaseModel):
    """`fish_disease_detect` 的返回体。"""

    model_config = ConfigDict(protected_namespaces=())

    tool: str = Field(default="fish_disease_detect", description="工具名")
    status: Literal["ok", "error"] = Field(default="ok", description="执行状态")
    image_path: str = Field(..., description="实际检测的图片路径")
    disease_type: str | None = Field(default=None, description="病害类型：烂鳃/出血/水霉/健康/其他")
    disease_group: str | None = Field(default=None, description="病害大类：细菌性/真菌性/其他/健康")
    confidence: float | None = Field(default=None, ge=0, le=1, description="判定类别的置信度")
    is_healthy: bool | None = Field(default=None, description="是否判定为健康")
    severity_weight: float | None = Field(
        default=None, description="严重度权重 0–1，供 M4 多源融合使用"
    )
    is_low_confidence: bool = Field(
        default=False, description="置信度是否低于阈值；true 时应提示人工复核"
    )
    all_scores: dict[str, float] = Field(
        default_factory=dict, description="各类别概率分布（降序，最多 top_k 项）"
    )
    raw_class: str | None = Field(
        default=None, description="底层 7 分类模型的原始判定类别名，用于人工复核与追溯"
    )
    model_version: str = Field(..., description="模型版本标识；以 mock- 开头表示非真实模型")
    degraded_mode: bool = Field(..., description="true 表示走的是 mock 后端")
    image_meta: dict[str, Any] = Field(default_factory=dict, description="图片元信息")
    note: str = Field(default="", description="给 Agent / 用户看的一句话提示")
    error_code: str | None = Field(default=None, description="status=error 时的错误码")
    error_message: str | None = Field(default=None, description="status=error 时的错误说明")


# --------------------------------------------------------------------------- #
# 三、输入校验与后端探测
# --------------------------------------------------------------------------- #

def _sniff_image_format(path: str) -> str | None:
    """通过文件头魔数判断图片格式；无法识别返回 None。"""
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
    except OSError:
        return None
    for magic, name in _MAGIC:
        if head.startswith(magic):
            return name
    # WEBP: RIFF....WEBP
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def _is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def resolve_weights(weights_path: str | None = None) -> str | None:
    """返回**实际会被使用**的权重文件绝对路径；没有可用权重时返回 ``None``。

    这是给外部（接口层的 ``/health``、诊断脚本）用的公开查询口，
    让它们不必 import 私有常量就能如实回答「图像模型到底有没有就绪」。
    ``ultralytics`` 装没装不影响本函数的判断——那由调用方另行检查。
    """
    if weights_path:
        return weights_path if os.path.isfile(weights_path) else None
    for name in _WEIGHTS_CANDIDATES:
        candidate = os.path.join(_MODELS_DIR, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _resolve_backend(weights_path: str | None) -> tuple[str, str | None]:
    """探测可用的推理后端，返回 (backend, weights_path)。

    backend 取值为 "yolo" 或 "mock"。

    未显式指定 ``weights_path`` 时按 :data:`_WEIGHTS_CANDIDATES` 的顺序找第一个存在的
    权重文件；一个都没有就落到 mock 后端（而不是抛异常），保证演示链路始终起得来。
    找了权重但 ``ultralytics`` 没装时同样落 mock。
    """
    candidate = resolve_weights(weights_path)
    if candidate is None:
        return "mock", None
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return "mock", None
    # import ultralytics 会把 sys.stdout 的编码改成 UTF-8；不改回来，
    # 调用方在中文控制台里从这一行之后打印的中文全会变乱码。
    _console.install()
    return "yolo", candidate


# --------------------------------------------------------------------------- #
# 四、Mock 后端
# --------------------------------------------------------------------------- #

def _folder_hint(image_path: str) -> str | None:
    """从路径里找真实类别提示（底层 7 分类的目录名或文件名）。

    数据集按类别分目录存放，所以这类提示几乎总是命中，**这也正是它不能作为
    推理结果的原因**——它是读标签，不是看图。

    Returns:
        命中的底层类别英文名；未命中返回 None。
    """
    haystack = os.path.abspath(image_path).lower().replace("\\", "/")
    # 长的类别名优先匹配，避免 "Bacterial Red disease" 被更短的规则抢先命中
    for raw in sorted(RAW_CLASS_TO_LABEL, key=len, reverse=True):
        if raw.lower() in haystack:
            return raw
    return None


def _mock_scores(
    image_path: str,
    n_classes: int = len(DISEASE_LABELS),
    source: str = MOCK_SOURCE_DEFAULT,
) -> tuple[str, dict[str, float]]:
    """生成确定性的假预测分布。

    种子来自「文件绝对路径 + 文件大小 + 模型版本」的 SHA-256，
    因此同一张图片在任意机器、任意次运行都得到相同结果——这是可测试性的前提。

    Args:
        image_path: 图片路径，用作种子来源，也用作 folder_hint 的匹配对象。
        n_classes: 保留参数，保持与 YOLO 后端的调用形状一致。
        source: "hash" 或 "folder_hint"，见 MOCK_SOURCE_DEFAULT 的说明。
    """
    try:
        size = os.path.getsize(image_path)
    except OSError:
        size = 0
    seed_src = f"{os.path.abspath(image_path)}|{size}|{MOCK_MODEL_VERSION}"
    digest = hashlib.sha256(seed_src.encode("utf-8")).digest()
    rnd = random.Random(int.from_bytes(digest[:8], "big"))

    winner: str | None = None
    if source == "folder_hint":
        raw = _folder_hint(image_path)
        if raw is not None:
            winner = RAW_CLASS_TO_LABEL[raw]
    if winner is None:
        winner = rnd.choice(DISEASE_LABELS)

    top = rnd.uniform(0.70, 0.97)
    others = [lab for lab in DISEASE_LABELS if lab != winner]
    raw_rest = [rnd.random() for _ in others]
    rest_sum = sum(raw_rest) or 1.0

    scores = {winner: top}
    for lab, r in zip(others, raw_rest):
        scores[lab] = (1.0 - top) * r / rest_sum
    return winner, scores


# --------------------------------------------------------------------------- #
# 五、YOLO 后端
# --------------------------------------------------------------------------- #

def _yolo_predict(image_path: str, weights_path: str) -> tuple[str, dict[str, float], str]:
    """执行 YOLOv8-cls 推理，并把 7 分类概率聚合到对外 5 类。

    Returns:
        (顶层类别中文名, {中文类别: 概率}, 原始 top-1 类别英文名)

    Raises:
        RuntimeError: 模型输出的类别名无法与映射表对齐时抛出，
            这属于"训练时的类别顺序与代码不一致"，必须显式报错而非静默出错。
    """
    from ultralytics import YOLO  # 延迟导入：未安装时不影响其他工具

    _console.install()  # 见 _resolve_backend：ultralytics 会改 stdout 编码
    model = YOLO(weights_path)
    results = model.predict(image_path, verbose=False)[0]
    probs = results.probs
    if probs is None:
        raise RuntimeError("模型未返回分类概率，请确认使用的是 YOLOv8-cls 分类权重而非检测权重")

    names: dict[int, str] = results.names  # type: ignore[assignment]
    raw_scores: dict[str, float] = {
        names[i]: float(p) for i, p in enumerate(probs.data.tolist())
    }

    # ---- 关键：多对一映射要「求和」而不是「取最大」 ----
    agg: dict[str, float] = {lab: 0.0 for lab in DISEASE_LABELS}
    unmapped: list[str] = []
    for raw_name, p in raw_scores.items():
        target = RAW_CLASS_TO_LABEL.get(raw_name)
        if target is None:
            unmapped.append(raw_name)
            target = "其他" if ALLOW_OTHER_FALLBACK else "出血"
        agg[target] += p

    if unmapped:
        raise RuntimeError(
            f"模型输出的类别 {unmapped} 未在 RAW_CLASS_TO_LABEL 中登记。"
            f"请先更新 src/fisheries/tools/fish_disease.py 的映射表，再重新推理。"
        )

    raw_top = max(raw_scores, key=lambda k: raw_scores[k])
    top_label = max(agg, key=lambda k: agg[k])
    return top_label, agg, raw_top


# --------------------------------------------------------------------------- #
# 六、对外工具函数
# --------------------------------------------------------------------------- #

def fish_disease_detect(
    image_path: str,
    conf_threshold: float = 0.25,
    top_k: int = 4,
    weights_path: str | None = None,
    force_mock: bool = False,
    mock_source: str = MOCK_SOURCE_DEFAULT,
) -> dict[str, Any]:
    """【工具】鱼体病害图像识别。

    入参
    ----
    image_path : str
        鱼体图片的本地绝对路径或 http(s) URL。接到 URL 时会先尝试用
        `ultralytics` 直接读取；本地路径则先做存在性与格式校验。
    conf_threshold : float
        最低置信度阈值，默认 0.25。低于该值仍返回结果，但
        `is_low_confidence` 会被置为 True。
    top_k : int
        返回概率最高的前 K 个类别，默认 4。
    weights_path : str | None
        自定义权重路径；None 时使用 `models/fish_disease_cls.pt`。
    force_mock : bool
        强制使用 mock 后端。仅供单元测试与链路联调。
    mock_source : str
        mock 后端的结果来源，``"hash"``（默认，与标签无关）或 ``"folder_hint"``
        （从文件名/目录名读标签）。**该参数不暴露给大模型**，只供开发期联调：
        模型权重尚未训练时，用它把「图像 → 病害 → 融合决策」这条链路先跑通。
        使用时返回值里会带 ``model_version="mock-v0.1+folder_hint"`` 且 note 明确标注。

    返回
    ----
    dict
        可直接 JSON 序列化；失败时返回 `status="error"` 而非抛异常。

    Examples:
        >>> out = fish_disease_detect("/path/to/gill_01.jpg", force_mock=True)
        >>> out["status"]
        'ok'
        >>> out["disease_type"] in ["烂鳃", "出血", "水霉", "健康", "其他"]
        True
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
                image_path, "UNSUPPORTED_IMAGE",
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
    backend, weights = ("mock", None) if force_mock else _resolve_backend(weights_path)

    raw_class: str | None = None
    hint_used = False
    if backend == "yolo" and weights:
        try:
            top_label, scores, raw_class = _yolo_predict(display_path, weights)
            model_version = os.path.basename(weights)
            degraded = False
        except Exception as exc:  # noqa: BLE001 —— 模型不可用时静默降级，保证 Agent 不断链
            top_label, scores = _mock_scores(display_path, source=mock_source)
            model_version = f"{MOCK_MODEL_VERSION} (yolo failed: {type(exc).__name__})"
            degraded = True
    else:
        top_label, scores = _mock_scores(display_path, source=mock_source)
        model_version = MOCK_MODEL_VERSION
        degraded = True

    # 标签提示只在降级（mock）路径上生效；真实 YOLO 推理成功时绝不读文件名
    hint_raw: str | None = None
    if degraded and mock_source == "folder_hint":
        hint_raw = _folder_hint(display_path)
        model_version += _MOCK_HINT_TAG
        hint_used = hint_raw is not None

    if hint_used:
        raw_class = hint_raw
    elif degraded:
        raw_class = _guess_raw_class(top_label) if not ALLOW_OTHER_FALLBACK else None

    # 归一化（浮点误差保护）
    total = sum(scores.values()) or 1.0
    scores = {k: v / total for k, v in scores.items()}
    confidence = scores[top_label]
    top_sorted = dict(
        sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max(1, top_k)]
    )

    is_low_conf = confidence < max(conf_threshold, LOW_CONFIDENCE_THRESHOLD)

    # ---- 3. 生成提示语 ----
    if backend == "mock" or degraded:
        note = (
            f"⚠ 当前为演示/Mock 结果（model_version={model_version}），"
            f"非真实模型输出，不可作为诊断依据。"
        )
        if hint_used:
            note += (
                f" 本次标签取自文件路径中的类别名提示（{hint_raw}），"
                f"**不是图像推理结果**，仅用于模型训练完成前的链路联调。"
            )
    elif top_label == "健康":
        note = "未检出明显病害特征。"
    elif top_label == "其他":
        note = (
            f"检出病灶，但不属于烂鳃/出血/水霉三类"
            f"（底层模型判定：{raw_class or '未知'}），建议人工复核。"
        )
    else:
        note = f"检出{LB_GROUP.get(top_label, '')}{top_label}特征，建议结合水质数据综合研判。"

    if is_low_conf:
        note += f" 置信度 {confidence:.2f} 偏低（阈值 {max(conf_threshold, LOW_CONFIDENCE_THRESHOLD):.2f}），建议补拍清晰照片或人工复核。"

    return FishDiseaseOutput(
        tool="fish_disease_detect",
        status="ok",
        image_path=display_path,
        disease_type=top_label,
        disease_group=LABEL_META[top_label]["group"],
        confidence=round(confidence, 4),
        is_healthy=(top_label == "健康"),
        severity_weight=LABEL_META[top_label]["severity_weight"],
        is_low_confidence=is_low_conf,
        all_scores={k: round(v, 4) for k, v in top_sorted.items()},
        raw_class=raw_class,
        model_version=model_version,
        degraded_mode=degraded,
        image_meta=image_meta,
        note=note,
    ).model_dump(mode="json")


def _guess_raw_class(label: str) -> str | None:
    """在关闭"其他"兜底时，反查该中文标签对应的典型原始类别名。"""
    for raw, lab in RAW_CLASS_TO_LABEL.items():
        if lab == label:
            return raw
    return None


def _error(image_path: str, code: str, message: str) -> dict[str, Any]:
    """构造统一的错误返回体。"""
    return FishDiseaseOutput(
        tool="fish_disease_detect",
        status="error",
        image_path=image_path,
        model_version=MOCK_MODEL_VERSION,
        degraded_mode=True,
        error_code=code,
        error_message=message,
        note=f"图片检测失败（{code}）：{message}",
    ).model_dump(mode="json")
