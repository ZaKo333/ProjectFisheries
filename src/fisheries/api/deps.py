"""接口层的运行时状态与依赖注入。

这里要解决三个在「把 CLI 脚本包成 HTTP 服务」时必然撞上的问题：

1. **别在每个请求里重建贵对象。**
   `AquacultureAgent` 第一次用到知识库时才打开向量库（要加载 90 MB 的
   中文向量模型）。如果把构建放进请求处理函数，第一个请求会莫名卡十几秒，
   并发时会重复加载。放在 ``lifespan`` 里初始化一次。

2. **Agent 实例不是线程安全的。**
   `AquacultureAgent.run()` 开头会 ``self.trace = []``、
   ``self.degraded_reasons = []``，运行期间往里追加，最后打包进返回值。
   两个请求并发调用同一个实例，轨迹和降级说明会互相串台——
   这不是理论风险，FastAPI 的同步端点就跑在线程池里。
   本模块用一把 ``RLock`` 串行化 ``run()``。
   > 根治办法是把 ``run()`` 改成无副作用（轨迹随返回值传递而不是挂在实例上），
   > 那是一次涉及 orchestrator 全文的重构，不属于 M7 的改动范围，
   > 已在 README「后续迭代」里记为待办。对「一个塘口看板」的并发量，
   > 串行化完全够用，且绝不会出错。

3. **会话数据必须按会话隔离，而且不能放在进程内存里。**
   原来三个 dict 只用 ``pond_id`` 当键（默认值还是 ``"unknown"``）：
   两个浏览器用同一个塘号就会互相看到对方的读数，进程一重启又全丢。
   现在交给 ``store.SessionStore``（SQLite），键是 ``(session_id, pond_id)``。
   上限仍然保留：读数按 ``buffer_max``、历史按 ``history_max`` 截断。
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent.orchestrator import (
    KNOWLEDGE_DISABLE_ENV,
    AquacultureAgent,
    knowledge_enabled_by_default,
)
from ..tools.water_risk import MAX_EXTRAPOLATION_HOURS, MIN_RELIABLE_POINTS, PERSISTENCE_CAP
from .store import SessionStore

__all__ = ["ApiSettings", "AppState", "get_state", "project_root"]


def project_root() -> Path:
    """项目根目录（``project/``）。

    用 ``__file__`` 推算而不是 ``Path.cwd()``：服务可能被从任意工作目录启动
    （uvicorn 的 ``--app-dir``、IDE 的运行配置、systemd 的 ``WorkingDirectory``），
    依赖 cwd 会让 ``knowledge/chroma`` 找错地方，表现为「知识库凭空消失」。
    """
    return Path(__file__).resolve().parents[3]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass
class ApiSettings:
    """接口层的可调参数，全部可通过环境变量覆盖（便于部署时不改代码）。"""

    upload_dir: Path = field(default_factory=lambda: project_root() / "data" / "uploads")
    #: 是否允许 /assess 直接读取服务器本地任意路径的图片。
    #: 演示与内网联调需要（demo 直接用数据集里的图），
    #: 但这是一个**任意文件读取**面，对公网部署必须关掉，改走 /ingest/image 上传。
    allow_local_path: bool = True
    max_upload_mb: int = 10
    history_max: int = 200
    buffer_max: int = 5000
    #: 会话库（SQLite）落点。None 表示跟着 ``upload_dir`` 走：
    #: 默认 ``data/uploads/api_state.sqlite3``。放上传目录旁边是刻意的——
    #: 换个上传目录就等于换一套会话数据，测试与演示天然隔离。
    state_db: Path | None = None
    api_title: str = "ProjectFisheries Agent API"
    api_version: str = "0.1.0"

    @classmethod
    def from_env(cls) -> ApiSettings:
        root = project_root()
        raw_dir = os.environ.get("FISHERIES_API_UPLOAD_DIR")
        upload_dir = Path(raw_dir) if raw_dir else root / "data" / "uploads"
        raw_db = os.environ.get("FISHERIES_API_STATE_DB")
        state_db = Path(raw_db) if raw_db else None
        return cls(
            upload_dir=upload_dir,
            allow_local_path=_env_bool("FISHERIES_API_ALLOW_LOCAL_PATH", True),
            max_upload_mb=_env_int("FISHERIES_API_MAX_UPLOAD_MB", 10),
            history_max=_env_int("FISHERIES_API_HISTORY_MAX", 200),
            buffer_max=_env_int("FISHERIES_API_BUFFER_MAX", 5000),
            state_db=state_db,
        )


class AppState:
    """进程级单例状态。由 ``create_app()`` 的 lifespan 建立。"""

    def __init__(self, settings: ApiSettings | None = None) -> None:
        self.settings = settings or ApiSettings.from_env()
        self.started_at = time.time()

        #: 串行化 agent.run()，理由见模块 docstring 第 2 条。
        self.run_lock = threading.RLock()
        #: 保护下面的三个内存字典。
        self.data_lock = threading.RLock()

        self.agent: AquacultureAgent = AquacultureAgent()

        # 这些容器必须在 `_prepare_knowledge()` **之前**建好。
        # `_prepare_knowledge()` 在降级路径里会往 `self.notes` 追加提示，
        # 如果它排在后面，知识库一旦不可用就会抛 AttributeError——
        # 恰好把「优雅降级」写成了「启动即崩」，而那正是最需要它工作的场景。
        # 会话数据（读数 / 历史 / 上传）统一走 SQLite：
        # 键是 (session_id, pond_id)，进程重启不丢，两个客户端也不会串。
        db_path = self.settings.state_db or (self.settings.upload_dir / "api_state.sqlite3")
        self.store = SessionStore(
            db_path,
            buffer_max=self.settings.buffer_max,
            history_max=self.settings.history_max,
        )
        self.notes: list[str] = []

        #: 启动时就把知识库打开/判定好，避免首个请求卡顿，也避免并发重复加载。
        self.knowledge_status: dict[str, Any] = {}
        self._prepare_knowledge()

        if self.settings.allow_local_path:
            self.notes.append(
                "已开启「按服务器路径读图」（FISHERIES_API_ALLOW_LOCAL_PATH=1），"
                "便于演示与内网联调；公网部署请关闭并改用 /api/v1/ingest/image 上传。"
            )

    # ------------------------------------------------------------------ #
    # 知识库
    # ------------------------------------------------------------------ #

    def _prepare_knowledge(self) -> None:
        """在启动阶段确定知识库可用性，并把结果记进 `knowledge_status`。"""
        if not knowledge_enabled_by_default():
            self.knowledge_status = {
                "status": "disabled",
                "detail": f"环境变量 {KNOWLEDGE_DISABLE_ENV} 已开启，本次运行不做 RAG 检索",
                "backend": None,
                "n_chunks": 0,
            }
            self.use_knowledge = False
            self.knowledge_base = None
            return

        try:
            from ..rag.retriever import open_knowledge_base

            kb = open_knowledge_base(auto_build=False)
        except Exception as exc:  # noqa: BLE001 —— 任何异常都只降级，不让服务起不来
            self.knowledge_base = None
            self.use_knowledge = False
            self.knowledge_status = {
                "status": "unavailable",
                "detail": f"{type(exc).__name__}: {exc}",
                "backend": None,
                "n_chunks": 0,
                "hint": "先跑 scripts/build_knowledge_base.py 建库",
            }
            self.notes.append(
                "知识库不可用，养殖建议将不含规范引证（告警与风险等级不受影响）。"
            )
            return

        self.knowledge_base = kb
        self.use_knowledge = True
        self.knowledge_status = {
            "status": "ready",
            "detail": "已加载混合检索知识库（稠密 + BM25）",
            "backend": getattr(kb, "backend", None),
            "n_chunks": len(getattr(kb, "ids", []) or []),
        }

    # ------------------------------------------------------------------ #
    # 会话数据（SQLite，按 session 隔离，进程重启不丢）
    # ------------------------------------------------------------------ #

    def append_readings(
        self,
        session_id: str,
        pond_id: str,
        readings: list[dict[str, Any]],
        *,
        replace: bool = False,
    ) -> tuple[int, str | None, str | None]:
        """把读数写进**这个会话的**缓冲，返回 (总数, 首条时间戳, 末条时间戳)。"""
        return self.store.append_readings(session_id, pond_id, readings, replace=replace)

    def get_readings(self, session_id: str, pond_id: str) -> list[dict[str, Any]]:
        return self.store.get_readings(session_id, pond_id)

    def register_upload(self, session_id: str, upload_id: str, path: Path) -> None:
        self.store.register_upload(session_id, upload_id, path)

    def lookup_upload(self, session_id: str, upload_id: str) -> Path | None:
        return self.store.lookup_upload(session_id, upload_id)

    def record(self, session_id: str, pond_id: str, result: Any) -> None:
        self.store.record(session_id, pond_id, result)

    def get_history(self, session_id: str, pond_id: str) -> list[Any]:
        return self.store.get_history(session_id, pond_id)

    def close(self) -> None:
        """释放 SQLite 连接（lifespan 结束时调用）。"""
        self.store.close()

    # ------------------------------------------------------------------ #

    def describe_tools(self) -> dict[str, str]:
        """给 /health 用：各工具当前是真实模型还是 mock 兜底。

        判据是**两个条件同时成立**：``ultralytics`` 可导入 **且** 默认权重文件存在。
        只看 ``ultralytics`` 会谎报——装了依赖但没训练权重时，工具实际仍在走 mock，
        而 /health 若报成 yolov8-cls，运维就会以为图像模态已经上线。
        权重是否存在通过 ``fish_disease.resolve_weights()`` 这个公开查询口问，
        不去 import 私有常量，保持接口层只依赖被依赖方的公开契约。
        """
        import importlib.util

        from fisheries.tools import fish_disease

        weights = os.environ.get("FISHERIES_VISION_WEIGHTS")
        resolved = fish_disease.resolve_weights(weights)
        has_ultralytics = importlib.util.find_spec("ultralytics") is not None

        if resolved and has_ultralytics:
            vision = f"yolov8-cls（权重 {os.path.basename(resolved)}）"
        elif resolved and not has_ultralytics:
            vision = "mock（找到权重但 ultralytics 未安装，输出带 degraded_mode=true）"
        elif has_ultralytics:
            vision = (
                "mock（ultralytics 已安装但权重未就绪："
                f"缺 {fish_disease.DEFAULT_WEIGHTS_PATH}，输出带 degraded_mode=true）"
            )
        else:
            vision = "mock（ultralytics 未安装 / 权重未就绪，输出带 degraded_mode=true）"
        return {
            "water_risk_predict": "rule-baseline（阈值规则 + 混合基线外推；LSTM 待接入）",
            "fish_disease_detect": vision,
        }

    def dump_config(self) -> dict[str, Any]:
        s = self.settings
        return {
            "upload_dir": str(s.upload_dir),
            "allow_local_path": s.allow_local_path,
            "max_upload_mb": s.max_upload_mb,
            "history_max": s.history_max,
            "buffer_max": s.buffer_max,
            "state_db": str(s.state_db or (s.upload_dir / "api_state.sqlite3")),
            "extrapolation_horizon_hours": MAX_EXTRAPOLATION_HOURS,
            "min_reliable_points": MIN_RELIABLE_POINTS,
            "persistence_cap": PERSISTENCE_CAP,
        }


#: 单例挂在 FastAPI 的 ``app.state`` 上（见 app.py 的 lifespan）。
_STATE_ATTR = "fisheries_state"


def get_state(request: Any) -> AppState:
    """FastAPI 依赖：从 ``request.app.state`` 取回单例。"""
    state = getattr(request.app.state, _STATE_ATTR, None)
    if state is None:  # pragma: no cover —— 正常由 lifespan 保证
        raise RuntimeError("AppState 未初始化；请通过 create_app() 构建应用")
    return state
