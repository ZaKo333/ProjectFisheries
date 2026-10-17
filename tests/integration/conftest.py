"""端到端集成测试的公共夹具。

为什么单开一层 ``tests/integration``
------------------------------------
``tests/unit`` 里的用例几乎都把「模型」「知识库」「数据集」换成替身：接口层关 RAG、
图像用任意字节、水质用内存里造的点。这样跑得快，但**测不到「真模型 + 真检索 + 真表格
解析器 + 真路由」串起来会不会散架**——比如权重和工具类约定不一致、CSV 列别名与规则
引擎口径不一致、检索回来的引文字段被接口层丢掉。

这一层只跑 P0 主链路，且**不 mock 任何路由函数**。缺料就 ``skip``：
缺训练好的权重、缺 ``ultralytics``、缺数据集、知识库索引没建，都是环境问题，
不是代码错了——``skip`` 的 reason 会写清补的办法。
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from fisheries import datasets, tables
from fisheries import weights as wt
from fisheries.api.app import create_app
from fisheries.api.deps import ApiSettings
from fisheries.evaluation.labels import RISK_LEVELS

#: 集成测试要用的那张真实图片：数据集里「细菌性烂鳃病」测试集的一张。
GILL_IMAGE_PARTS = ("Test", "Bacterial gill disease", "Bacterial gill disease (100).jpg")

#: 接口层能给出的病害类别（四分类 + 归并后的「其他」）。
DISEASE_LABELS = ("烂鳃", "出血", "水霉", "健康", "其他")

#: 品种档位（与水质阈值档案一一对应）。
SPECIES_LABELS = ("鲤科", "罗非鱼", "虾蟹")


def level_index(level: str) -> int:
    """把风险等级换成可比较的序数，用于「A 比 B 严重」这类断言。"""
    return RISK_LEVELS.index(level)


def assert_level(value: str) -> None:
    assert value in RISK_LEVELS, f"风险等级 {value!r} 不在 {RISK_LEVELS} 里"


@pytest.fixture(scope="session")
def client(tmp_path_factory) -> TestClient:
    """整个集成测试共用一个真 ASGI 应用（含 lifespan、RAG、中间件）。

    上传目录指向临时目录：跑测试不该往仓库的 ``data/uploads`` 里堆图片。
    """
    uploads = tmp_path_factory.mktemp("integration-uploads")
    app = create_app(ApiSettings(upload_dir=uploads))
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def templates(client: TestClient) -> dict[str, dict]:
    """``key -> 模板元数据``：里面带着 ``expected_risk``，是模板自带的验收值。"""
    body = client.get("/api/v1/config/templates").json()
    return {t["key"]: t for t in body["templates"]}


def readings_from_template(
    client: TestClient, templates: dict[str, dict], key: str
) -> tuple[dict, list[dict]]:
    """下载模板 CSV → 走真实表格解析器 → 得到喂给水质工具的读数。

    这一步故意不走捷径直接手写 24 个点：前端「上传水质表格」用的就是同一套列别名
    （``fisheries.tables``），模板 CSV 又分 ``units_en`` / ``zh`` / 简写三种表头，
    解析器认不认这三种表头，单元测试里是分文件验的，串起来验只有这里能覆盖。
    """
    tpl = templates[key]
    resp = client.get(tpl["csv_url"])
    assert resp.status_code == 200, f"模板 {key} 的 CSV 拿不到：{resp.status_code}"
    rows = list(csv.reader(io.StringIO(resp.text)))
    assert len(rows) > 1
    ingest = tables.ingest_rows(rows[0], rows[1:])
    assert ingest.ok, f"模板 {key} 解析后不可用：{ingest.errors}"
    assert ingest.n_points == tpl["n_points"], (
        f"模板 {key} 自称 {tpl['n_points']} 个点，解析出 {ingest.n_points} 个"
    )
    return tpl, ingest.readings


def post_assess(client: TestClient, **body) -> dict:
    """调一次 ``/api/v1/assess`` 并校验状态码，返回 body。"""
    resp = client.post("/api/v1/assess", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.fixture(scope="session")
def gill_image() -> Path:
    """真实病害图片；缺模型/缺数据集就 skip（附上补的办法）。"""
    if "fish_disease" in wt.missing():
        pytest.skip("缺病害识别权重：先跑 `fetch_datasets.py --fetch-weights all`（见 README 3.6）")
    try:
        import ultralytics  # noqa: F401
    except ImportError:  # pragma: no cover - 取决于机器
        pytest.skip("没装 ultralytics：`pip install -e .[vision]`")

    try:
        path = datasets.resolve("fish_disease", *GILL_IMAGE_PARTS)
    except Exception:  # noqa: BLE001 - pragma: no cover - 取决于机器
        pytest.skip("本地没有 Fish Disease Dataset：`fetch_datasets.py --fetch fish_disease`")
    if not path.is_file():  # pragma: no cover - 取决于机器
        pytest.skip(f"数据集里缺这张图：{path}")
    return path


@pytest.fixture(scope="session")
def species_ready() -> bool:
    """鱼种权重在不在；不在时调用方自己 skip（不同用例的降级方向不一样）。"""
    return not wt.missing(["fish_species"])


@pytest.fixture(scope="session")
def chat_answer(client: TestClient) -> dict:
    """跑一次真实知识库问答；索引没建就 skip。

    这条 fixture 同时是「离线部署能不能答」的探针：``docs/_api_samples.json`` 里的
    chat 样例就是同一个查询跑出来的。
    """
    resp = client.post("/api/v1/chat", json={"query": "溶氧低怎么办", "k": 3})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    if not body.get("chunks"):
        pytest.skip(
            f"知识库索引不可用（backend={body.get('backend')!r}）："
            "先跑 scripts/build_knowledge.py 建索引"
        )
    return body
