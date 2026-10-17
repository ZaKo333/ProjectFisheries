"""P0 主链路①：上传一张真实的病害图片 → 接口识别 → Agent 给处置建议。

这条链路单元测试覆盖不到的地方：``/api/v1/ingest/image`` 落盘后的路径必须能被
``fish_disease_detect`` 真模型读进去（权重、类别表、图像解码、病灶分割四处约定），
识别结果又要被 Agent 翻成「等级 + 处置动作」。任何一环错位，单测都还是绿的。
"""

from __future__ import annotations

import pytest
from conftest import DISEASE_LABELS, assert_level, level_index, post_assess

pytestmark = pytest.mark.integration


def test_uploaded_gill_image_is_diagnosed_by_the_real_model(client, gill_image):
    """上传 → 200，且落到响应里的诊断来自真模型（不是 mock 兜底）。"""
    with gill_image.open("rb") as fh:
        resp = client.post(
            "/api/v1/ingest/image",
            files={"file": (gill_image.name, fh, "image/jpeg")},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()

    # 契约字段：前端就是靠这几个键渲染预览与后续评估
    for key in ("upload_id", "filename", "size_bytes", "content_type", "stored_path", "vision"):
        assert key in body, f"上传响应缺字段 {key}"

    assert body["filename"] == gill_image.name
    assert body["size_bytes"] == gill_image.stat().st_size
    assert body["upload_id"]

    vision = body["vision"]
    assert vision["status"] == "ok", vision
    assert vision["degraded_mode"] is False, "集成测试要的就是真模型，这里退化成 mock 了"
    assert vision["disease_type"] in DISEASE_LABELS
    assert 0.0 <= vision["confidence"] <= 1.0
    assert vision["is_low_confidence"] is False
    # 这张图来自数据集的「细菌性烂鳃病」目录：真模型应该认出烂鳃，认错说明回归了
    assert vision["disease_type"] == "烂鳃", vision["all_scores"]
    assert vision["raw_class"] == "gill_rot"
    assert set(vision["all_scores"]) == {"烂鳃", "出血", "水霉", "健康"}
    assert abs(sum(vision["all_scores"].values()) - 1.0) < 0.01
    # 病灶分割权重在场时才会给出面积（模型版本号里带 lesion）
    assert "yolov8-fish.pt" in vision["model_version"]


def test_disease_image_flows_into_the_agent_report(client, gill_image):
    """同一张图喂给 ``/assess``：等级、证据、处置动作三样都要对得上。"""
    payload = post_assess(client, pond_id="IT-DISEASE-1", image_path=str(gill_image))

    assert payload["pond_id"] == "IT-DISEASE-1"
    assert_level(payload["risk_level"])
    # 检出烂鳃（severity 0.85）→ 不可能判「正常」
    assert level_index(payload["risk_level"]) >= level_index("预警"), payload["reason"]
    assert 0.0 <= float(payload["fused_score"]) <= 100.0

    codes = [e["code"] for e in payload["evidence"] if e["source"] == "vision"]
    assert "VISION_DISEASE" in codes
    assert "VISION_RAW_CLASS" in codes

    report = payload["report"]
    assert report["risk_level"] == payload["risk_level"]
    assert report["disease_actions"], "检出了病害却没有任何病害处置动作"
    first = report["disease_actions"][0]
    for key in ("priority", "target", "action", "timing"):
        assert key in first, f"处置动作缺字段 {key}"
    assert "烂鳃" in first["target"]

    # 只给了图片：处置动作以病害为主，水质侧顶多顺带一条（烂鳃要停料）
    assert isinstance(report["water_actions"], list)
    assert payload["headline"]
    assert payload["observation_window"]

    # 图像模态的评估结果要和上传接口的识别一致（同一张图、同一份权重）
    assert payload["vision"]["disease_type"] == "烂鳃"
