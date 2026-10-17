"""P0 主链路⑤：品种识别 → 换阈值档位。

为什么这条要端到端跑：品种是**水质阈值的开关**（鲤科和虾蟹的溶氧红线不一样）。
识别结果必须能从接口回到评估请求里，并且真的改变结论——单测只验了
``availability()`` 和映射表，验不了「页面上的那条链路接得通」。
"""

from __future__ import annotations

import pytest
from conftest import (
    SPECIES_LABELS,
    level_index,
    post_assess,
    readings_from_template,
)

pytestmark = pytest.mark.integration


def test_species_availability_reports_missing_weights_explicitly(client, species_ready):
    """可用性接口：有权重就说可用，没权重必须说明原因和补法，不能含糊。"""
    resp = client.get("/api/v1/species")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["available"] is species_ready
    assert set(SPECIES_LABELS) == set(body["species"])
    # expected 是「权重应该在哪个路径」，前端据此显示排查用的路径
    assert body["expected"].endswith("yolov8-species.pt")
    if species_ready:
        assert body["weights"].endswith("yolov8-species.pt")
        assert body["message"] == ""
    else:  # pragma: no cover - 取决于机器
        assert body["reason"]
        assert "隐藏" in body["message"]
        assert "--fetch-weights" in body["message"]

def test_detect_returns_one_of_the_three_profiles(client, gill_image, species_ready):
    """传一张鲤科图片：识别结果落在三个档位里，置信度与模型版本都要带回来。"""
    if not species_ready:  # pragma: no cover - 取决于机器
        pytest.skip("缺鱼种权重：`fetch_datasets.py --fetch-weights fish_species`")

    with gill_image.open("rb") as fh:
        resp = client.post(
            "/api/v1/species/detect",
            files={"file": (gill_image.name, fh, "image/jpeg")},
        )

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    # 上传三件套 + 识别结果（和 /ingest/image 同一个落盘器，字段命名保持一致）
    for key in ("upload_id", "filename", "size_bytes", "stored_path", "species"):
        assert key in payload, f"响应缺字段 {key}"
    assert payload["filename"] == gill_image.name
    assert payload["size_bytes"] == gill_image.stat().st_size

    result = payload["species"]
    assert result["detected"] is True, result.get("error_message") or result
    assert result["species"] in SPECIES_LABELS
    assert result["species"] == "鲤科", result["all_scores"]  # 数据集里这张图就是鲤科
    assert 0.0 <= float(result["confidence"]) <= 1.0
    assert set(result["all_scores"]) == set(SPECIES_LABELS)
    assert abs(sum(result["all_scores"].values()) - 1.0) < 0.01
    assert result["is_low_confidence"] is False
    assert "yolov8-species.pt" in result["model_version"]
    assert result["degraded_mode"] is False, "鱼种模块没有 mock 兜底，这里不该是降级模式"
    assert result["image_meta"]["size_bytes"] > 0


def test_detected_species_switches_the_risk_profile(client, gill_image, templates, species_ready):
    """识别出「鲤科」后拿它去评估：结论要和直接按鲤科评估完全一致，且不同于虾蟹。"""
    if not species_ready:  # pragma: no cover - 取决于机器
        pytest.skip("缺鱼种权重：`fetch_datasets.py --fetch-weights fish_species`")

    with gill_image.open("rb") as fh:
        detected = client.post(
            "/api/v1/species/detect",
            files={"file": (gill_image.name, fh, "image/jpeg")},
        ).json()["species"]
    species = detected["species"]

    _, readings = readings_from_template(client, templates, "normal_shrimp")

    via_detection = post_assess(
        client, pond_id="IT-SPECIES-1", species=species, readings=readings, mock_source="hash"
    )
    by_hand = post_assess(
        client, pond_id="IT-SPECIES-1", species="鲤科", readings=readings, mock_source="hash"
    )
    as_shrimp = post_assess(
        client, pond_id="IT-SPECIES-1", species="虾蟹", readings=readings, mock_source="hash"
    )

    # 同一种水、同一品种 ⇒ 同一结论（识别结果和手选等价）
    assert via_detection["risk_level"] == by_hand["risk_level"]
    assert via_detection["water"]["risk_score"] == pytest.approx(by_hand["water"]["risk_score"])

    # 换成虾蟹就该更宽松：这正是「品种决定阈值档位」的直接证据
    assert level_index(via_detection["risk_level"]) >= level_index(as_shrimp["risk_level"])
    assert via_detection["water"]["species"] == "鲤科"
