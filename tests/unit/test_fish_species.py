"""鱼种识别工具（``fisheries.tools.fish_species``）的单元测试。

这里钉的是**没有模型时也必须有确定行为**的那一层：

* 类别名 → 三个阈值档案 的映射（数据集命名五花八门，映射错就等于判错档位）；
* 模型缺位时**不编造品种**（这点和病害工具的 mock 策略刻意相反，见模块文档）；
* 输入校验（不存在 / 不是文件 / 不是图片）。

真实的 YOLO 推理不在覆盖范围内：那要求 ``models/yolov8-species.pt`` 存在，
而权重是训练产物、不入库。所以这里全部离线，不碰 GPU 也不碰数据集。
"""

from __future__ import annotations

import base64
import os

import pytest

from fisheries.tools import TOOL_HANDLERS
from fisheries.tools.fish_species import (
    DEFAULT_WEIGHTS_PATH,
    RAW_SPECIES_ALIASES,
    SPECIES_LABELS,
    UNAVAILABLE_MODEL_VERSION,
    availability,
    fish_species_detect,
    map_raw_species,
    resolve_weights,
)
from fisheries.tools.water_risk import SPECIES_PROFILES

#: 1×1 的合法 PNG，用来做「文件存在且是图片」的输入。
_PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def png_file(tmp_path):
    """一张真实的图片文件（不依赖 data/datasets，克隆下来就能跑）。"""
    path = tmp_path / "fish.png"
    path.write_bytes(_PNG_1PX)
    return path


# --------------------------------------------------------------------------- #
# 一、标签体系
# --------------------------------------------------------------------------- #

class TestSpeciesTaxonomy:
    def test_labels_are_derived_from_threshold_profiles(self):
        """对外品种必须等于有阈值档案的品种，不能各抄一份。"""
        assert tuple(SPECIES_PROFILES) == SPECIES_LABELS
        assert SPECIES_LABELS == ("鲤科", "罗非鱼", "虾蟹")

    def test_default_weights_path_lives_under_models(self):
        assert DEFAULT_WEIGHTS_PATH.endswith("yolov8-species.pt")
        assert "models" in DEFAULT_WEIGHTS_PATH.replace("\\", "/")

    def test_every_alias_is_non_empty_and_normalised(self):
        """别名表里混进空串会让 ``in`` 判断恒真，整张表失效。"""
        for species, aliases in RAW_SPECIES_ALIASES.items():
            assert species in SPECIES_LABELS
            assert aliases, f"{species} 没有任何别名"
            for alias in aliases:
                assert alias.strip(), f"{species} 有空别名"
                assert alias == alias.strip()

    def test_species_detector_is_not_exposed_as_an_llm_tool(self):
        """品种来自用户上传的图，不该让大模型自己决定调不调。

        注册进 ``TOOL_HANDLERS`` 会把它写进 ``TOOL_SCHEMAS``，等于把"要不要看图"
        交给模型判断；而这条链路的输入是页面上的文件框，由接口层直接触发。
        """
        assert "fish_species_detect" not in TOOL_HANDLERS


# --------------------------------------------------------------------------- #
# 二、底层类别名 → 品种档案
# --------------------------------------------------------------------------- #

class TestRawClassMapping:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # 罗非鱼：拉丁属名 / 英文俗名 / 中文
            ("Nile Tilapia", "罗非鱼"),
            ("tilapia", "罗非鱼"),
            ("Oreochromis niloticus", "罗非鱼"),
            ("罗非鱼", "罗非鱼"),
            # 鲤科：多种写法，含空格与下划线
            ("Grass_Carp", "鲤科"),
            ("Carp", "鲤科"),
            ("Goldfish", "鲤科"),
            ("Rohu", "鲤科"),
            ("silver carp", "鲤科"),
            ("鲤", "鲤科"),
            ("鲫鱼", "鲤科"),
            # 虾蟹
            ("Freshwater Prawn", "虾蟹"),
            ("crab", "虾蟹"),
            ("Crayfish", "虾蟹"),
            ("虾", "虾蟹"),
        ],
    )
    def test_known_names_map_to_a_profile(self, raw, expected):
        assert map_raw_species(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["Catfish", "Fish", "Anabas", "", "   ", "unknown-class-0"],
    )
    def test_unknown_names_stay_unknown(self, raw):
        """收敛不了就返回 None —— 硬塞进鲤科会让氨氮警戒线宽一倍。"""
        assert map_raw_species(raw) is None

    def test_case_and_separator_are_ignored(self):
        assert map_raw_species("NILE_TILAPIA") == map_raw_species("nile-tilapia") == "罗非鱼"

    def test_longer_alias_wins_over_shorter_one(self):
        """``grass carp`` 与 ``carp`` 同档，但匹配顺序必须先长后短。"""
        assert map_raw_species("Grass Carp") == "鲤科"
        assert map_raw_species("carp") == "鲤科"


# --------------------------------------------------------------------------- #
# 三、模型缺位时的行为
# --------------------------------------------------------------------------- #

class TestWithoutModel:
    def test_resolve_weights_returns_none_for_missing_explicit_path(self, tmp_path):
        assert resolve_weights(str(tmp_path / "nope.pt")) is None

    def test_returns_unavailable_instead_of_a_random_species(self, png_file):
        """没有权重时**不许**编一个品种：品种决定整套阈值，编错就是错档结论。"""
        out = fish_species_detect(str(png_file), weights_path=str(png_file) + ".missing")

        assert out["status"] == "ok"
        assert out["detected"] is False
        assert out["species"] is None
        assert out["degraded_mode"] is True
        assert out["model_version"] == UNAVAILABLE_MODEL_VERSION
        assert "未就绪" in out["note"]

    def test_result_is_deterministic(self, png_file):
        """同一张图跑两次必须完全一样（不存在随机兜底）。"""
        kwargs = {"weights_path": str(png_file) + ".missing"}
        first = fish_species_detect(str(png_file), **kwargs)
        second = fish_species_detect(str(png_file), **kwargs)
        assert first == second


# --------------------------------------------------------------------------- #
# 四、输入校验
# --------------------------------------------------------------------------- #

class TestInputValidation:
    def test_missing_file_is_reported_not_raised(self, tmp_path):
        out = fish_species_detect(str(tmp_path / "absent.png"))
        assert out["status"] == "error"
        assert out["error_code"] == "IMAGE_NOT_FOUND"

    def test_directory_is_reported_as_not_a_file(self, tmp_path):
        out = fish_species_detect(str(tmp_path))
        assert out["status"] == "error"
        assert out["error_code"] == "NOT_A_FILE"

    def test_non_image_is_reported(self, tmp_path):
        path = tmp_path / "note.txt"
        path.write_text("这不是图片", encoding="utf-8")
        out = fish_species_detect(str(path))
        assert out["status"] == "error"
        assert out["error_code"] == "UNSUPPORTED_IMAGE"

    def test_error_body_still_carries_a_model_version(self, tmp_path):
        """错误体也要能被前端统一渲染，所以字段必须齐。"""
        out = fish_species_detect(str(tmp_path / "absent.png"))
        assert out["model_version"] == UNAVAILABLE_MODEL_VERSION
        assert out["degraded_mode"] is True
        assert out["detected"] is False
        assert out["note"]


# --------------------------------------------------------------------------- #
# 五、可用性探测（前端据此显示或隐藏鱼种模块）
# --------------------------------------------------------------------------- #

class TestAvailability:
    def test_missing_weights_hide_the_module_and_say_how_to_get_them(self, tmp_path):
        info = availability(str(tmp_path / "absent.pt"))

        assert info["available"] is False
        assert info["reason"] == "缺失权重"
        assert info["weights"] is None
        # 对外只给「相对工程根」的路径：公开接口不能泄露部署目录结构
        assert info["expected"] == "models/yolov8-species.pt"
        assert info["species"] == list(SPECIES_LABELS)
        # 提示里必须给出「怎么补」，否则用户只知道坏了、不知道怎么办
        assert "--fetch-weights" in info["message"]
        assert "隐藏" in info["message"]

    def test_weights_without_ultralytics_are_reported_separately(
        self, tmp_path, monkeypatch
    ):
        """两种不可用要能分开：缺文件 vs 缺依赖，补法完全不同。"""
        import sys

        weights = tmp_path / "yolov8-species.pt"
        weights.write_bytes(b"stub")
        monkeypatch.setitem(sys.modules, "ultralytics", None)  # import 时会抛 ImportError

        info = availability(str(weights))

        assert info["available"] is False
        assert info["reason"] == "未安装 ultralytics"
        # 文件确实在，所以要把文件名报出来，而不是说「缺失」（但只报文件名，不报目录）
        assert info["weights"] == "yolov8-species.pt"
        assert "ultralytics" in info["message"]

    def test_available_when_both_weights_and_ultralytics_exist(
        self, tmp_path, monkeypatch
    ):
        import sys
        import types

        weights = tmp_path / "yolov8-species.pt"
        weights.write_bytes(b"stub")
        monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(__version__="test"))

        info = availability(str(weights))

        assert info["available"] is True
        assert info["reason"] == ""
        assert info["weights"] == "yolov8-species.pt"
        assert info["message"] == ""

    def test_the_two_registries_agree_on_the_file(self):
        """权重注册表（weights.py）与工具默认路径必须指向同一个文件。

        两处各写一份路径迟早会漂：注册表改了、工具没改，于是
        `--fetch-weights` 下了一个没人用的文件，模块照样隐藏。
        """
        from fisheries import weights as wt

        assert wt.spec("fish_species").file == "models/yolov8-species.pt"
        assert os.path.basename(DEFAULT_WEIGHTS_PATH) == "yolov8-species.pt"


class TestPublicPaths:
    """`/api/v1/species` 是公开接口，绝不能把绝对路径吐给客户端。

    生产上曾经直接回 `C:\\fisheries\\app\\models\\yolov8-species.pt`，
    既泄露部署布局，又和「快照里不得出现本机路径」的守卫自相矛盾。
    """

    def test_availability_only_exposes_relative_paths(self):
        info = availability()

        for key in ("expected", "weights"):
            value = info[key]
            if value is None:
                continue
            assert not value.startswith(("/", "\\")), value
            assert ":" not in value, value

    def test_paths_inside_the_project_are_relative_to_it(self):
        from fisheries.tools.fish_species import _public_path

        assert _public_path(DEFAULT_WEIGHTS_PATH) == "models/yolov8-species.pt"

    def test_paths_outside_the_project_are_reduced_to_the_file_name(self):
        from fisheries.tools.fish_species import _public_path

        assert _public_path(r"C:\Users\someone\else\yolov8-species.pt") == "yolov8-species.pt"

    def test_empty_inputs_stay_none(self):
        from fisheries.tools.fish_species import _public_path

        assert _public_path(None) is None
        assert _public_path("") is None
