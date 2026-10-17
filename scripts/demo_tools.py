"""工具层演示脚本 —— 直接运行即可看到真实输入输出。

用途：
    1. 答辩现场演示"Agent 调工具"的实际效果；
    2. 给 README / 文档提供**真实跑出来**的输入输出样例（不是手编的）。

运行：
    cd <工程>
    .venv\\Scripts\\python.exe scripts\\demo_tools.py

注意：图像识别当前默认走 Mock 后端（模型权重尚未训练），
输出中 degraded_mode=True 会明确标注这一点，避免与真实模型结果混淆。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 允许脚本直接运行时找到 src/ 下的包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Windows 控制台默认是 GBK 编码页，装饰符号编不出来会把脚本打断。
# 详见 src/fisheries/console.py。
from fisheries import console as _console

_console.install()

from fisheries import datasets as _datasets  # noqa: E402
from fisheries.tools import TOOL_SCHEMAS, fish_disease_detect, water_risk_predict  # noqa: E402


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)

def show(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))

# --------------------------------------------------------------------------- #
# 场景一：正常水质
# --------------------------------------------------------------------------- #
def case_normal_water() -> dict:
    """24 小时平稳数据，各项指标都在适宜区间。"""
    readings = [
        {
            "timestamp": f"2026-10-14T{i:02d}:00:00+08:00",
            "water_temp": 26.0 + (i % 3) * 0.2,
            "ph": 7.4,
            "dissolved_oxygen": 6.4,
            "ammonia_n": 0.10,
            "nitrite_n": 0.04,
        }
        for i in range(24)
    ]
    return water_risk_predict(readings, pond_id="POND-01", horizon_hours=72)

# --------------------------------------------------------------------------- #
# 场景二：持续恶化（本演示的核心场景）
# --------------------------------------------------------------------------- #
def case_degrading_water() -> dict:
    """24 小时持续恶化：溶氧从 6.2 一路掉到 2.6，氨氮同步爬升到 0.62。

    这组数据不是随便编的，它的形态对应高温晴天后的典型"倒藻—缺氧"过程：
    白天藻类过饱和产氧 → 夜间耗氧 → 凌晨溶氧谷值 → 氨氮因硝化受阻而累积。
    """
    readings = [
        {
            "timestamp": f"2026-07-18T{i:02d}:00:00+08:00",
            "water_temp": round(28.0 + 1.5 * (i / 23), 2),
            "ph": round(8.05 + 0.20 * (i / 23), 2),
            "dissolved_oxygen": round(6.2 - 3.6 * (i / 23) ** 1.3, 2),
            "ammonia_n": round(0.15 + 0.47 * (i / 23) ** 1.5, 3),
            "nitrite_n": round(0.05 + 0.10 * (i / 23), 3),
        }
        for i in range(24)
    ]
    return water_risk_predict(
        readings, pond_id="POND-03", horizon_hours=72, species="鲤科"
    )

# --------------------------------------------------------------------------- #
# 场景三：缺少亚硝酸盐探头的降级采集
# --------------------------------------------------------------------------- #
def case_missing_sensor() -> dict:
    """塘口没装亚硝酸盐探头：该指标缺失，其余指标权重自动归一化。"""
    readings = [
        {
            "water_temp": 30.0,
            "ph": 8.2,
            "dissolved_oxygen": 4.1,
            "ammonia_n": 0.30,
        }
        for _ in range(12)
    ]
    return water_risk_predict(readings, pond_id="POND-05", interval_minutes=60)

# --------------------------------------------------------------------------- #
# 场景四：图像识别
# --------------------------------------------------------------------------- #
SAMPLE_IMAGE = str(_datasets.resolve(
    "fish_disease", "Test", "Bacterial gill disease", "Bacterial gill disease (100).jpg"
))

def case_image() -> dict:
    """默认 mock 后端（hash 模式）——诚实地展示"模型还没训练，结果就是随机猜的"。"""
    return fish_disease_detect(SAMPLE_IMAGE, conf_threshold=0.25, top_k=4)

def case_image_hinted() -> dict:
    """folder_hint 模式——模型未训练期间用它把「图像 → 病害 → 融合」链路先跑通。

    注意 model_version 会带上 +folder_hint 后缀，note 也会写明"不是图像推理结果"。
    """
    return fish_disease_detect(
        SAMPLE_IMAGE, conf_threshold=0.25, top_k=4, mock_source="folder_hint"
    )

def case_image_missing() -> dict:
    """失败路径也要演示：Agent 必须能优雅处理图片不存在。"""
    return fish_disease_detect(r"D:\not\exist\fish.jpg")

def main() -> None:
    hr("工具声明（传给大模型的 function calling schema）")
    for name, schema in TOOL_SCHEMAS.items():
        fn = schema["function"]
        print(f"\n▸ {name}")
        print(f"  必填参数: {fn['parameters']['required']}")
        print(f"  可选参数: "
              f"{[k for k in fn['parameters']['properties'] if k not in fn['parameters']['required']]}")

    hr("【场景一】水质正常 —— water_risk_predict")
    show(case_normal_water())

    hr("【场景二】持续恶化 —— water_risk_predict")
    show(case_degrading_water())

    hr("【场景三】亚硝酸盐探头缺失 —— water_risk_predict")
    show(case_missing_sensor())

    hr("【场景四】鱼体图片识别 · 默认 mock（hash，与标签无关） —— fish_disease_detect")
    show(case_image())

    hr("【场景五】鱼体图片识别 · folder_hint（仅供联调，已显式标注） —— fish_disease_detect")
    show(case_image_hinted())

    hr("【场景六】图片不存在（失败路径） —— fish_disease_detect")
    show(case_image_missing())

if __name__ == "__main__":
    main()
