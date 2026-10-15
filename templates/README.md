# 水质模板

> 本目录由 `scripts/build_water_templates.py` 生成，**不要手工编辑**：
> 数值源头是 `src/fisheries/water_templates.py`，改模板请改代码后重跑脚本。

## 怎么用

三种用法，选一个就行：

1. **导入文件** —— 把任意 `<key>.csv` 上传到演示页的「水质时序表格」，或直接丢进你自己的工具链；
2. **用模板填充** —— 演示页左栏选好模板后点「📥 用该模板填充」，会自动填文件、切换品种、清零粘贴框；
3. **调接口** —— `GET /api/v1/config/templates` 拿目录，`GET /api/v1/config/templates/{key}/csv` 直接下载 CSV。

每个模板都是 24 个点、1 小时间隔（08:00 到次日 07:00），正好达到 `water_risk.MIN_RELIABLE_POINTS`，趋势外推不会被判低置信度。

## 表头风格

模板的表头**故意不统一**，因为真实导出文件本来就不统一。`upper_en` 风格还保留了数据集里原样的拼写错误 `DISOLVED OXYGEN`：

| 风格 | 表头 | 用到的模板 |
| --- | --- | --- |
| `canonical` | `timestamp,water_temp,ph,dissolved_oxygen,ammonia_n,nitrite_n` | `dawn_hypoxia`、`ph_high_alkaline`、`ammonia_climbing` |
| `units_en` | `timestamp,water_temp_c,ph,dissolved_oxygen_mg_l,ammonia_n_mg_l,nitrite_n_mg_l` | `normal_summer`、`nitrite_spike`、`low_do_warning` |
| `upper_en` | `TIMESTAMP,TEMPERATURE,PH,DISOLVED OXYGEN,AMMONIA_N,NITRITE_N` | `do_redline_crash` |
| `zh` | `采集时间,水温,酸碱度,溶解氧,氨氮,亚硝酸盐` | `normal_shrimp`、`winter_ice`、`spring_low_temp` |
| `zh_units` | `采集时间,水温(C),pH值,溶氧(mg/L),氨氮(mg/L),亚硝酸盐(mg/L)` | `ammonia_accumulation`、`temp_drop_after_rain` |

## 模板清单

「预期等级」不是描述，是**被单元测试钉住的断言**：`tests/unit/test_water_templates.py` 会真的调用规则引擎跑一遍，等级对不上就测试失败。

| 文件 | 名称 | 品种 | 预期等级 | 演示点 | 数据起点 |
| --- | --- | --- | --- | --- | --- |
| `normal_summer.csv` | 夏季正常塘（鲤科） | 鲤科 | **正常** | 基准模板：用来对比「我的塘口比正常差多少」。 | 2026-07-14 08:00 |
| `normal_shrimp.csv` | 虾蟹塘正常（对虾/河蟹） | 虾蟹 | **正常** | 品种维度：同一份水，换品种档案结论可能不同（这里溶氧 4.35 对虾蟹正常，对鲤科已警戒）。 | 2026-08-05 08:00 |
| `dawn_hypoxia.csv` | 黎明前缺氧（夜间耗氧） | 鲤科 | **高危** | 最典型的泛塘前兆：白天看着没事，天亮前最危险。 | 2026-07-20 08:00 |
| `do_redline_crash.csv` | 溶氧破红线（泛塘临界） | 鲤科 | **高危** | 红线规则：任何一项破红线即判高危，不看加权分。 | 2026-07-21 08:00 |
| `ammonia_accumulation.csv` | 高温高 pH 氨氮累积 | 鲤科 | **高危** | 分子氨协同：pH 与水溫同时偏高时氨氮毒性放大。 | 2026-07-25 08:00 |
| `nitrite_spike.csv` | 亚硝酸盐超标 | 鲤科 | **高危** | 亚硝酸盐单独超标：它能独立把等级推到高危。 | 2026-09-10 08:00 |
| `ph_high_alkaline.csv` | 午后 pH 偏高（碱度大） | 鲤科 | **高危** | pH 单指标进危险带（>9.0）；溶氧高不等于安全。 | 2026-08-18 08:00 |
| `winter_ice.csv` | 冬季冰封期 | 鲤科 | **高危** | 反直觉案例：溶氧漂亮但水温致命，等级仍为高危。 | 2027-01-12 08:00 |
| `low_do_warning.csv` | 溶氧偏低（警戒） | 鲤科 | **预警** | 警戒带：还没到要抢救的程度，但必须开始干预。 | 2026-09-02 08:00 |
| `temp_drop_after_rain.csv` | 暴雨降温后 | 鲤科 | **预警** | 应激场景：水温跌进警戒带 + pH 剧烈波动。 | 2026-09-22 08:00 |
| `ammonia_climbing.csv` | 氨氮缓慢爬升 | 鲤科 | **预警** | 趋势预警：数值还好看，但方向不对。 | 2026-08-28 08:00 |
| `spring_low_temp.csv` | 早春水温偏低 | 鲤科 | **预警** | 季节性警戒：水瘦温低，投料与用药都要改口径。 | 2027-03-18 08:00 |

## 场景说明

- **夏季正常塘（鲤科）**（`normal_summer`）：7 月中旬，晴，水温 26–30℃，溶氧充沛，氨氮与亚硝酸盐都在适宜区间。
- **虾蟹塘正常（对虾/河蟹）**（`normal_shrimp`）：8 月上旬，水温 26.5–29.5℃，溶氧 4.3–5.5 mg/L。这是个「对虾蟹正常、对鲤科偏紧」的溶氧水平——换品种档案结论就会变。
- **黎明前缺氧（夜间耗氧）**（`dawn_hypoxia`）：7 月下旬晴热无风，溶氧从午后 6.5 一路耗到黎明前 2.8 mg/L。
- **溶氧破红线（泛塘临界）**（`do_redline_crash`）：溶氧在 18 小时内从 5.2 掉到 1.2 mg/L，已跌破 1.5 的泛塘临界值。
- **高温高 pH 氨氮累积**（`ammonia_accumulation`）：7 月末高温，pH 8.5–8.9，氨氮爬到 0.72 mg/L（危险带，未破 1.0 红线）。
- **亚硝酸盐超标**（`nitrite_spike`）：9 月中旬，亚硝酸盐从 0.12 升到 0.44 mg/L（危险带，未破 0.5 红线）。
- **午后 pH 偏高（碱度大）**（`ph_high_alkaline`）：8 月中旬蓝藻旺发，pH 全天 9.1–9.4，溶氧因光合作用过饱和。
- **冬季冰封期**（`winter_ice`）：1 月中旬封冰，水温 4℃ 上下；低温溶氧充足，但水温远低于危险线。
- **溶氧偏低（警戒）**（`low_do_warning`）：9 月初，溶氧全天在 3.5–5.1 mg/L，够不上危险但已离开适宜区。
- **暴雨降温后**（`temp_drop_after_rain`）：9 月下旬一场暴雨，24 小时内水温 29.4→18.4℃、pH 8.25→7.25。
- **氨氮缓慢爬升**（`ammonia_climbing`）：8 月末投料量大，氨氮从 0.12 一路爬到 0.38 mg/L，尚未破危险线。
- **早春水温偏低**（`spring_low_temp`）：3 月中旬，水温 15.7–19.3℃，落在警戒带；其它指标干净。

## 免责

模板数值是**构造出来的演示数据**，不是任何真实塘口的实测记录，也不是标准原文里的数值。它只保证一件事：在当前这套阈值档案下，规则引擎会给出表中那个等级。
真实生产决策请以现场实测与有资质的水产技术人员意见为准。
