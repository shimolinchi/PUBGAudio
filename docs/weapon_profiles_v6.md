# v6 枪型与射击概率表

由 `generation_v6.json` 逐项导出；2026-09-08。资产名是客户端声音组标识，未核实的别名不强行翻译。角色列表示库中可用源，不等于最终场景每种角色数量相同。射速/弹量是本版合成参数，非当前游戏补丁精确统计。

模式按整条轨迹抽取；DMR与栓狙逐发。模式20/30/50为单点/短扫/持续扫，40/60为单点/固定连发。暂停范围是每次松开扳机后附加间隔；逐发最小间隔另列。详见[完整规范](spatial_v6.md)。

| 声音组 | 类型 | 模式与概率 | 弹量上限 | 逐发最小间隔(s) | 附加暂停(s) | 可用角色 |
|---|---|---|---:|---:|---|---|
| AK47 | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部/自身 |
| AWM | 栓动/手动 | single 100% | 5 | 1.8 | 0.4–1.6 | 外部/自身 |
| Ace32 | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部/自身 |
| Berreta686 | 非自动霰弹 | single 100% | 2 | 0.8 | 0.7–2.2 | 外部/自身 |
| BerretaM9 | 逐发 | single 100% | 15 | 0.2 | 0.15–0.7 | 外部/自身 |
| Bizon | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 53 | 0.09 | 0.7–2.2 | 外部/自身 |
| DP12 | 非自动霰弹 | single 100% | 14 | 0.8 | 0.7–2.2 | 自身 |
| DP28 | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 47 | 0.09 | 0.7–2.2 | 自身 |
| DesertEagle | 逐发 | single 100% | 7 | 0.2 | 0.15–0.7 | 外部/自身 |
| Dragunov | 连狙 | single 100% | 10 | 0.25 | 0.15–0.7 | 外部/自身 |
| FNFAL | 连狙 | single 100% | 10 | 0.25 | 0.15–0.7 | 外部/自身 |
| FamasG2 | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部/自身 |
| G36C | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部/自身 |
| Glock18C | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 17 | 0.065 | 0.7–2.2 | 外部/自身 |
| Groza | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部/自身 |
| HK416 | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部 |
| JS9 | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部 |
| K2 | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部/自身 |
| Kar98k | 栓动/手动 | single 100% | 5 | 1.8 | 0.4–1.6 | 外部/自身 |
| L6 | 连狙 | single 100% | 10 | 1.0 | 0.15–0.7 | 自身 |
| M14EBR | 连狙 | single 100% | 10 | 0.25 | 0.15–0.7 | 外部/自身 |
| M16A4 | 固定连发 | single 40% / burst 60%（每组3发） | 30 | 0.1 | 0.7–2.2 | 外部/自身 |
| M1911 | 逐发 | single 100% | 7 | 0.2 | 0.15–0.7 | 外部/自身 |
| M24 | 栓动/手动 | single 100% | 5 | 1.8 | 0.4–1.6 | 外部/自身 |
| M249 | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 75 | 0.09 | 0.7–2.2 | 外部/自身 |
| M762 | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部/自身 |
| MK12 | 连狙 | single 100% | 20 | 0.25 | 0.15–0.7 | 外部/自身 |
| MP5K | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部/自身 |
| MicroUzi | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 25 | 0.065 | 0.7–2.2 | 外部/自身 |
| Mini14 | 连狙 | single 100% | 20 | 0.25 | 0.15–0.7 | 外部/自身 |
| Mk47 | 固定连发 | single 40% / burst 60%（每组2发） | 20 | 0.1 | 0.7–2.2 | 外部/自身 |
| MosinNagant | 栓动/手动 | single 100% | 5 | 1.8 | 0.4–1.6 | 自身 |
| NagantM1895 | 逐发 | single 100% | 7 | 0.2 | 0.15–0.7 | 外部/自身 |
| O12 | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 12 | 0.25 | 0.7–2.2 | 外部/自身 |
| QBU | 连狙 | single 100% | 10 | 0.25 | 0.15–0.7 | 外部/自身 |
| QBZ | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部/自身 |
| Rhino | 逐发 | single 100% | 6 | 0.2 | 0.15–0.7 | 外部/自身 |
| SCAR_L | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部/自身 |
| SKS | 连狙 | single 100% | 10 | 0.25 | 0.15–0.7 | 外部 |
| Saiga12 | 非自动霰弹 | single 100% | 5 | 0.8 | 0.7–2.2 | 外部/自身 |
| SawedOffShotgun | 非自动霰弹 | single 100% | 2 | 0.8 | 0.7–2.2 | 外部/自身 |
| Scorpion | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 20 | 0.09 | 0.7–2.2 | 外部/自身 |
| Steyr_AUG_A3 | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部/自身 |
| Thompson | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 30 | 0.09 | 0.7–2.2 | 外部/自身 |
| UMP | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 25 | 0.09 | 0.7–2.2 | 外部/自身 |
| VSS | 连狙 | single 100% | 10 | 0.25 | 0.15–0.7 | 外部/自身 |
| Vector | 自动 | single 20% / short_auto 30% / sustained_auto 50% | 19 | 0.065 | 0.7–2.2 | 外部/自身 |
| Winchester1894 | 栓动/手动 | single 100% | 8 | 1.8 | 0.4–1.6 | 外部/自身 |
| Winchester1897 | 非自动霰弹 | single 100% | 5 | 0.8 | 0.7–2.2 | 自身 |
