# 衰减核查与玩家转头 · v5.1

更新：2026-09-07。反馈 F020/F021。当前是新的试听配置，**没有更新旧25小时数据、没有开训**。音量与原生游戏完整曲线尚未完成实录标定。

## 查到的证据及其边界

PUBG 官方 [2020年脚步开发日志](https://pubg.com/zh-cn/news/4742)说明：脚步距离感除了音量、高频衰减，还使用按地面/动作划分的两到三层距离原声并交叉淡化；公开文字没有给出全部米数与增益点。2019年方案曾回退，不能把旧测试转述当成当前PC参数。

本机已提取的 `2616915968.txt` SoundBanksInfo 中找到事件的 `MaxAttenuation`。仅提取与当前声音类别有关的事件记录，结果保存在 [元数据审计](../outputs/reports/attenuation_metadata_20260907.json)，包含1,427条事件、原始值、事件ID、路径，以及原文件SHA-256 `da8511767d8f5f9ebe5c30f521d68ce1459e5186a9acc5a3e4c6b1e0ce6de5d0`。不复制客户端WAV到Git。

[Unreal默认距离单位是厘米](https://dev.epicgames.com/documentation/unreal-engine/units-of-measurement-in-unreal-engine?lang=en-US)；[Wwise有效半径为 ScalingFactor × MaxAttenuation](https://www.audiokinetic.com/zh/public-library/2024.1.5_8803/?id=features_blueprintcomponent.html&source=UE4)。本批暂按厘米与缩放1换算。这是**事件半径上界的推定**：未验证PUBG运行时缩放、完整子声音曲线、RTPC、遮挡、混响与实际耳机听阈。缺少或为0的属性不能理解成“静音”或“无限传播”。

| 客户端远端脚步事件 | 原始值 | 按厘米换算 | 当前素材映射 |
|---|---:|---:|---|
| Stand_Slow | 3500 | 35m | walk，命名推断 |
| Stand_Normal | 4500 | 45m | run，命名推断 |
| Stand_Fast | 6000 | 60m | sprint，命名推断 |
| Crouch_Slow | 2500 | 25m | crouch_walk，原声名含 Walk_Slow |
| Crouch_Normal / Fast | 4000 / 5000 | 40 / 50m | 当前没有单独输出这两种步态 |
| Prone_Normal / Slow | 2000 / 2000 | 20m | prone_crawl |

每行Shoes与Barefoot的事件半径相同，不代表全部地面的每个子声音音量相同。

| 其他声音 | 事件半径推定 | 本批处理 / 局限 |
|---|---:|---|
| 多个普通车辆Engine_Remote | 300m | 发动机/起步层上界；REV音色仍是近似 |
| Engine_Stop | 50m | 熄火事件独立衰减 |
| Surface_Roll / Brake / Spin | 80m | 滚动、制动分层；当前skid用Spin半径做代理值，尚未恢复原生skid曲线 |
| Common_Impact / BRDM_Impact | 30 / 75m | 按碰撞源名区分；不借发动机半径 |
| TireExplosion | 100m | 爆胎独立层 |
| Delayed_Explosion_Basic_Car | 150m | 当前BlowUp/Fire/Damaged源来自该复合家族；只知道整个事件上界，各子声音半径待查 |
| Vehicle_Explosion / Explosion_Burn | 300 / 25m | 另一个事件家族，不直接套到当前Delayed素材上 |
| Grenade_Explosion | 1200m | 普通陆地爆炸上界；非实际清晰可辨距离 |
| C4 Attach / Switch步骤 / Beep / Explosion | 23 / 30 / 45 / 1600m | 各步骤分别衰减，保留完整因果链 |
| FireBomb_Impact / Blaze / Burn | 35m | 燃烧瓶命中/燃烧；点燃准备缺失半径 |
| CarePackage_Aircraft | 3000m | C130；飞机专项100s完整通过 |
| SandStorm / BlizzardZone | 700 / 400m | 区域环境事件上界；当前沙尘暴为周围弥散背景，不当成原点点声源套距离衰减。暴风雪未唯一解码 |
| 普通枪声、闪光弹爆响、拉环、cook、点燃准备 | 未确认 | 缺失可用半径，继续衰减但不编造游戏截止距离 |

当前普通枪声的事件路径出现 Attenuation M/L 等分类，但没有导出其完整曲线；消音器和原生近/中/远层也没有在本批复现。[PUBG 16.2官方说明](https://www.pubg.com/en/news/1717)提供有枪型/距离提示的声音训练场，可作为后续实录校准入口；本轮没有录制真实游戏，也没有把旧论坛听感当标定值。

## 修正的曲线

旧合成器所有外部声源共用1/8/30/90/400单位对应振幅1.15/1/.5/.22/.045的曲线。`np.interp`在400之外保持.045不变；这会让极远声音保留固定底音，脚步在90单位仍有22%振幅也过大。

脚步改为相对同一原声在5m的振幅 `A(d)=min(1,5/d) × taper(d,R)`，R按当前步态选择。taper在0–0.7R为1，最后30%采用 `1−3q²+2q³` 平滑降至0；d≥R为0。这是可调整的试听估算，不是从最远一点唯一反演出的原生曲线。原有高频滤波与HRIR保留，原生距离分层尚待接入。

| 距离 | 行走R=35振幅 | 跑步R=45振幅 | 冲刺R=60振幅 |
|---|---:|---:|---:|
| 5m | 100% | 100% | 100% |
| 10m | 50% | 50% | 50% |
| 20m | 25% | 25% | 25% |
| 30m | 约7.74% | 约16.67% | 约16.67% |
| 40m | 0 | 约3.88% | 12.5% |
| 50m | 0 | 0 | 约5.83% |
| 60m及以上 | 0 | 0 | 0 |

100%以同种原声5m为参考，**是线性振幅比例，不是人耳响度百分比**。50%振幅为约−6.02dB，25%为约−12.04dB，均非绝对声压级。素材入库的RMS调整仍在，跨步态、地面的近处响度并未复现原生总线混音。

其他外部事件保留旧400以内曲线，在有依据的半径外端乘平滑taper；超过400继续末段对数斜率，取消底音平台。未知半径继续下降，禁止伪装为已标定的硬截止。自身声保持原生立体声直通；周围弥散天气背景无唯一点方向。混音后仍以分轨功率重算遮蔽，无逐片音量归一、无把远处静音重新放大。

## 玩家视角与训练标签

世界轨迹先生成，玩家水平朝向另生成。设世界位置 `(x,y)`、顺时针朝向yaw，耳朵坐标为 `(x*cos(yaw)−y*sin(yaw), x*sin(yaw)+y*cos(yaw))`，相对方位等于世界方位减yaw。转头不改变距离，也不会让飞机在世界中拐弯。

随机视角模式整段固定25%、自然活动75%；初始朝向0–360°均匀、左右等权。在自然模式内，先停留（80%抽2–8s，20%抽10–25s），再抽动作：小晃动60%（0.5–6°、0.5–2s），普通观察25%（8–35°、1–3s），大转向15%（60–160°、0.8–2s）。角速度上限140°/s，必要时延长转动时间；smoothstep连接、角速度在起止处为0。小晃动65%概率回摆原幅度的40–100%，避免机械等幅周期。没有逐帧掷硬币切换视角。详见实际 [v5.1概率表](generation_probability_v5_1.md)。当前仅水平角度，玩家世界位置固定、俯仰0；不是完整六自由度仿真。

音频HRIR使用当前耳朵坐标。10ms标签中的 `track_xy`、`track_azimuth_degrees` 是同一听者坐标，训练投影沿用这些字段；另存 `track_world_xy`、`track_world_azimuth_degrees`、`listener_yaw_degrees` 供审核，不能混用。

脚步额外存 `track_distance_m_estimate`、`track_footstep_cutoff_m_estimate`、`track_footstep_distance_gain_estimate`、`track_footstep_emission_scheduled`、`track_outside_footstep_radius`。不适用字段为NaN，表示没有该估计。范围外的物理人物、脚步计划与位置仍保留，实际静音不造训练正例；有声但被遮蔽的目标仍mask=0。其他事件半径与证据记录在配方各事件/车辆层内。

足迹远处模式仍80%权重，但远处偏移改为本轨最大步态R的55–80%，中距离30–50%，近处8–22%；轨迹长短和速度不被强行压缩，可自然进出范围。该比例是抽样权重，不是强制每段80%有声远帧。其他类别的距离采样保持原设计。

## 新试听与复现

目录 `outputs/attenuation-view-preview-20260907-v5_1-final`，22段、共22分50秒，包含6随机、10原场景专项及6新增对照。D01/D02/D03为行走/跑步/冲刺距离阶梯，每4秒一个5–80m位置，固定同一组混凝土原声；这是独立距离测量对照，不是人物瞬移轨迹，不进入训练或随机统计。S09与D04为同一运输机世界轨迹的固定/转头对照，D05为固定枪口加转头。D06为120秒连续场景，含自然概率视角、曲线脚步、燃烧瓶过程、滑车和自身声。其他样本含完整事件链与遮蔽。

```powershell
python tools/synthesis/audit_attenuation.py --metadata C:/Projects/PUBGAssistant-cpp/.tmp/audio-pilot-20260906/metadata/2616915968.txt
python tools/synthesis/write_probability_reference.py --config configs/generation_v5_1.json --output docs/generation_probability_v5_1.md
python tools/synthesis/generate_attenuation_preview.py --output outputs/新的试听目录
python tools/synthesis/build_causal_preview.py outputs/新的试听目录
python tools/synthesis/verify_causal_preview.py outputs/新的试听目录 --replay D03 D04 S06
python tools/synthesis/serve_preview.py outputs/新的试听目录 --port 8767
```

核验与人工反馈状态以批次 `verification.json` 为准。旧8766页面、旧配方和用户反馈保留；本批反馈使用独立dataset_id及feedback目录。后续正式数据仍需源家族隔离与真实游戏录音验证，不能直接将试听对照混进测试集。

## 正式长音频与历史窗口（方案，尚未开训）

用户希望用前一段时间的信息判断当前时刻，尤其是燃烧等持续事件。正式生成建议120/180/300秒连续场景，概率25%/50%/25%；计量预算以总小时数为准，不继续沿用3000条×新长度而无意扩大规模。长片段统一生成世界状态、视角、音频、10ms标签，动作跨裁剪边界时不重新起步或重新点燃。

训练从长录音按时间索引裁剪，先保留当前5秒基线，再对比10/20/30秒历史。窗口只取 `[t−H,t]`，监督当前末尾时间步；不要用 `[t−H/2,t+H/2]` 的未来声音预测当前。建议正式长录音带30秒前置历史，前置部分不作评价目标。数据中保留燃烧完整过程，输入窗可覆盖点燃、命中和持续燃烧；固定10秒燃烧时长仍待校准，不因文件变长就当作游戏事实。

现有网络为5秒输入的离线BiGRU基线。**文件变长本身不会扩展模型的上下文**；后续须同步修改裁剪器/归一化统计/输入长度与训练策略，并只对最后时间步使用过去窗口，或另实现有状态的因果网络。用户当前要求先完善数据，因此这里只保存方案与120秒试听，不改网络、训练加载器或正式25小时集。

先按素材家族与父长场景划分train/val/test，再裁剪；同一长录音的相邻或重叠窗口不能跨分区。随机裁剪复用一份音频文件，不把每个滑动窗复制落盘。正式发布前还需核验长场景的调度、活动分布和起止边界；现有60秒试听的随机统计不能直接当作2–5分钟场景统计。
