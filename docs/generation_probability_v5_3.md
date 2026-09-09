# v5.3 实际生成概率与声学参数表

由实际配置 `configs/generation_v5_3.json` 生成；完整声学证据与边界见 [衰减核查](attenuation_and_listener_v5_1.md)。上面的旧距离模式对脚步有如下专用覆盖。

配置 SHA-256：`cb5eeb63ed3823dded1ef70585abcd1c8247c9a54bd5a4e0b73507e84d023eef`

所有值是可调整的初始设计，不是实战统计。每个满足前提的 0.5 秒调度步独立抽是否新建事件；已有事件由状态机推进。

## 密度与事件发生

| 密度 | 选择权重 | 角色上限 | 脚步 | 车辆 | 枪声 | 投掷物 | C4 | 运输机 | 天气 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low | 1 | 2 | 0.8% | 0.8% | 0.8% | 0.3% | 0.06% | 0.08% | 0.05% |
| medium | 1 | 4 | 1.8% | 1.8% | 1.8% | 0.6% | 0.12% | 0.12% | 0.1% |
| high | 1 | 6 | 3.5% | 3.5% | 3.5% | 1.2% | 0.2% | 0.12% | 0.1% |

密度权重归一化后低/中/高各 1/3。角色上限计入人物、车辆、枪声、直接操作；一个角色的关联效果不是一个新角色。运输机要求完整 60 秒通场，本试听配置只有 t=0 合法；其他类别也必须预留完整事件时间。天气最多一次、固定 12 秒，当前仅沙尘暴。

## 条件选择

| 选择 | 配置值 | 适用条件 |
|---|---|---|
| 直接自身声 | 50% | 没有其他正在进行的自身角色；天气、飞机不适用 |
| 静态声源远处 | 80% | 外部枪口/准备操作等，远处 100–180、其他 12–45 合成单位 |
| 移动声源距离模式 | far=0.8, near_pass=0.05, mid_pass=0.05, approach=0.05, recede=0.05 | 权重，不等于整段可听帧的远近比例 |
| 曲线 | straight=1, arc=1, sine=1, bezier=1 | 合法曲线等权；按速度积分后沿弧长采样 |
| 地面/武器/车族/同类原声 | 可用候选等权 | 先过滤素材和动作前提；同一脚步角色保持地面不变 |
| 步态选择权重 | walk=0.75, run=1.25, sprint=1, crouch_walk=1, prone_crawl=1 | 先过滤同地面可用步态，再归一化；不等于声音个数/帧数占比 |
| 长度桶 | 每类三个桶等权，桶内均匀 | 脚步 12–18/18–25/25–34s；车辆 18–24/24–32/32–40s；枪声 4–7/7–10/10–13s |
| 步态段后明显停步 | 50% | 停 1–3s；其他情况保留 0.3s 转换间隔 |
| 投掷物种类 | frag/flash/molotov 等权 | 抽中新投掷事件后；准备与后果保持关联 |
| 提前计时（烹雷） | 50% | 仅手雷/闪光弹 |
| 碰撞 | 10% | 每个存在合法运动时段的车辆片段抽一次 |
| 遭射击 | 15% | 每个存在合法运动时段的车辆片段抽一次；附带枪声前因 |
| 射击致命 | 25% | 已抽中受攻击的条件分支 |
| 非致命攻击后爆胎 | 50% | 先排除致命分支；总条件比例为致命25%、爆胎37.5%、普通车体37.5% |
| 自身车存在时保留外部脚步 | 10% | 候选接纳抽样；混音后仍重算可观测性 |
| 自身车存在时保留很远的外部车 | 5% | 路径最近距离≥90单位的候选 |

## 状态维持与切换

| 状态 | 最短持续 | 之后每 0.5s 切换概率 | 最长持续 |
|---|---:|---:|---:|
| 脚步当前步态 | 3s | 12.5% | 12s |
| 车辆 accelerate | 3s | 10% | 12s |
| 车辆 cruise | 1.5s | 25% | 6s |
| 车辆 decelerate | 1s | 20% | 6s |
| 车辆 coast_off | 1s | 35% | 3s |

调度实现先经过最短持续时间，再逐个调度步试切换；片末收尾可以截短。车辆第一次加速还需向 12–24 单位/s 的目标速度靠拢，并受加速度、最长持续时间和剩余时长限制。

| 车辆后继候选 | 权重 |
|---|---:|
| accelerate | 2.5 |
| cruise | 0.6 |
| decelerate | 1 |
| brake | 0.5 |
| coast_off | 0.2 |

删除当前状态及不合法后继后重新归一化。熄火滑行不能直接接巡航/动力减速，动力加速前先启动。急刹为 0.5–1.1s 过程，停留 0.3–1.2s；临近片末另有平滑刹停收尾，不作为随机巡航。

## 非概率时间与声学约束

| 项目 | 当前值 |
|---|---|
| C4 | 放置约 4s + 激活后 16s；提示间隔节点 [[0, 1.0], [8, 0.65], [12, 0.35], [15, 0.16], [16, 0.16]]，间隔为未实测的加速近似 |
| 手雷/闪光 | 手雷5s；闪光烹雷2.5s、不烹雷5s，与首次撞击后0.7s取早 |
| 燃烧瓶 | 固定10s，当前PC权威时长未确认，等待实测 |
| 车辆普通毁坏 | 起火后5s爆炸；当前只做射击分支 |
| 运输机 | 完整60s，速度65–85单位/s，最近侧向偏移120–300单位 |
| 遮蔽启发式 | 一般SIR −30dB；自身车内脚步相对车声 −12dB、远车 −15dB；保留详细活动，隐藏正例不监督 |
| 响度 | 全批固定总增益0.6；每声源增益均匀抖动−1.5至+1.5dB；不逐条归一到同一峰值 |

枪族发射间隔、单弹匣上限、连发长度及间歇详见 JSON 的 `gunfire` 表。这些是有弹量约束的试听轮廓，不是当前官方枪械数据。轨迹距离、速度及 HRTF 衰减仍为合成近似，不能据此宣称米级标定。

其他规则、证据链接和未接入类别见 [连续场景规范](natural_scene_rules.md)。

## 脚步距离专用覆盖

远处模式仍为80%权重，最近横向偏移按本轨出现步态中最大的估算半径 R 抽取。路径长度仍由速度积分决定，不缩短速度来强行留在可听范围；走出范围时保留几何，不强制正例。

| 模式 | 最近横向偏移 / R |
|---|---|
| far | 0.55–0.8 |
| near_pass | 0.08–0.22 |
| mid_pass | 0.3–0.5 |

## 玩家水平视角

| 模式 | 抽样概率 |
|---|---:|
| fixed | 10% |
| natural | 90% |

自然模式每段先停留：88%抽0.35–1.6s、12%抽4–10s；再选以下动作。初始朝向均匀0–360°、左右等权，smoothstep连接，角速度上限140°/s。小晃动50%概率向原朝向回摆40–100%，幅度和时间不同，避免机械周期。世界原点固定、俯仰0°。

| 段内动作 | 概率 | 幅度 | 持续时间 |
|---|---:|---|---|
| small_sway | 60% | [0.5, 6]° | [0.35, 1.1]s；必要时延长以满足角速度上限 |
| look_around | 18% | [8, 35]° | [0.7, 2]s；必要时延长以满足角速度上限 |
| large_turn | 22% | [60, 160]° | [0.8, 1.8]s；必要时延长以满足角速度上限 |

## 衰减参数（待实录标定）

脚步参考5m、振幅指数1；最后30%半径平滑衰减到0。其他已知事件在旧曲线外乘相同平滑截尾；缺半径的角色继续400以后的对数斜率，不填造截止距离。

| 脚步素材步态 | 估算半径 m | 客户端事件 |
|---|---:|---|
| walk | 35 | `Footstep_Stand_Slow_Remote_Shoes` |
| run | 45 | `Footstep_Stand_Normal_Remote_Shoes` |
| sprint | 60 | `Footstep_Stand_Fast_Remote_Shoes` |
| crouch_walk | 25 | `Footstep_Crouch_Slow_Remote_Shoes` |
| prone_crawl | 20 | `Footstep_Prone_Normal_Remote_Shoes` |

| 事件/分层 | 估算半径 m | 证据状态 |
|---|---:|---|
| engine | 300 | event_radius_upper_bound_source_family_mapping |
| startup | 300 | event_radius_upper_bound_source_family_mapping |
| shutdown | 50 | event_radius_upper_bound_source_family_mapping |
| roll | 80 | event_radius_upper_bound_source_family_mapping |
| brake | 80 | event_radius_upper_bound_source_family_mapping |
| skid | 80 | proxy_from_surface_spin_event_not_native_skid_radius |
| collision | 30 | event_radius_upper_bound_source_family_mapping |
| collision_brdm | 75 | event_radius_upper_bound_source_family_mapping |
| tire_burst | 100 | event_radius_upper_bound_source_family_mapping |
| vehicle_explosion | 150 | event_radius_upper_bound_source_family_mapping |
| vehicle_fire | 150 | composite_event_upper_bound_child_radius_unknown |
| vehicle_damage | 150 | composite_event_upper_bound_child_radius_unknown |
| frag_explosion | 1200 | event_radius_upper_bound_source_family_mapping |
| molotov_impact | 35 | event_radius_upper_bound_source_family_mapping |
| molotov_fire | 35 | event_radius_upper_bound_source_family_mapping |
| c4_beep | 45 | event_radius_upper_bound_source_family_mapping |
| c4_attach | 23 | event_radius_upper_bound_source_family_mapping |
| c4_switch_1 | 30 | event_radius_upper_bound_source_family_mapping |
| c4_switch_2 | 30 | event_radius_upper_bound_source_family_mapping |
| c4_switch_3 | 30 | event_radius_upper_bound_source_family_mapping |
| c4_explosion | 1600 | event_radius_upper_bound_source_family_mapping |
| aircraft_pass | 3000 | event_radius_upper_bound_source_family_mapping |
| transition_rustle | 30 | CharacterBank HIRC attenuation 515586057 max distance 3000; centimetres/scale1 assumed; intermediate response approximate |
| vault_grab | 30 | provisional shared transition-rustle radius proxy; named vault event-specific radius not recovered |
| vault_climb | 30 | provisional shared transition-rustle radius proxy; named vault event-specific radius not recovered |
| vault_contact | 30 | provisional shared transition-rustle radius proxy; named vault event-specific radius not recovered |
| vault_rustle | 30 | provisional shared transition-rustle radius proxy; named vault event-specific radius not recovered |

历史v5.1专项阶梯、成对转头、近车检查不计入随机比例；其中运输机专项100秒，首末均在估算3km半径以外。旧数据文件不自动更新。

## 自身投掷与车辆调整

自身手雷、闪光弹、燃烧瓶在出手时间绑定玩家水平朝向，沿该方向前抛；落点随后固定在世界坐标。后续转头只改变耳相对方位，不改变投掷路径。准备声标为自身，远处爆响和燃烧保持外部效果标签。

车辆启动至少 1s，若原生启动素材更长则完整保留。刹停后的额外停留缩短；熄火滑行仍保留但较少抽中。距最高速度不足 2 单位/s 时排除再次抽加速，避免以加速标签覆盖最高速巡航。未提高滑车素材音量。

v5.2历史试听包含强制前抛、车辆、长片段和飞机专项及独立随机片段；专项不计入随机概率。详见 [v5.2 行为调整](forward_throw_and_motion_v5_2.md)。

## 普通行走、跑步与身体动作

本版保留脚步角色总生成率。完整五步态均可用时：

| 步态 | 权重 | 条件选择概率 |
|---|---:|---:|
| walk | 0.75 | 15% |
| run | 1.25 | 25% |
| sprint | 1 | 20% |
| crouch_walk | 1 | 20% |
| prone_crawl | 1 | 20% |

同地面缺少步态时只在实际可用候选中归一化；普通行走与跑步的权重从各1改为0.75/1.25。蹲行与匍匐仍是移动，不能作为原地蹲起/趴起标签。

| 停步后的附加动作 | 合法候选齐全时概率 |
|---|---:|
| continue | 62% |
| crouch_cycle | 25% |
| prone_cycle | 8% |
| vault | 5% |

仅站立步态停止后、距上次动作结束至少7s、剩余至少11s时附加动作可用；候选不足时重新归一化。continue表示不追加动作，随后按步态表选下一段；步态导致的必要站/蹲/趴转换另行执行，不计入25%/8%的附加循环。

原地蹲/趴后保持2–3.5s再起身；转换至少覆盖完整原声。翻越约3.6s，规划障碍位置和接触阶段，期间不生成正常走跑脚步。附加动作不占新的角色名额，自身/外部身份继承同一角色。

蹲起/趴起使用已追踪到原生 Transitions_Ruslte 事件的共享转换衣物声；准确姿态来自合成计划，不是独有音色证据。翻越使用原生抓扶/攀越/接触/衣物素材；具体动画时序仍近似。细动作全部保留，默认三类训练投影将这些动作作为干扰。

本次实际试听为6段40秒动作专项和4段60秒独立随机，共10段8分钟；专项采用显式覆盖，不计上述自然概率。历史v5.1/v5.2的飞机、前抛与长片段示例继续保留，并非本批新增。见[动作调整](body_actions_v5_3.md)。
