"""Publish the probability table from the actual v5 generator input."""
import hashlib
import json
import argparse
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=root/'configs/generation_v5.json')
    parser.add_argument('--output',type=Path,default=root/'docs/generation_probability_v5.md')
    args=parser.parse_args();path=args.config;c=json.loads(path.read_text(encoding='utf-8'))
    lines=['# v5 实际生成概率表','',
        '本表由 `tools/synthesis/write_probability_reference.py` 从实际配置生成。修改 `configs/generation_v5.json` 后重新运行；不要只改此表。',
        '',f'配置 SHA-256：`{hashlib.sha256(path.read_bytes()).hexdigest()}`','',
        '所有值是可调整的初始设计，不是实战统计。每个满足前提的 0.5 秒调度步独立抽是否新建事件；已有事件由状态机推进。','',
        '## 密度与事件发生','',
        '| 密度 | 选择权重 | 角色上限 | 脚步 | 车辆 | 枪声 | 投掷物 | C4 | 运输机 | 天气 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for density,row in c['spawn_probability_per_eligible_step'].items():
        vals=[density,str(c['density_weights'][density]),str(c['max_concurrent_actors'][density])]+[f'{p*100:g}%' for p in row.values()]
        lines.append('| '+' | '.join(vals)+' |')
    lines += ['', '密度权重归一化后低/中/高各 1/3。角色上限计入人物、车辆、枪声、直接操作；一个角色的关联效果不是一个新角色。运输机要求完整 60 秒通场，本试听配置只有 t=0 合法；其他类别也必须预留完整事件时间。天气最多一次、固定 12 秒，当前仅沙尘暴。','',
        '## 条件选择','', '| 选择 | 配置值 | 适用条件 |','|---|---|---|',
        f"| 直接自身声 | {c['self_probability_when_eligible']*100:g}% | 没有其他正在进行的自身角色；天气、飞机不适用 |",
        f"| 静态声源远处 | {c['stationary_distance']['far_probability']*100:g}% | 外部枪口/准备操作等，远处 100–180、其他 12–45 合成单位 |",
        '| 移动声源距离模式 | '+', '.join(f'{k}={v:g}' for k,v in c['distance_mode_weights'].items())+' | 权重，不等于整段可听帧的远近比例 |',
        '| 曲线 | '+', '.join(f'{k}={v:g}' for k,v in c['curve_weights'].items())+' | 合法曲线等权；按速度积分后沿弧长采样 |',
        '| 地面/武器/车族/同类原声 | 可用候选等权 | 先过滤素材和动作前提；同一脚步角色保持地面不变 |',
        '| 步态选择权重 | '+', '.join(f'{k}={v:g}' for k,v in c['footsteps']['initial_weights'].items())+' | 先过滤同地面可用步态，再归一化；不等于声音个数/帧数占比 |',
        '| 长度桶 | 每类三个桶等权，桶内均匀 | 脚步 12–18/18–25/25–34s；车辆 18–24/24–32/32–40s；枪声 4–7/7–10/10–13s |',
        f"| 步态段后明显停步 | {c['footsteps']['pause_probability_after_bout']*100:g}% | 停 1–3s；其他情况保留 0.3s 转换间隔 |",
        '| 投掷物种类 | frag/flash/molotov 等权 | 抽中新投掷事件后；准备与后果保持关联 |',
        f"| 提前计时（烹雷） | {c['throwable']['cook_probability']*100:g}% | 仅手雷/闪光弹 |",
        f"| 碰撞 | {c['vehicle']['collision_probability_per_vehicle_with_motion']*100:g}% | 每个存在合法运动时段的车辆片段抽一次 |",
        f"| 遭射击 | {c['vehicle']['attack_probability_per_vehicle_with_motion']*100:g}% | 每个存在合法运动时段的车辆片段抽一次；附带枪声前因 |",
        f"| 射击致命 | {c['vehicle']['fatal_damage_probability_given_attack']*100:g}% | 已抽中受攻击的条件分支 |",
        f"| 非致命攻击后爆胎 | {c['vehicle']['tire_failure_probability_given_attack']*100:g}% | 先排除致命分支；总条件比例为致命25%、爆胎37.5%、普通车体37.5% |",
        f"| 自身车存在时保留外部脚步 | {c['masking']['cabin_foot_keep_probability']*100:g}% | 候选接纳抽样；混音后仍重算可观测性 |",
        f"| 自身车存在时保留很远的外部车 | {c['masking']['cabin_far_vehicle_keep_probability']*100:g}% | 路径最近距离≥90单位的候选 |",'',
        '## 状态维持与切换','',
        '| 状态 | 最短持续 | 之后每 0.5s 切换概率 | 最长持续 |','|---|---:|---:|---:|',
        f"| 脚步当前步态 | {c['footsteps']['minimum_dwell_seconds']}s | {c['footsteps']['switch_probability_after_minimum']*100:g}% | {c['footsteps']['maximum_dwell_seconds']}s |"]
    for state,minimum in c['vehicle']['minimum_dwell_seconds'].items():
        maximum=c['vehicle'].get('maximum_dwell_seconds_by_state',{}).get(state,c['vehicle']['maximum_dwell_seconds'])
        lines.append(f"| 车辆 {state} | {minimum:g}s | {c['vehicle']['switch_probability_after_minimum'][state]*100:g}% | {maximum:g}s |")
    lines += ['', '调度实现先经过最短持续时间，再逐个调度步试切换；片末收尾可以截短。车辆第一次加速还需向 12–24 单位/s 的目标速度靠拢，并受加速度、最长持续时间和剩余时长限制。',
        '', '| 车辆后继候选 | 权重 |','|---|---:|']
    lines += [f'| {k} | {v:g} |' for k,v in c['vehicle']['next_state_weights'].items()]
    stopped=c['vehicle']['stopped_seconds']
    lines += ['', f'删除当前状态及不合法后继后重新归一化。熄火滑行不能直接接巡航/动力减速，动力加速前先启动。急刹为 0.5–1.1s 过程，停留 {stopped[0]:g}–{stopped[1]:g}s；临近片末另有平滑刹停收尾，不作为随机巡航。',
        '', '## 非概率时间与声学约束','',
        '| 项目 | 当前值 |','|---|---|',
        '| C4 | 放置约 4s + 激活后 16s；提示间隔节点 '+str(c['c4']['beep_interval_knots'])+'，间隔为未实测的加速近似 |',
        '| 手雷/闪光 | 手雷5s；闪光烹雷2.5s、不烹雷5s，与首次撞击后0.7s取早 |',
        '| 燃烧瓶 | 固定10s，当前PC权威时长未确认，等待实测 |',
        '| 车辆普通毁坏 | 起火后5s爆炸；当前只做射击分支 |',
        '| 运输机 | 完整60s，速度65–85单位/s，最近侧向偏移120–300单位 |',
        '| 遮蔽启发式 | 一般SIR −30dB；自身车内脚步相对车声 −12dB、远车 −15dB；保留详细活动，隐藏正例不监督 |',
        '| 响度 | 全批固定总增益0.6；每声源增益均匀抖动−1.5至+1.5dB；不逐条归一到同一峰值 |',
        '', '枪族发射间隔、单弹匣上限、连发长度及间歇详见 JSON 的 `gunfire` 表。这些是有弹量约束的试听轮廓，不是当前官方枪械数据。轨迹距离、速度及 HRTF 衰减仍为合成近似，不能据此宣称米级标定。',
        '', '其他规则、证据链接和未接入类别见 [连续场景规范](natural_scene_rules.md)。']
    if 'listener_view' in c:
        version=c.get('display_version') or ('v5.2' if c['throwable'].get('self_throws_forward_at_release') else 'v5.1')
        lines[0]=f'# {version} 实际生成概率与声学参数表'
        lines[2]=f'由实际配置 `configs/{path.name}` 生成；完整声学证据与边界见 [衰减核查](attenuation_and_listener_v5_1.md)。上面的旧距离模式对脚步有如下专用覆盖。'
        lines+=['','## 脚步距离专用覆盖','','远处模式仍为80%权重，最近横向偏移按本轨出现步态中最大的估算半径 R 抽取。路径长度仍由速度积分决定，不缩短速度来强行留在可听范围；走出范围时保留几何，不强制正例。','',
            '| 模式 | 最近横向偏移 / R |','|---|---|']
        lines += [f'| {k} | {v[0]:g}–{v[1]:g} |' for k,v in c['footstep_acoustics']['closest_radius_fraction'].items()]
        lines+=['','## 玩家水平视角','','| 模式 | 抽样概率 |','|---|---:|']
        lines += [f'| {k} | {v*100:g}% |' for k,v in c['listener_view']['mode_weights'].items()]
        view=c['listener_view'];hold=view['hold_seconds'];long_hold=view['long_hold_seconds'];return_fraction=view['return_fraction']
        lines+=['',f"自然模式每段先停留：{(1-view['long_hold_probability'])*100:g}%抽{hold[0]:g}–{hold[1]:g}s、{view['long_hold_probability']*100:g}%抽{long_hold[0]:g}–{long_hold[1]:g}s；再选以下动作。初始朝向均匀0–360°、左右等权，smoothstep连接，角速度上限{view['maximum_angular_speed_degrees_per_second']:g}°/s。小晃动{view['small_sway_return_probability']*100:g}%概率向原朝向回摆{return_fraction[0]*100:g}–{return_fraction[1]*100:g}%，幅度和时间不同，避免机械周期。世界原点固定、俯仰0°。",'',
            '| 段内动作 | 概率 | 幅度 | 持续时间 |','|---|---:|---|---|']
        for k,v in c['listener_view']['episode_weights'].items():
            spec=c['listener_view']['episodes'][k]
            lines.append(f"| {k} | {v*100:g}% | {spec['amplitude_degrees']}° | {spec['seconds']}s；必要时延长以满足角速度上限 |")
        lines+=['',
            '## 衰减参数（待实录标定）','','脚步参考5m、振幅指数1；最后30%半径平滑衰减到0。其他已知事件在旧曲线外乘相同平滑截尾；缺半径的角色继续400以后的对数斜率，不填造截止距离。','',
            '| 脚步素材步态 | 估算半径 m | 客户端事件 |','|---|---:|---|']
        lines += [f"| {k} | {v['cutoff_m_estimate']:g} | `{v['event']}` |" for k,v in c['footstep_acoustics']['profiles'].items()]
        lines += ['', '| 事件/分层 | 估算半径 m | 证据状态 |','|---|---:|---|']
        lines += [f"| {k} | {v['cutoff_m_estimate']:g} | {v['evidence']} |" for k,v in c['event_acoustics']['profiles'].items()]
        lines += ['', '历史v5.1专项阶梯、成对转头、近车检查不计入随机比例；其中运输机专项100秒，首末均在估算3km半径以外。旧数据文件不自动更新。']
        if c['throwable'].get('self_throws_forward_at_release'):
            lines += ['', '## 自身投掷与车辆调整', '',
                '自身手雷、闪光弹、燃烧瓶在出手时间绑定玩家水平朝向，沿该方向前抛；落点随后固定在世界坐标。后续转头只改变耳相对方位，不改变投掷路径。准备声标为自身，远处爆响和燃烧保持外部效果标签。', '',
                f"车辆启动至少 {c['vehicle']['minimum_startup_seconds']:g}s，若原生启动素材更长则完整保留。刹停后的额外停留缩短；熄火滑行仍保留但较少抽中。距最高速度不足 {c['vehicle']['acceleration_headroom_units_per_second']:g} 单位/s 时排除再次抽加速，避免以加速标签覆盖最高速巡航。未提高滑车素材音量。", '',
                'v5.2历史试听包含强制前抛、车辆、长片段和飞机专项及独立随机片段；专项不计入随机概率。详见 [v5.2 行为调整](forward_throw_and_motion_v5_2.md)。']
    if c.get('body_actions'):
        ac=c['body_actions'];w=c['footsteps']['initial_weights'];total=sum(w.values())
        lines+=['','## 普通行走、跑步与身体动作','','本版保留脚步角色总生成率。完整五步态均可用时：','','| 步态 | 权重 | 条件选择概率 |','|---|---:|---:|']
        lines += [f'| {k} | {v:g} | {100*v/total:g}% |' for k,v in w.items()]
        lines+=['','同地面缺少步态时只在实际可用候选中归一化；普通行走与跑步的权重从各1改为0.75/1.25。蹲行与匍匐仍是移动，不能作为原地蹲起/趴起标签。','','| 停步后的附加动作 | 合法候选齐全时概率 |','|---|---:|']
        total=sum(ac['after_bout_weights'].values())
        lines += [f'| {k} | {100*v/total:g}% |' for k,v in ac['after_bout_weights'].items()]
        lines+=['',f"仅站立步态停止后、距上次动作结束至少{ac['minimum_action_gap_seconds']:g}s、剩余至少{ac['required_remaining_seconds']:g}s时附加动作可用；候选不足时重新归一化。continue表示不追加动作，随后按步态表选下一段；步态导致的必要站/蹲/趴转换另行执行，不计入25%/8%的附加循环。",
            '',f"原地蹲/趴后保持{ac['posture_hold_seconds'][0]:g}–{ac['posture_hold_seconds'][1]:g}s再起身；转换至少覆盖完整原声。翻越约{ac['vault_seconds']:g}s，规划障碍位置和接触阶段，期间不生成正常走跑脚步。附加动作不占新的角色名额，自身/外部身份继承同一角色。",
            '', '蹲起/趴起使用已追踪到原生 Transitions_Ruslte 事件的共享转换衣物声；准确姿态来自合成计划，不是独有音色证据。翻越使用原生抓扶/攀越/接触/衣物素材；具体动画时序仍近似。细动作全部保留，默认三类训练投影将这些动作作为干扰。',
            '', '本次实际试听为6段40秒动作专项和4段60秒独立随机，共10段8分钟；专项采用显式覆盖，不计上述自然概率。历史v5.1/v5.2的飞机、前抛与长片段示例继续保留，并非本批新增。见[动作调整](body_actions_v5_3.md)。']
    args.output.write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
