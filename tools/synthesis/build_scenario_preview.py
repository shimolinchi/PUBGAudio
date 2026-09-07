"""Build a local audio player with synchronized trajectory and action timelines."""
import argparse
import json
from pathlib import Path

import numpy as np

import self_audio_synthesis as motion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preview', type=Path, required=True)
    args = p.parse_args()
    root = args.preview.resolve()
    rows = [json.loads(line) for line in (root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    sources = {s['id']: s for s in json.loads((root / 'source_manifest.json').read_text(encoding='utf-8'))['sources']}
    bundle = []
    for row in rows:
        tracks, timeline = [], []
        with np.load(root / row['label_file'], allow_pickle=False) as z:
            frame_activity = z['track_activity'].tolist()
            frame_observable = z['track_observable'].tolist()
        for i, track in enumerate(row['tracks']):
            cls = track['class_index']
            path = motion.trajectory(track['trajectory'], 30)
            points = np.column_stack([path['xy'], path['azimuth'], path['distance'], path['speed'], path['motion_state']])[::5]
            surface_names = {'Concrete': '水泥地', 'Dirt': '泥地', 'Fabric': '织物地面', 'Rock': '岩石地面', 'Snow': '雪地'}
            name = (surface_names.get(track.get('surface'), track.get('surface', '')) + '脚步') if cls == 0 else track['source_group_id'].split(':')[1].replace('Vehicle_', '').replace('Bank', '') if cls == 1 else '远处枪声'
            own = track.get('source_role') == 'self'
            if own:
                name = '自身' + (name if cls != 2 else '枪声')
            actor = dict(cls=cls, name=name, own=own, points=np.round(points, 4).tolist(), activity=frame_activity[i],
                observable=frame_observable[i],collisionOnly=track.get('collision_only',False))
            if cls == 0:
                gait=np.digitize(path['speed'],[.15,2.4,4.8])
                changes=np.r_[0,np.flatnonzero(np.diff(gait))+1,len(gait)-1]
                for begin,end in zip(changes,changes[1:]):
                    timeline.append([name,f'{path["time"][begin]:.2f}–{path["time"][end]:.2f} 秒',
                        ['停步；保留尾音','行走；同地面交替','跑步；同地面交替','冲刺；同地面交替'][gait[begin]]])
            elif cls == 1:
                actor['lifecycle'] = track['lifecycle']
                start_event = next((e for e in track['events'] if e['role'] == 'startup'),None)
                actor['startupEnd'] = start_event['onset_sample'] / 44100 + sources[start_event['source_id']]['duration_seconds'] if start_event else 0
                for e in track['events']:
                    onset = e['onset_sample'] / 44100
                    end = min(30, onset + sources[e['source_id']]['duration_seconds'])
                    timeline.append([name, f'{onset:.1f}–{end:.1f} 秒', {'startup':'发动机启动（原声）',
                        'shutdown':'发动机熄火（原声）','collision':'碰撞（客户端原声，车辆标签）'}[e['role']]])
                if 'emergency_brake_seconds' in track and not track.get('collision_only'):
                    a,b=track['emergency_brake_seconds']
                    timeline.append([name,f'{a:.2f}–{b:.2f} 秒','急刹；原生轮胎摩擦层随制动力变化'])
                segments = track['trajectory']['speed_knots']
                for (begin, v0), (end, v1) in zip(segments, segments[1:]):
                    if v0 == v1 == 0:
                        if begin > track['lifecycle']['move_seconds'] and end < track['lifecycle']['shutdown_seconds']:
                            timeline.append([name, f'{begin:.1f}–{end:.1f} 秒', '刹停后怠速；位置保持不变'])
                        continue
                    action = '起步加速' if v0 == 0 else '加速' if v1 > v0 else '巡航' if v1 == v0 else '减速并刹停' if v1 == 0 else '减速'
                    timeline.append([name, f'{begin:.1f}–{end:.1f} 秒', action])
            else:
                timeline.append([name, '0–30 秒', '自身 Local 枪声；听者相对位置固定' if own else '间歇枪声；枪手位置固定'])
            tracks.append(actor)
        order = {track['name']: i for i, track in enumerate(tracks)}
        timeline.sort(key=lambda entry: (order[entry[0]], float(entry[1].split('–')[0])))
        for weather in row.get('backgrounds',[]):
            timeline.append(['环境背景','0–30 秒',weather['name']+'；无单一方位，三类目标均为负例'])
        csv_file = f'逐帧标注/{row["split"]}/{row["id"]}.csv'
        bundle.append(dict(id=row['id'], title=row['title'], audio=row['audio_file'], tracks=tracks, timeline=timeline,
            backgrounds=row.get('backgrounds',[]),frameCsv=csv_file if (root / csv_file).exists() else None))
    template = Path(__file__).with_name('scenario_preview.html').read_text(encoding='utf-8')
    data = json.dumps(bundle, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    html = template.replace('/*__SCENE_DATA__*/', 'const sceneData = ' + data + ';')
    (root / '试听与轨迹.html').write_text(html, encoding='utf-8')
    lines = ['# 轨迹试听', '', '每段 30 秒。每条脚步轨迹固定一种地面；左右脚仅为交替节奏标记，原始素材没有左右脚标签。',
        '', '车辆启动、熄火来自客户端原声。发动机转速随车速/负载变化，但该部分为合成近似，未还原 REV `.model`。', '',
        f'[打开同步轨迹试听页](<{(root / "试听与轨迹.html").as_posix()}>)', '',
        '各段保持相同播放音量比较。自身位置为听者相对原点，方位不适用；近处外部声音仍标为外部。', '',
        '| 样本 | 内容 | 生成的方位与距离 | 方位/远近是否正确 | 动作/自身听感是否自然 | 备注 |', '|---|---|---|---|---|---|']
    for row in rows:
        descriptions = []
        for i, track in enumerate(row['tracks']):
            path = motion.trajectory(track['trajectory'], 30)
            if track.get('source_role') == 'self':
                descriptions.append(f'声源{i+1}：自身，方位不适用，位置0')
            else:
                descriptions.append(f'声源{i+1}：{round(path["azimuth"][0]) % 360}°→{round(path["azimuth"][-1]) % 360}°，距离{path["distance"].min():.1f}–{path["distance"].max():.1f}')
        if not row['tracks']:
            descriptions.append('天气背景，方位与距离不适用；脚步/车辆/枪声负例')
        lines.append(f'| [{row["id"]}]({(root / row["audio_file"]).as_posix()}) | {row["title"]} | {"；".join(descriptions)} |  |  |  |')
    lines.extend(['', '轨迹文件中的方位角：正前 0°、右侧 90°、正后 180°、左侧 270°。距离为合成单位，只用于粗分远近。', '',
        '音频、标签和轨迹已通过程序检查；表中听感项留空，未代填人工验收。'])
    if not (root / '试听索引.md').exists():
        (root / '试听索引.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (root / '试听列表.m3u8').write_text('#EXTM3U\n' + '\n'.join(str(root / r['audio_file']) for r in rows) + '\n', encoding='utf-8')
    print(json.dumps(dict(player=str(root / '试听与轨迹.html'), samples=len(rows), bytes=len(html.encode('utf-8'))), ensure_ascii=False))


if __name__ == '__main__':
    main()
