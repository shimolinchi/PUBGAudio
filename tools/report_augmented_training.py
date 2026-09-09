"""Verify and report the F034 combined data/augmentation/budget update."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

CLASSES=('footsteps','vehicle','gunfire')
NAMES=('脚步','车辆','枪声')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def history(path):
    return [json.loads(line) for line in (path/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]


def macro(report):
    return sum(report['diagnostics']['external']['per_class'][c]['f1'] for c in CLASSES)/3


def percent(x):return '—' if x is None else f'{x*100:.2f}%'


def number(x):return '—' if x is None else f'{x:.2f}'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Output stem for JSON/Markdown')
    args=parser.parse_args()
    verification=read(args.dataset/'verification.json')
    data=read(args.dataset/'summary.json')
    info=read(args.run/'run.json');old_info=read(args.baseline/'run.json')
    rows=history(args.run);old_rows=history(args.baseline)
    assert verification['status']=='passed' and verification['clips']==108
    assert verification['extension']['held_out_manifests_unchanged']
    assert info['status']=='training_completed' and info['completed_epochs']==len(rows)==400
    assert [r['epoch'] for r in rows]==list(range(400))
    assert all(r['train_batches']==72 and not r['smoke_only'] for r in rows)
    assert info['early_stopping_enabled'] is False
    assert info['train_scenes']==96 and info['train_crops']==1152
    assert info['train_base_fixed_crops']==2304
    assert info['augmentation']==dict(enabled=True,mirror_probability=.5,crops_per_scene=12)
    assert info['model_config']==old_info['model_config']
    assert info['seed']==old_info['seed'] and info['batch_size']==old_info['batch_size']==16
    assert all(r['learning_rate']==.001 for r in rows+old_rows)
    identity=read(args.run/'training_source_identity.json')
    for name,sha in identity.items():
        assert digest(args.run/'source_snapshot'/name)==sha
        assert digest(Path('src/pubg_audio')/name)==sha,'Training source changed during run'
    comparisons=[]
    for title,folder,records in [('父版250轮',args.baseline,old_rows),('扩充增强400轮',args.run,rows)]:
        reports={}
        for rule,checkpoint,filename in [('loss','best.pt','test.json'),('detection','best_detection.pt','test_detection.json')]:
            chosen=min(records,key=lambda r:r['validation_loss']) if rule=='loss' else max(records,key=lambda r:r['validation_external_macro_f1'])
            test=read(folder/filename)
            assert test['checkpoint_epoch']==chosen['epoch']
            assert test['checkpoint_sha256']==digest(folder/checkpoint)
            assert test['split']=='test' and test['scenes']==6 and test['crops']==144
            assert test['diagnostics']['threshold']==.5
            for split in ('validation','test'):
                assert test['dataset_fingerprint'][split]==verification['manifests_sha256'][split]
            if folder==args.run:assert test['dataset_fingerprint']==verification['manifests_sha256']
            reports[rule]=dict(selected_epoch=chosen['epoch']+1,validation_loss=chosen['validation_loss'],
                validation_macro_f1=chosen['validation_external_macro_f1'],test_macro_f1=macro(test),test=test)
        comparisons.append(dict(name=title,run=str(folder),completed_epochs=len(records),
            optimizer_updates=sum(r['train_batches'] for r in records),reports=reports))
    old=comparisons[0]['reports']['detection']['test'];new=comparisons[1]['reports']['detection']['test']
    for role in ('external','self'):
        for c in CLASSES:
            for key in ('positive_frames','valid_frames','ignored_frames'):
                assert old['diagnostics'][role]['per_class'][c][key]==new['diagnostics'][role]['per_class'][c][key]
    added=set(data['extension']['new_scene_ids'])
    new_scenes=[r for r in data['results'] if r['id'] in added]
    kinds=Counter();observable=Counter()
    for r in new_scenes:kinds.update(r['track_kinds']);observable.update(r['observable_by_class_role'])
    tracking=read(args.run/'tracking-live/wandb_run.json')
    summary=dict(status='verified_completed_combined_update',combined_update_not_single_factor_ablation=True,
        completed_epochs=len(rows),optimizer_updates=sum(r['train_batches'] for r in rows),
        training_validation_seconds=sum(r['elapsed_seconds'] for r in rows),
        dataset_seconds=data['seconds'],new_scenes=len(new_scenes),new_track_kinds=dict(kinds),
        new_observable_track_frames=dict(observable),new_collisions=sum(r['collision_events'] for r in new_scenes),
        fixed_held_out_hashes=verification['manifests_sha256'],reused_existing_test_split=True,
        fixed_threshold=.5,real_game_accuracy_not_evaluated=True,comparisons=comparisons,wandb_url=tracking['url'])
    lines=['# 轻增强与枪声、车辆扩充训练结果（F034）','',
        '已完成400轮、28,800次参数更新。训练场景从48增至96段（96→192分钟），新增枪声/车辆为主的数据；启用每轮随机五秒裁剪和50%空间镜像。原有数据、验证/测试保持原样。',
        '',f"新增48段中有枪声{kinds['gunfire']}条、车辆{kinds['vehicle']}条、脚步{kinds['footsteps']}条轨迹，含{summary['new_collisions']}次车辆碰撞。全数据108段共3.6小时，362源家族/PCM分区核验及4段精确重放通过。",'',
        '本次同时改变数据、增强和预算，是组合更新，不能将效果归因于某一项。沿用相同模型、损失、初始化种子、batch16和Adam固定学习率0.001。每轮从每个场景抽12个窗口，共1152次采样/72次更新；400轮比旧250轮多60%参数更新。','',
        f"训练与验证循环累计{summary['training_validation_seconds']/60:.1f}分钟，不含数据生成、缓存与最终测试。训练过程见[新W&B运行]({tracking['url']})。",'',
        '## 检测模型：在相同固定合成测试集上复评','',
        '主要比较使用两个版本各自按最高外部验证宏平均F1保存的模型，阈值固定0.5。测试没有用于选轮次、权重或阈值。该测试集此前已查看，不是新盲测，也不是实战准确率。','',
        'W&B验证曲线的最后一个点不是最终测试结果。例如外部枪声：','',
        '| 指标 | 使用模型 | 外部枪声F1 |','|---|---:|---:|',
        f"| `validation/external/gunfire/f1`，最后一轮 | 第400轮 | {percent(rows[-1]['validation_diagnostics']['external']['per_class']['gunfire']['f1'])} |",
        f"| `validation/external/gunfire/f1`，检测规则选中轮 | 第{new['checkpoint_epoch']+1}轮 | {percent(rows[new['checkpoint_epoch']]['validation_diagnostics']['external']['per_class']['gunfire']['f1'])} |",
        f"| `test_detection/external/gunfire/f1`，固定测试集 | 第{new['checkpoint_epoch']+1}轮 | {percent(new['diagnostics']['external']['per_class']['gunfire']['f1'])} |",'',
        '验证枪型为M16A4，测试枪型为Kar98k，均与训练枪型家族隔离。此处80.37%→55.91%体现跨测试分布的泛化差距；不能把验证集上的0.7以上当成测试F1。另须区分`self`（自身）和`external`（外部）的枪声指标。','',
        '| 外部声音 | 父版250轮F1 | 本次400轮F1 | 变化（百分点） | 本次召回率 |','|---|---:|---:|---:|---:|']
    for c,name in zip(CLASSES,NAMES):
        a=old['diagnostics']['external']['per_class'][c];b=new['diagnostics']['external']['per_class'][c]
        lines.append(f"| {name} | {percent(a['f1'])} | {percent(b['f1'])} | {(b['f1']-a['f1'])*100:+.2f} | {percent(b['recall'])} |")
    lines.extend([f"| 等权宏平均 | {percent(macro(old))} | {percent(macro(new))} | {(macro(new)-macro(old))*100:+.2f} | — |",'',
        '本次车辆检测改善，但枪声漏检增加、脚步F1下降，宏平均F1也下降。不能称为全面优于父版，旧模型保留。测试使用训练未见的枪型/车辆家族；验证提升没有完全迁移到测试，单凭这次组合实验不能判定是增强、分布比例还是训练预算造成。','',
        '| 自身声音 | 父版F1 | 本次F1 | 本次召回率 |','|---|---:|---:|---:|'])
    for c,name in zip(CLASSES,NAMES):
        a=old['diagnostics']['self']['per_class'][c];b=new['diagnostics']['self']['per_class'][c]
        lines.append(f"| {name} | {percent(a['f1'])} | {percent(b['f1'])} | {percent(b['recall'])} |")
    lines.extend(['','## 方向与距离','',
        '以下只统计已检出的单声源帧；两次参与统计的帧数可能不同，需结合上面的召回率。距离是未实录标定的合成单位，不能称为米级精度。','',
        '| 外部声音 | 方向MAE：旧→新（°） | 距离MAE：旧→新（合成单位） | 定位帧数：旧→新 |','|---|---:|---:|---:|'])
    for c,name in zip(CLASSES,NAMES):
        a=old['diagnostics']['external']['per_class'][c]['conditional_localization'];b=new['diagnostics']['external']['per_class'][c]['conditional_localization']
        lines.append(f"| {name} | {number(a['angular_mae_degrees'])} → {number(b['angular_mae_degrees'])} | {number(a['distance_mae_units'])} → {number(b['distance_mae_units'])} | {a['detected_single_source_frames']} → {b['detected_single_source_frames']} |")
    lines.extend(['','## 两种预先确定的选模规则','',
        '| 运行 | 规则 | 选中轮次 | 验证loss | 验证宏F1 | 测试宏F1 |','|---|---|---:|---:|---:|---:|'])
    for comparison in comparisons:
        for key,title in [('loss','最低总验证loss'),('detection','最高外部验证宏F1')]:
            r=comparison['reports'][key]
            lines.append(f"| {comparison['name']} | {title} | {r['selected_epoch']} | {r['validation_loss']:.6f} | {percent(r['validation_macro_f1'])} | {percent(r['test_macro_f1'])} |")
    loss_test=comparisons[1]['reports']['loss']['test']['diagnostics']['external']['per_class']
    lines.extend(['','保留`best.pt`及`best_detection.pt`，分别导出`model.pt`与`model_detection.pt`。表格没有根据测试结果在两种规则之间另选最优模型。',
        f"另一条预定规则（总验证损失最低，第229轮）得到脚步/车辆/枪声测试F1：{percent(loss_test['footsteps']['f1'])} / {percent(loss_test['vehicle']['f1'])} / {percent(loss_test['gunfire']['f1'])}。两种规则的效果差异一并保留，不用测试结果回头更改选模标准。",'',
        f"- [检测验证最优模型，第339轮](../{args.run.name}/model_detection.pt)",
        f"- [总验证损失最优模型，第229轮](../{args.run.name}/model.pt)",
        f"- 新数据目录：`{args.dataset.as_posix()}`",'- 具体生成概率、源家族限制、增强标签同步及复现命令：[F034规范](../../docs/light_augmentation_gunvehicle_v5_3.md)。','',
        '![训练曲线与固定测试对比]('+args.output.name+'.png)','',
        '新数据仍只含脚步、车、枪与自身角色/碰撞；未包含完整身体动作、投掷物、天气及实录分布。未额外加噪、调增益或做频谱遮挡，网络仍为五秒离线上下文。音量/声卡标定、实时接入和真实游戏效果尚未验证。',''])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    args.output.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status=summary['status'],epochs=len(rows),updates=summary['optimizer_updates'],
        selected_detection_epoch=new['checkpoint_epoch']+1,old_macro_f1=macro(old),new_macro_f1=macro(new)),ensure_ascii=False))


if __name__=='__main__':main()
