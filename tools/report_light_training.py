"""Write a factual report from completed local training and held-out metrics."""
import argparse
import hashlib
import json
from pathlib import Path


def percent(value):return '—' if value is None else f'{value*100:.2f}%'
def number(value):return '—' if value is None else f'{value:.2f}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run=args.run
    info=json.loads((run/'run.json').read_text())
    test=json.loads((run/'test.json').read_text())
    metrics=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    verification=json.loads((args.dataset/'verification.json').read_text())
    selected=min(metrics,key=lambda r:r['validation_loss'])
    assert info['status']=='training_completed' and verification['status']=='passed'
    assert test['checkpoint_epoch']==selected['epoch']
    assert test['checkpoint_sha256']==hashlib.sha256((run/'best.pt').read_bytes()).hexdigest()
    assert len(metrics)==info['completed_epochs']
    assert test['dataset_fingerprint']==verification['manifests_sha256']
    tracking=json.loads((run/'tracking-live/wandb_run.json').read_text()) if (run/'tracking-live/wandb_run.json').exists() else {}
    labels={'footsteps':'脚步','vehicle':'车辆','gunfire':'枪声'}
    epoch=selected['epoch']+1
    lines=['# v5.3 轻量数据训练结果','',
        f'已完成 {info["completed_epochs"]} 轮完整 GPU 训练，按验证损失选择第 **{epoch} 轮**权重，再对独立测试集评估一次。以下是合成数据的诊断结果，不是实战准确率或官方 SELD 分数。','',
        '## 数据与核验','',
        '- 新生成 60 段 × 120 秒，共 2 小时；训练/验证/测试为 48/6/6 段，即 96/12/12 分钟。',
        '- 仅保留脚步、车辆、枪声及自身来源，含曲线运动、转头、连续步态、车辆状态与 10 次碰撞；复杂环境及事件链省略。',
        f'- {verification["source_files_verified"]} 个客户端源文件，完整家族与 PCM 分区，无跨分区原声；全部 60 段文件、标签、几何、三类投影核验通过。',
        '- 训练/验证/测试各一段精确重放通过：'+', '.join(verification['exact_replays'])+'。',
        f'- PCM 峰值 {verification["pcm_peak"]:.4f}，无削波。原始 10ms 多轨真值保留，训练使用 100ms 多源目标和遮蔽/裁剪边界屏蔽。','',
        '## 训练设置','',
        f'- 参数量 {info["parameters"]:,}；3 CNN → 2 BiGRU → 2 MHA，ADPIT 外部三类 + 自身三类。5 秒离线上下文，未改为因果网络。',
        f'- {info["device_name"]}，PyTorch {info["torch_version"]}，batch {info["batch_size"]}，Adam 0.001；最多 80 轮、验证损失 15 轮未改善停止。',
        f'- 每轮完整遍历 {info["train_crops"]} 个训练裁剪、{info["validation_crops"]} 个验证裁剪；归一化仅拟合训练集。未新增在线噪声或镜像增强。',
        f'- 训练与每轮验证合计 {sum(r["elapsed_seconds"] for r in metrics):.1f} 秒，不含生成、缓存、初始化、最终测试和同步。',
        f'- 最佳验证损失 {selected["validation_loss"]:.6f}；独立测试损失 {test["loss"]["total"]:.6f}。阈值固定 0.5，没有用测试集调参。','',
        '## 独立测试集','',
        '| 声音 | 外部 Precision | 外部 Recall | 外部 F1 | 自身 F1 |',
        '|---|---:|---:|---:|---:|']
    d=test['diagnostics']
    for cls,label in labels.items():
        ext=d['external']['per_class'][cls];own=d['self']['per_class'][cls]
        lines.append(f'| {label} | {percent(ext["precision"])} | {percent(ext["recall"])} | {percent(ext["f1"])} | {percent(own["f1"])} |')
    lines+=['','| 声音 | 检出的单声源帧 | 条件方向 MAE | 条件距离 MAE（合成单位） |',
            '|---|---:|---:|---:|']
    for cls,label in labels.items():
        loc=d['external']['per_class'][cls]['conditional_localization']
        lines.append(f'| {label} | {loc["detected_single_source_frames"]} | {number(loc["angular_mae_degrees"])}° | {number(loc["distance_mae_units"])} |')
    lines+=['','方向和距离误差仅统计已检出的单声源帧，漏检不在其中，必须结合召回率阅读。三条 ADPIT 轨可能重复，分类指标取最大方向向量模长，不把重复轨当成三个目标。','',
        '## 权重与复现','',
        f'- 推理权重：`{run.as_posix()}/model.pt`（不含优化器，保留归一化、配置和类别顺序）。',
        f'- 训练权重：`{run.as_posix()}/best.pt`；断点：`last.pt`；完整指标：`metrics.jsonl`、`test.json`。',
        f'- 源码快照：`{run.as_posix()}/source_snapshot`；清单指纹：`training_source_identity.json`。',
        '- 生成规范、概率与命令见 [轻量训练规范](../../docs/light_training_v5_3.md)。']
    if tracking.get('url'):lines.append('- W&B：['+run.name+']('+tracking['url']+')。')
    lines+=['','## 适用边界','',
        '这是轻量合成基线。测试保留了未参与训练的 Snow 脚步、Kar98k 枪声和 Minibus/Sedan 车声家族；少量家族不能代表全部游戏声源。轮胎辅助录音不足以覆盖三个独立完整分区，缺失层省略，详见生成规范。',
        '距离尚未实录标定；没有真实游戏录音、跨系统音量/声卡验证，也未实现实际遮挡、混响及完整客户端声学。这一轮结果不证明实战已可用。','']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(status='report_written',path=str(args.output),epochs=info['completed_epochs'],best_epoch=epoch)),flush=True)


if __name__=='__main__':main()
