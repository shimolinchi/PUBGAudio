"""Compare fixed budgets within one optimization run using frozen selection rules."""
import argparse
import hashlib
import json
from pathlib import Path


CLASSES = ('footsteps', 'vehicle', 'gunfire')
NAMES = ('脚步', '车辆', '枪声')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def records(path):
    return [json.loads(s) for s in (path/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]


def macro(report):
    rows = report['external']['per_class']
    values = [rows[c]['f1'] for c in CLASSES if rows[c]['positive_frames'] > 0]
    return sum(values)/len(values)


def pct(value):
    return '—' if value is None else f'{value*100:.2f}%'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    info = read(run/'run.json')
    history = records(run)
    early = records(run/'budget_30')
    verification = read(args.dataset/'verification.json')
    assert info['status'] == 'training_completed'
    assert info['completed_epochs'] == len(history) == 250
    assert [r['epoch'] for r in history] == list(range(250))
    assert early == history[:30], 'Paired snapshot must belong to this same run'
    assert all(r['train_batches'] == 72 for r in history)
    assert info['early_stopping_enabled'] is False
    assert verification['status'] == 'passed'
    rows = []
    for budget, folder, hist in [(30, run/'budget_30', early), (250, run, history)]:
        for rule, checkpoint, filename in [
            ('总验证损失最低', 'best.pt', 'test.json'),
            ('验证检测宏平均F1最高', 'best_detection.pt', 'test_detection.json')]:
            chosen = min(hist, key=lambda r:r['validation_loss']) if checkpoint == 'best.pt' else max(
                hist, key=lambda r:r['validation_external_macro_f1'])
            test = read(folder/filename)
            assert test['split'] == 'test' and test['scenes'] == 6 and test['crops'] == 144
            assert test['checkpoint_epoch'] == chosen['epoch']
            assert test['checkpoint_sha256'] == hashlib.sha256((folder/checkpoint).read_bytes()).hexdigest()
            assert test['dataset_fingerprint'] == verification['manifests_sha256']
            assert test['diagnostics']['threshold'] == .5
            rows.append(dict(budget=budget, rule=rule, checkpoint=str(folder/checkpoint),
                epoch=chosen['epoch']+1, validation_loss=chosen['validation_loss'],
                validation_macro_f1=chosen['validation_external_macro_f1'],
                test_macro_f1=macro(test['diagnostics']), test=test))
    budget_curve = []
    for cap in [10, 30, 50, 80, 100, 150, 200, 250]:
        subset = history[:cap]
        low = min(subset, key=lambda r:r['validation_loss'])
        high = max(subset, key=lambda r:r['validation_external_macro_f1'])
        budget_curve.append(dict(budget=cap, best_loss_epoch=low['epoch']+1,
            best_loss=low['validation_loss'], best_macro_epoch=high['epoch']+1,
            best_macro_f1=high['validation_external_macro_f1']))
    tracking = read(run/'tracking-live/wandb_run.json')
    summary = dict(status='verified_completed_budget_comparison', completed_epochs=250,
        optimizer_updates=sum(r['train_batches'] for r in history),
        training_validation_seconds=sum(r['elapsed_seconds'] for r in history),
        paired_prefix_verified=True, fixed_threshold=.5,
        reused_existing_test_split=True, real_game_accuracy_not_evaluated=True,
        rows=rows, budget_curve=budget_curve, wandb_url=tracking['url'])
    lines = ['# 30轮与250轮训练对照（F032）', '',
        '已完成250轮训练。下面的30轮截面与250轮结果来自同一次初始化和优化轨迹；'
        '前30轮权重在第30轮结束时冻结，因此训练轮数的比较不混入两次GPU运行的差异。', '',
        '原先的30轮运行仍保留，验证总损失的最佳轮为15，但检测宏平均F1在第26轮仍有提高。'
        '提前停止偏保守的证据及论文250轮/75轮patience见[轮数复查](../../docs/training_budget_review.md)。', '',
        '## 对照设置', '',
        '- 同一2小时轻量合成数据，训练/验证/测试96/12/12分钟；362个源文件按家族及PCM隔离。',
        '- 同一约106万参数论文骨干、损失、batch16、Adam固定学习率0.001和种子202609077。未新增数据增强或学习率调度。',
        '- 每轮1,152个五秒训练裁剪、72次更新。250轮共18,000次更新；重复暴露400小时，不等于新增400小时独立数据。',
        '- 显式关闭提前停止，完整跑250轮；该固定预算设置不同于论文的75轮patience。',
        '- 两种选模规则在测试前固定：总验证损失最低、外部三类验证宏平均F1最高。阈值固定0.5；不用测试结果更改选模。',
        '- 测试集与上一轮相同，属于复用既有测试划分，不是新的盲测集。每个不同权重仅做一次本次测试；预算之间若checkpoint哈希相同，复用同一份评估。',
        f'- 训练及逐轮验证合计{summary["training_validation_seconds"]:.1f}秒，不含缓存准备、初始化、最终测试及同步。', '',
        '## 同一次训练的预算截面（仅验证集）', '',
        '| 最多训练轮数 | 最低损失所在轮 | 最低验证损失 | 最高检测F1所在轮 | 最高检测宏平均F1 |',
        '|---|---:|---:|---:|---:|']
    for item in budget_curve:
        lines.append(f'| {item["budget"]} | {item["best_loss_epoch"]} | {item["best_loss"]:.6f} | '
                     f'{item["best_macro_epoch"]} | {pct(item["best_macro_f1"])} |')
    lines += ['', '总损失与检测F1是不同目标。更多轮数不会保证每一轮或每一个输出都更好；'
              '上表记录预算以内的最佳值，不是最后一轮的值。', '',
              '## 测试结果：外部声音', '',
              '| 预算 | 验证集选模规则 | 选中轮次 | 脚步F1 | 车辆F1 | 枪声F1 | 三类宏平均F1 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        d = row['test']['diagnostics']['external']['per_class']
        lines.append(f'| {row["budget"]} | {row["rule"]} | {row["epoch"]} | ' +
                     ' | '.join(pct(d[c]['f1']) for c in CLASSES) + f' | {pct(row["test_macro_f1"])} |')
    before, after = rows[1], rows[3]
    gain = (budget_curve[-1]['best_macro_f1']-budget_curve[-2]['best_macro_f1'])*100
    lines += ['', f'相同检测选模规则下，30轮到250轮的测试宏平均F1由{pct(before["test_macro_f1"])}变为'
              f'{pct(after["test_macro_f1"])}，变化{(after["test_macro_f1"]-before["test_macro_f1"])*100:+.2f}个百分点。'
              f'总损失规则在30/250轮预算内分别选择第{rows[0]["epoch"]}/{rows[2]["epoch"]}轮；'
              '增加训练轮数必须配合合适的选模目标。', '',
              f'验证检测F1在200轮预算内最高{pct(budget_curve[-2]["best_macro_f1"])}，'
              f'250轮预算内最高{pct(budget_curve[-1]["best_macro_f1"])}，后50轮最佳值增加约{gain:.2f}个百分点。'
              '是否继续增加预算应结合这条曲线；结论仅针对这一个种子和这批合成数据，不应预设1000轮为必要门槛。', '',
              'F1为100ms帧级检测指标，不是准确率或官方SELD分数。同类三条ADPIT轨取最大活动向量模长；'
              '重复轨不计作多个正确检出。', '',
              '| 预算 / 规则 | 脚步召回 | 车辆召回 | 枪声召回 | 自身脚步F1 | 自身车F1 | 自身枪F1 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        d = row['test']['diagnostics']
        values = [d['external']['per_class'][c]['recall'] for c in CLASSES]
        values += [d['self']['per_class'][c]['f1'] for c in CLASSES]
        lines.append(f'| {row["budget"]} / {row["rule"]} | ' + ' | '.join(map(pct, values)) + ' |')
    lines += ['', '## 条件定位诊断', '',
              '| 预算 / 规则 | 声音 | 已检出的单声源帧 | 方向MAE | 距离MAE（合成单位） |',
              '|---|---|---:|---:|---:|']
    for row in rows:
        for c, name in zip(CLASSES, NAMES):
            loc = row['test']['diagnostics']['external']['per_class'][c]['conditional_localization']
            fmt = lambda x:'—' if x is None else f'{x:.2f}'
            lines.append(f'| {row["budget"]} / {row["rule"]} | {name} | '
                         f'{loc["detected_single_source_frames"]} | {fmt(loc["angular_mae_degrees"])}° | '
                         f'{fmt(loc["distance_mae_units"])} |')
    lines += ['', '需要逐项对照方向、距离与自身类别的结果，不能把外部检测的改善概括成所有任务都更好。', '',
              '定位误差只针对已检出的单声源帧，模型之间的统计样本可能不同，必须同时看召回率。'
              '距离尚未实录标定，不能称为米级精度；未验证真实游戏、跨音量或跨声卡效果。', '',
              '## 曲线与文件', '',
              '![训练及验证曲线](training_budget_comparison_20260907.png)', '',
              f'- [W&B完整曲线]({tracking["url"]})；总损失选模测试在`test/*`，检测选模测试在`test_detection/*`。',
              f'- 本次运行：`{run.as_posix()}`；两类训练checkpoint为`best.pt`和`best_detection.pt`。',
              '- 导出的推理权重分别为`model.pt`和`model_detection.pt`；保留配置、归一化和选模依据，不包含优化器。',
              '- `budget_30`保存同一运行的30轮权重及其测试结果；旧30轮运行未改动。',
              '- `training_source_identity.json`及`source_snapshot`记录本轮实际训练代码，`metrics.jsonl`保存逐轮原始值。',
              '- 本报告旁的同名JSON保留完整测试指标与检查记录。没有扩大网络或重新生成数据。', '']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    args.output.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=summary['status'], report=str(args.output),
        optimizer_updates=summary['optimizer_updates'], rows=[{k:v for k,v in r.items() if k!='test'} for r in rows])))


if __name__ == '__main__':
    main()
