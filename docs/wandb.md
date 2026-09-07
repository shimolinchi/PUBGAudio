# W&B 训练记录

所有云端记录都显式指定模块专用的 W&B entity、项目名和项目 ID，不使用全局默认项目。项目绑定放在 Git 忽略的 `configs/wandb.local.json`；不改其他项目或全局项目设置。

## 首次连接

```powershell
conda activate pubg
python -m pip install -e '.[tracking]'
python -m wandb login
python tools/setup_wandb.py
```

登录凭据只交给 W&B 的本机登录流程。脚本优先新建 `PUBGAudio`；若该名称已存在，则使用日期和随机后缀创建另一个新项目。已有绑定时脚本拒绝再次创建，后续直接复用该绑定。默认使用账户的默认团队，也可显式传入 `--entity TEAM`。

## 导入本次已完成训练

```powershell
python tools/import_wandb_history.py --run-dir outputs/train-small-20260907 --overfit-report outputs/reports/small_overfit_check_20260907.json
```

生成两条用途明确的记录：

- `small-72scenes-5epochs`：5 个真实 epoch 的训练/验证损失；验证最佳轮次；独立测试集分类表和 F1 柱状图。固定阈值下外部三类没有检出，这一结果会如实显示。
- `overfit-2-training-crops`：两个已见训练片段的损失与 F1，用于检查能否拟合；不作为测试准确率。归入单独的 `sanity-checks` 分组，指标使用 `sanity/` 前缀。

导入仅使用已保存的实际记录，不补造 batch 曲线。历史导入关闭系统监控，避免把上传时的 GPU 状态当成训练时监控。运行链接保存在训练输出下的 `tracking-history/wandb_run.json` 和 `import_complete.json`；重复完整导入会返回已有结果。

## 后续实时记录

```powershell
python -m pubg_audio.train --config configs/pubg_small_test.json --dataset datasets/scenario-small-20260907 --output outputs/NEW_RUN --device cuda --batch-size 4 --workers 2 --wandb-config configs/wandb.local.json
```

每轮保存本地权重和日志后上传损失。W&B 用 epoch 作为曲线横轴，真实运行可采集系统监控。断点恢复复用该输出目录 `tracking-live/wandb_run.json` 中的运行 ID；缺少 `--wandb-config` 时仅记录到本地。SDK 会核对远端项目 ID，拒绝把同一记录改投另一个项目。

音频、训练标签、客户端素材和模型权重不由该集成上传；上传内容是训练配置、数据清单指纹、指标、分类表和图。当前禁用代码自动保存及 Git 差异上传。

离线验证可用 `--wandb-mode offline`（训练）或 `--mode offline`（导入），必须使用独立的输出目录；离线运行不能被当作已成功上传。参考 [W&B 项目](https://docs.wandb.ai/models/track/project-page)、[日志横轴](https://docs.wandb.ai/models/track/log/customize-logging-axes) 和 [SDK init](https://docs.wandb.ai/models/ref/python/functions/init)。
