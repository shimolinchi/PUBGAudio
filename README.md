# PUBGAudio

双耳音频的脚步、车辆、枪声识别，以及声源方向、粗距离估计。训练代码、合成器、本地数据集和试听页面放在同一个独立项目中。

## 当前实现

- 按 Krause、Politis、Mesaros 的 EUSIPCO 2024 论文实现 **3 CNN → 2 层 BiGRU → 2 层多头注意力 → 输出头**，支持 multi-ACCDDOA/ADPIT 和 multi-task 两种输出。
- 论文 13 类配置与本项目 3 类配置分开。本项目额外有 3 类自身声音头；它是扩展，不属于论文原网络。
- 已实现训练、验证、独立测试集诊断、训练集归一化、checkpoint、断点恢复和离线预测。小规模训练结果见 `outputs/reports/small_training_20260907.md`；尚未进行完整 25 小时训练或真实游戏准确率验证。
- 本地 v4 数据为 3000 段 × 30 秒：训练 20 小时，验证、测试各 2.5 小时。约 84.69% 的外部发声帧为远处；含 867 次碰撞、707 段天气背景，其中 141 段只有天气。
- 15 段试听包含远处曲线脚步/车辆、自身声音、急刹与原生碰撞、巡航和三种沙尘暴循环声对照。界面同步显示位置和方位，可填写结果并导出 CSV。

论文：[Sound Event Detection and Localization with Distance Estimation](https://arxiv.org/abs/2403.11827)。详细对应关系见 [复刻说明](docs/paper_reproduction.md)，生成规则见 [数据说明](docs/dataset.md)。

小规模试训已完成；固定阈值下外部三类仍未检出，当前权重尚不可用于实战。增强方法与 20→40→80 小时训练集的对照建议见 [数据增强与扩充](docs/augmentation_and_scaling.md)，该扩充方案尚未执行。

已加入独立项目的 W&B 历史导入和实时训练记录，连接方式见 [W&B 说明](docs/wandb.md)。

## 目录

```text
src/pubg_audio/        网络、特征、损失、数据适配、训练和预测
configs/              论文 13 类 / PUBG 3 类加自身头
tools/synthesis/      客户端源音频准备、连续轨迹、混音、校验、试听页
tests/                网络、标签、遮蔽和连续运动测试
datasets/sources-v4/  已溯源客户端素材（本地，Git 忽略）
datasets/hrir/        空间化 HRIR（本地，Git 忽略）
datasets/audio-synthetic-20260907-v4/  完整 25 小时 WAV/NPZ
datasets/registry/    小型数据登记与校验记录（Git 跟踪）
outputs/scenario-preview-20260907-v4-final/  试听与轨迹.html、WAV、每帧 CSV
outputs/reports/      小型检查报告（Git 跟踪）
```

源码仓库跟踪代码、配置、文档与数据登记。**实际音频、标签、客户端素材和权重都在本地子模块目录内，未上传 GitHub**；普通 clone 不会获得这些大文件。保留原有 v1/v2/v3 数据，不把不同版本的素材划分直接混合训练。

## 安装与训练（在子模块目录执行）

支持 Python 3.10 及以上。本机优先使用 Anaconda 的 `pubg` 环境；原 `.venv` 保留用于 CPU 检查。以下是新机器的 CPU 安装方式；CUDA 机器按其驱动选择相应 PyTorch 2.8 安装包。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e '.[test]'
.venv/Scripts/python.exe -m pytest -q
```

```powershell
# 快速检查整个流程：2 个训练 batch、1 个验证 batch，不代表准确率
.venv/Scripts/python.exe -m pubg_audio.train --dataset datasets/audio-synthetic-20260907-v4 --output outputs/smoke --smoke --batch-size 2

# 完整训练：默认最多 250 epochs，验证集连续 75 epochs 未改善则停止
.venv/Scripts/python.exe -m pubg_audio.train --dataset datasets/audio-synthetic-20260907-v4 --output outputs/train-v4 --device cuda --batch-size 8

# 恢复已有训练；配置和数据清单必须与 checkpoint 相同
.venv/Scripts/python.exe -m pubg_audio.train --dataset datasets/audio-synthetic-20260907-v4 --output outputs/train-v4 --resume outputs/train-v4/last.pt --device cuda --epochs 20

# 离线推理，音频长度须为 5 秒的整数倍
.venv/Scripts/python.exe -m pubg_audio.predict --checkpoint outputs/train-v4/best.pt --audio outputs/scenario-preview-20260907-v4-final/audio/preview/M02.wav --output outputs/predictions/M02.json
```

`predict` 输出每 100ms 的原始方向向量、距离和自身概率。ADPIT 的 3 个轨道可能重复，尚未进行检测阈值校准、去重和事件合并。BiGRU 和注意力使用整个 5 秒窗口，当前是离线模型，不是因果实时网络。

## 少量数据训练测试

72 段 × 30 秒，共 36 分钟；训练 / 验证 / 测试为 48 / 12 / 12 段，各自切成不重叠的 5 秒输入。新轨迹使用独立 `scene-seed`，保持原素材家族划分不变。不要用测试集选择 epoch、阈值或参数。

```powershell
conda activate pubg
# 环境首次安装；已有正确 CUDA 版 torch 时无需重复安装
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[test]'

# 每次重新生成请换新的输出目录；已有本地小数据可以跳过这两步
python tools/synthesis/generate_scenario_dataset.py --sources datasets/sources-v4 --hrir datasets/hrir/kemar-diffuse.zip --output datasets/scenario-small-20260907 --split-counts 48 12 12 --scene-seed 202609071 --workers 4
python tools/synthesis/verify_scenario_dataset.py --dataset datasets/scenario-small-20260907 --reproduce 6

# 5 轮完整遍历训练集，每轮用验证集选取 best.pt
python -m pubg_audio.train --config configs/pubg_small_test.json --dataset datasets/scenario-small-20260907 --output outputs/train-small-20260907 --device cuda --batch-size 4
python -m pubg_audio.evaluate --checkpoint outputs/train-small-20260907/best.pt --dataset datasets/scenario-small-20260907 --split test --output outputs/train-small-20260907/test.json --device cuda --batch-size 4
```

`evaluate` 使用 checkpoint 中冻结的训练集归一化参数，并核对三个数据清单的哈希。输出忽略被遮蔽标签的逐帧 precision / recall / F1；外部三个轨道取最大向量长度判断该类是否存在，避免把重复轨道计成三次。默认阈值固定为 0.5，未校准。方向、距离误差仅在检出的单声源帧上计算，必须结合检测召回率阅读；这些是流程诊断，不是官方 SELD 分数。距离仍为合成单位。

## 本地数据准备与试听

在已有命名素材库上校验并复制，原文件不修改：

```powershell
.venv/Scripts/python.exe tools/import_sources.py --input ORIGINAL_SOURCES EXTRA_CONFUSER_SOURCES --output datasets/sources-v4
.venv/Scripts/python.exe tools/synthesis/generate_scenario_dataset.py --sources datasets/sources-v4 --hrir datasets/hrir/kemar-diffuse.zip --output datasets/new-v4-run --workers 6
.venv/Scripts/python.exe tools/synthesis/verify_scenario_dataset.py --dataset datasets/new-v4-run

.venv/Scripts/python.exe tools/synthesis/generate_scenario_dataset.py --sources datasets/sources-v4 --hrir datasets/hrir/kemar-diffuse.zip --output outputs/new-preview --preview-only
.venv/Scripts/python.exe tools/synthesis/export_motion_labels.py --dataset outputs/new-preview --output outputs/new-preview/逐帧标注
.venv/Scripts/python.exe tools/synthesis/build_scenario_preview.py --preview outputs/new-preview
```

直接打开生成目录的 `试听与轨迹.html`。F01–F04 为远处、P01–P03 为自身、M01–M02 为混合、N01 为近处外部反例，C01 碰撞、C02 巡航，W01–W03 沙尘暴。试听反馈初始为空，不代填人工验收。

原始 PAK 审计和 vgmstream 解码器是本机既有的外部准备工具，不随源码仓库分发；`prepare_*_sources.py` 的命令行显式接收它们的位置。正常训练和再次混音只依赖已准备的 `sources.json`、WAV 和 HRIR，不读取游戏进程。

## 仓库状态

公开仓库：[shimolinchi/PUBGAudio](https://github.com/shimolinchi/PUBGAudio)，主项目内路径为 `modules/PUBGAudio`。训练数据保存在本地，不随 Git 克隆下载。数据路径可整体移动，复现工具可用 `--sources`、`--hrir` 指定新位置。

已生成全量 v4 的生产脚本保存在 `datasets/registry/v4_generator_snapshot/`，其中内部数据 ID 保留生成时的标识。当前生成器的产品名与试听碰撞对照元数据已有更新；复刻既有全量数据请使用该快照，以保留原参数和场景序列。
