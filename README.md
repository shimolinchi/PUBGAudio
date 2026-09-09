# PUBGAudio

双耳音频的脚步、车辆、枪声识别，以及声源方向、粗距离估计。训练代码、合成器、本地数据集和试听页面放在同一个独立项目中。

## 当前实现

F039待用生成规则：减少强车声下的外部脚步，提高外部脚步候选率，加入25%概率的240秒纯外部脚步场景（1–2人、偏远曲线巡逻、更频繁转头）。**只改脚本和配置，尚未生成新数据；当前v6训练保持原样。** 见[v6.1概率表](docs/external_footsteps_v6_1.md)，后续入口`configs/dataset_v6_1.json`。

F036/F037：新的 **24小时v6数据已生成并通过全部核验**（训练20小时，验证/测试各2小时，约19.312 GiB）。覆盖49个武器声音组、8种完整车型；加入全自动射击，降低脚步电平与范围，修订远声滤波和方向监督。主评估明确为“已知客户端素材的新场景测试”。旧合成集/缓存已归档清理，净释放12.375 GiB；**用户已授权并启动400轮正式训练**，查看[W&B进度](https://wandb.ai/simonlynch-harbin-institute-of-technology/PUBGAudio/runs/235fa5787d18)。见[训练记录](outputs/reports/spatial_v6_training_20260908.md)、[当前规范与概率表](docs/spatial_v6.md)和[枪型射击表](docs/weapon_profiles_v6.md)。下列历史数据统计不表示文件仍保留，旧模型保留。

最新F034已完成 **3.6小时数据扩充及400轮训练**：新增48段枪声/车辆为主的训练场景，训练数据192分钟；启用50%空间镜像和每轮随机五秒裁剪，共28800次更新。检测规则选择第339轮，外部测试脚步/车辆/枪声F1为88.5%/93.1%/55.9%，较父版91.1%/85.9%/64.8%，车声改善但枪声退步，不能称为全面升级。总验证loss最优第229轮也保留。两份模型导出、CPU推理及W&B finished核验完成，见[结果与模型](outputs/reports/augmentation_gunvehicle_training_20260907.md)和[概率/增强规范](docs/light_augmentation_gunvehicle_v5_3.md)。旧模型继续保留，未验证实战。

此前已完成 **250轮训练对照**，沿用2小时v5.3三类数据（60×120秒，train/val/test=48/6/6段）。按验证检测宏平均F1选择第206轮，外部测试脚步/车辆/枪声F1为91.1%/85.9%/64.8%；同次运行前30轮的对应结果为69.6%/79.9%/44.3%。推理权重 `outputs/train-light-v5_3-250epochs-20260907/model_detection.pt` 约4.3MB，CPU推理及W&B同步核验通过。总损失规则另保留第19轮；车与枪的方向仍不理想，未验证实战。见[250轮对照报告](outputs/reports/training_budget_comparison_20260907.md)、[轮数复查](docs/training_budget_review.md)及[轻量数据规范](docs/light_training_v5_3.md)。原30轮运行和模型保留。

当前标签组织、10ms原始标注与100ms多源训练目标的关系、真实示例及已修复的裁剪边界问题，见[标注与训练目标](docs/annotation_and_training_targets.md)。

- 最新v5.3：保持脚步角色出现率，减少普通行走、增加跑步；加入同角色连续蹲起、趴起和翻越，含自身/外部声音与10ms动作标签。10段8分钟见[动作说明](docs/body_actions_v5_3.md)及[当前概率表](docs/generation_probability_v5_3.md)。姿态转换使用共享原生衣物声，细姿态是计划标签，不能声称音色能唯一辨认。

- 历史 v5.2 试听：自身手雷/闪光/燃烧瓶按出手时的前方投出，之后转头不移动落点；提高视角活动和车辆加速比例，减少怠速及滑车。14段、15分40秒（含120秒连续场景）见 [行为调整](docs/forward_throw_and_motion_v5_2.md) 和 [当前概率表](docs/generation_probability_v5_2.md)。继承 [v5.1衰减核查与长窗口方案](docs/attenuation_and_listener_v5_1.md)；曲线仍待实录标定，旧25小时数据没有回写。

- 按 Krause、Politis、Mesaros 的 EUSIPCO 2024 论文实现 **3 CNN → 2 层 BiGRU → 2 层多头注意力 → 输出头**，支持 multi-ACCDDOA/ADPIT 和 multi-task 两种输出。
- 论文 13 类配置与本项目 3 类配置分开。本项目额外有 3 类自身声音头；它是扩展，不属于论文原网络。
- 已实现训练、验证、独立测试集诊断、训练集归一化、checkpoint、断点恢复和离线预测。小规模训练结果见 `outputs/reports/small_training_20260907.md`；尚未进行完整 25 小时训练或真实游戏准确率验证。
- 历史 v4 曾生成 3000 段 × 30 秒：训练 20 小时，验证、测试各 2.5 小时；音频与标签已于2026-09-08删除，生成记录保留。历史统计约84.69%的外部发声帧为远处，867次碰撞、707段天气背景，其中141段只有天气。
- 15 段试听包含远处曲线脚步/车辆、自身声音、急刹与原生碰撞、巡航和三种沙尘暴循环声对照。界面同步显示位置和方位，可填写结果并导出 CSV。
- 新增 v5 概率驱动试听：18 段 × 60 秒，8 段独立随机、10 段专项事件链；曲线/连续步态、熄火滑车、投掷物、C4、车辆损伤和 C130。10ms 详细标注与三类训练投影分别保存，不扩大网络、不启动训练。全批核验及 6 段精确重放通过，等待人工试听。

论文：[Sound Event Detection and Localization with Distance Estimation](https://arxiv.org/abs/2403.11827)。详细对应关系见 [复刻说明](docs/paper_reproduction.md)，当前生效规则见 [数据生成规范](docs/dataset.md)，后续调整统一记入 [反馈与修订记录](docs/data_feedback.md)。

早期v4小规模试训的外部三类未检出；最新v5.3轻量结果见上方报告。增强方法与 20→40→80 小时训练集的对照建议见 [数据增强与扩充](docs/augmentation_and_scaling.md)，该扩充方案尚未执行。

已加入独立项目的 W&B 历史导入和实时训练记录，连接方式见 [W&B 说明](docs/wandb.md)。

本地声音的可用范围见 [声音素材清单](docs/audio_source_inventory.md)。v5历史素材池为776个WAV，v5.3最终池为815个（新增39个身体动作原声）；[连续场景规范](docs/natural_scene_rules.md)与[实际概率表](docs/generation_probability_v5.md)记录本版规则、机制来源和待确认项。C4 重复提示间隔为近似，燃烧瓶固定 10 秒仍待校准；并未接入全部 31 个讨论类别。旧 25 小时 v4 数据没有被回写。

最新本地试听：`outputs/body-action-preview-20260907-v5_3-final/试听与轨迹.html`。训练选择配置为 `configs/training_targets_v5.json`：默认脚步/车辆/枪声及其自身标签；其他类别的波形作为干扰保留，被遮蔽的目标保留详细记录并屏蔽训练损失。这批只有试听分组，尚无训练/验证/测试素材隔离。旧v5与v5.1页面及反馈保留。

## 目录

```text
src/pubg_audio/        网络、特征、损失、数据适配、训练和预测
configs/              论文 13 类 / PUBG 3 类加自身头
tools/synthesis/      客户端源音频准备、连续轨迹、混音、校验、试听页
tests/                网络、标签、遮蔽和连续运动测试
datasets/sources-v4/  已溯源客户端素材（本地，Git 忽略）
datasets/hrir/        空间化 HRIR（本地，Git 忽略）
datasets/spatial-v6-24h-20260908/  新 24 小时数据
datasets/registry/    小型数据登记与校验记录（Git 跟踪）
outputs/scenario-preview-20260907-v4-final/  试听与轨迹.html、WAV、每帧 CSV
outputs/reports/      小型检查报告（Git 跟踪）
```

源码仓库跟踪代码、配置、文档与数据登记。**实际音频、标签、客户端素材和权重都在本地子模块目录内，未上传 GitHub**；普通 clone 不会获得这些大文件。试听保留；旧训练音频按用户要求归档元数据后清理，不把不同版本的划分直接混合训练。

## 安装与训练（在子模块目录执行）

支持 Python 3.10 及以上。本机优先使用 Anaconda 的 `pubg` 环境；原 `.venv` 保留用于 CPU 检查。以下是新机器的 CPU 安装方式；CUDA 机器按其驱动选择相应 PyTorch 2.8 安装包。

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -e '.[test]'
.venv/Scripts/python.exe -m pytest -q
```

```powershell
# 下列为历史v4命令：旧25小时数据已删除，须先重建才能执行；最新训练见上方F034规范。
# 快速检查整个流程：2 个训练 batch、1 个验证 batch，不代表准确率
.venv/Scripts/python.exe -m pubg_audio.train --dataset datasets/audio-synthetic-20260907-v4 --output outputs/smoke --smoke --batch-size 2

# 完整训练：默认最多 250 epochs，验证集连续 75 epochs 未改善则停止
.venv/Scripts/python.exe -m pubg_audio.train --dataset datasets/audio-synthetic-20260907-v4 --output outputs/train-v4 --device cuda --batch-size 8

# 恢复已有训练；配置和数据清单必须与 checkpoint 相同
.venv/Scripts/python.exe -m pubg_audio.train --dataset datasets/audio-synthetic-20260907-v4 --output outputs/train-v4 --resume outputs/train-v4/last.pt --device cuda --epochs 20

# 离线推理，音频长度须为 5 秒的整数倍
.venv/Scripts/python.exe -m pubg_audio.predict --checkpoint outputs/train-v4/best.pt --audio outputs/scenario-preview-20260907-v4-final/audio/preview/M02.wav --output outputs/predictions/M02.json
```

`predict` 输出每 100ms 的原始方向向量、距离和自身 sigmoid 分数，并增加每类存在分数与可选校准概率。未加载校准文件时概率为 `null`，原始分数不是已校准的正确率。详见[置信度说明与使用方法](docs/confidence.md)。ADPIT 的 3 个轨道可能重复，尚未进行检测阈值校准、去重和事件合并；没有方向/距离置信区间。BiGRU 和注意力使用整个 5 秒窗口，当前是离线模型，不是因果实时网络。

下一版 v6.1 数据规范已引用 `configs/confidence_v1.json`：生成器保留0/1真值与遮蔽mask、冻结验证校准场景计划；对含此规则的新数据，新版 `tools/run_spatial_training.py` 会在主模型导出后自动校准并输出存在概率。当前运行的旧冻结训练不受影响，本版数据仍未生成。

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
