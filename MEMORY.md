# PUBGAudio 项目记忆

- 产品/仓库名 **PUBGAudio**（含 PUBG 且简短），主项目路径 `C:/Projects/PUBGAssistant-cpp/modules/PUBGAudio`；`shimolinchi/PUBGAudio` 已按用户要求设为 public。训练、合成器和本地数据同处模块；音频、NPZ、权重不上传普通 Git。
- 先复刻 Krause/Politis/Mesaros EUSIPCO2024 arXiv:2403.11827v2：24kHz 双耳4×250×512；3CNN128（时间5/1/1、频率8/8/4）→2BiGRU128→2×8headMHA→FC。multi-ACCDDOA3tracks×C×4/ADPIT及MT都实现。论文文本末频点4与池化矛盾，按表得到2；缺省细节用DCASE2024基线并记录。非因果，未复现论文分数。
- `paper_13class` 无自身头1,062,812参数；`pubg_paper_backbone` 外部3类+自身3类1,064,231。按训练集归一化，5秒输入/100ms目标；原真值10ms。被遮正例mask0；近≠自身；Local/FPP来源独立标自身，外部不区分敌友；飞机正例缺失mask0。距离为合成单位，训练除100。
- 脚步先曲线轨迹，整条含走跑冲刺及交替左右始终同地面；素材无真正L/R标签。停住保留尾音/步相。车辆原生启停/制动/碰撞；REV ADM3未解码，怠速变速+负载滤波是近似。自身相对原点，方位无效，保留近处外部反例。
- v4本地 `datasets/audio-synthetic-20260907-v4`：3000×30秒，2400/300/300=20/2.5/2.5小时；691素材含新增30碰撞5天气。84.688%外部有声帧远；2167车轨迹/867碰撞；707天气，141纯天气。自身车时脚步候选保留10%、整段≥90单位外车5%；额外功率门限脚步−12dB、远车−15dB、一般−30dB；均合成假设未实测。恒速平台移动时长2.5%，原18%。所有帧保留位置/活动/observable，遮住不伪造负例。
- 沙尘暴Inner/Outer/Center已溯源解码；Pillar风及SLBStorm分别验证/测试家族，与训练沙尘暴隔离。暴风雪仅查到媒体269897396的命名，未唯一定位/解码，不冒充。流式前缀需按全RIFF长度+字节前缀匹配唯一完整载荷并验hash。
- 全量3000WAV/NPZ及每帧几何审计通过；15段音频及所有标签精确复现。来源/PCM/家族不跨划分。冻结生成器在 `datasets/registry/v4_generator_snapshot`；其中内部dataset_id沿用当时名称，不回写已生成数据。整体移动后复现用 `--sources`/`--hrir` 覆盖历史绝对路径。
- 推荐试听 `outputs/scenario-preview-20260907-v4-final/试听与轨迹.html`：15×30秒，曲线、原声碰撞、巡航vs3种沙尘暴、每帧CSV、手填反馈CSV。15段全部精确复现；反馈保持空。原v3/v4早期输出保留。本轮浏览器工具拒绝file页面访问，因此不能声称已完成最终UI交互验收。
- 用户指定使用 Anaconda `pubg`：`C:/Users/Rui Wang/anaconda3/envs/pubg/python.exe`，Python3.10.20/PyTorch2.8.0+cu128，RTX5070Ti16GB；NumPy2.2.6/SciPy1.15.3，editable包；保留旧`.venv`CPU环境。代码支持Python3.10/3.12，16项测试及pip check通过。
- 新小数据 `datasets/scenario-small-20260907`：72×30秒，48/12/12=24/6/6分钟；`scene-seed=202609071`独立改变轨迹而不改素材家族划分。88.21%外部远帧，18碰撞、16天气含2纯天气。72段全量审计及9段音频/标签精确复现通过。
- `outputs/train-small-20260907`：5轮完整GPU训练，batch4/workers2；第3轮最佳验证loss0.23529，测试loss0.32602。固定0.5阈值下外部三类F1均0，自身脚步/车/枪F1为0/.5278/.6696；方向/距离因无外部检出而无有效统计，不能记成0。权重未达到实战可用。两段训练crop从随机权重300步记忆检查loss.9212→.000543、F1=1，仅证实可拟合已见样本，非泛化准确率。详细报告在outputs/reports。
- `evaluate`用checkpoint冻结归一化并核对三个manifest哈希；100ms类存在诊断取ADPIT最大向量范数、忽略遮蔽帧，非官方SELD分数。完整25小时尚未GPU训练；ADPIT去重/阈值校准、真实游戏、ONNX/实时C++接入待做。
- 用户询问噪声增强和scale-up：原2024论文仅明确额外20h合成，未说明在线白噪声/SpecAugment；其2021生成方法有6–30dB真实背景。Yeow2025 arXiv2507.00874有左右交换/标签镜像、ITFM/FilterAugment及约86.7h训练。`docs/augmentation_and_scaling.md`为建议：原/轻/中/强30/40/25/5，20→40→80h训练、验证测试固定，尚未执行。新增噪声要重渲染分轨并更新逐帧遮蔽；现有NPZ不含每轨功率，不能直接加噪沿用全部监督。
- 用户要求W&B新项目且不影响其他项目。已安装wandb0.26.1；`setup_wandb.py`新建独立项目并写Git忽略的`configs/wandb.local.json`，按entity/project/id绑定；历史导入和实时训练接入（`--wandb-config`）已离线验证。历史5轮与2crop记忆检查用不同run/group，保留外部未检出事实；不上传音频/权重/源码差异。当前已为用户打开登录终端，云端项目/上传仍待本机登录完成，不能把离线日志说成已上传。
- 原客户端PAK审计/本机vgmstream不分发；重新混音与训练使用模块内已准备WAV/HRIR。保留主项目及旧v1/v2/v3源码/数据，不改C++或打包。
