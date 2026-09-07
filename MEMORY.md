# PUBGAudio 项目记忆

- 产品/仓库名最终为 **PUBGAudio**：用户要求含 PUBG 且简短。主项目路径 `C:/Projects/PUBGAssistant-cpp/modules/PUBGAudio`；私有仓库 `shimolinchi/PUBGAudio`。训练、合成器和本地数据同处模块；音频、NPZ、权重不上传普通 Git。
- 先复刻 Krause/Politis/Mesaros EUSIPCO2024 arXiv:2403.11827v2：24kHz 双耳4×250×512；3CNN128（时间5/1/1、频率8/8/4）→2BiGRU128→2×8headMHA→FC。multi-ACCDDOA3tracks×C×4/ADPIT及MT都实现。论文文本末频点4与池化矛盾，按表得到2；缺省细节用DCASE2024基线并记录。非因果，未复现论文分数。
- `paper_13class` 无自身头1,062,812参数；`pubg_paper_backbone` 外部3类+自身3类1,064,231。按训练集归一化，5秒输入/100ms目标；原真值10ms。被遮正例mask0；近≠自身；Local/FPP来源独立标自身，外部不区分敌友；飞机正例缺失mask0。距离为合成单位，训练除100。
- 脚步先曲线轨迹，整条含走跑冲刺及交替左右始终同地面；素材无真正L/R标签。停住保留尾音/步相。车辆原生启停/制动/碰撞；REV ADM3未解码，怠速变速+负载滤波是近似。自身相对原点，方位无效，保留近处外部反例。
- v4本地 `datasets/audio-synthetic-20260907-v4`：3000×30秒，2400/300/300=20/2.5/2.5小时；691素材含新增30碰撞5天气。84.688%外部有声帧远；2167车轨迹/867碰撞；707天气，141纯天气。自身车时脚步候选保留10%、整段≥90单位外车5%；额外功率门限脚步−12dB、远车−15dB、一般−30dB；均合成假设未实测。恒速平台移动时长2.5%，原18%。所有帧保留位置/活动/observable，遮住不伪造负例。
- 沙尘暴Inner/Outer/Center已溯源解码；Pillar风及SLBStorm分别验证/测试家族，与训练沙尘暴隔离。暴风雪仅查到媒体269897396的命名，未唯一定位/解码，不冒充。流式前缀需按全RIFF长度+字节前缀匹配唯一完整载荷并验hash。
- 全量3000WAV/NPZ及每帧几何审计通过；15段音频及所有标签精确复现。来源/PCM/家族不跨划分。冻结生成器在 `datasets/registry/v4_generator_snapshot`；其中内部dataset_id沿用当时名称，不回写已生成数据。整体移动后复现用 `--sources`/`--hrir` 覆盖历史绝对路径。
- 推荐试听 `outputs/scenario-preview-20260907-v4-final/试听与轨迹.html`：15×30秒，曲线、原声碰撞、巡航vs3种沙尘暴、每帧CSV、手填反馈CSV。15段全部精确复现；反馈保持空。原v3/v4早期输出保留。本轮浏览器工具拒绝file页面访问，因此不能声称已完成最终UI交互验收。
- 12项测试通过；CPU PyTorch2.8.0，已安装editable包。对全量数据进行了2训练batch+1验证batch，再resume同规模；有300帧离线预测输出。权重仅流程smoke，未全量GPU训练/准确率验证/ADPIT去重校准/ONNX/实时C++接入。
- 原客户端PAK审计/本机vgmstream不分发；重新混音与训练使用模块内已准备WAV/HRIR。保留主项目及旧v1/v2/v3源码/数据，不改C++或打包。
