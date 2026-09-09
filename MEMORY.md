# PUBGAudio 项目记忆

## 约定与环境
- 独立 public 子模块 C:/Projects/PUBGAssistant-cpp/modules/PUBGAudio，https://github.com/shimolinchi/PUBGAudio，main分支。2026-09-09用户明确授权最新版代码提交同步，主仓库master也同步子模块引用；提交版本以git日志/远端引用为准。代码/配置/文档/小型登记报告纳入，音频/标签/权重/凭据/聊天导出仅本地；不改设备或游戏。
- 2026-09-09提交前全套108测试通过；主仓库旧音频工具15测试通过。registry按字节SHA校验，.gitattributes禁其换行转换；.gitignore排除日志，主仓库排除outputs和exports。无新v6.1数据/训练/实际置信校准。
- Python C:/Users/Rui Wang/anaconda3/envs/pubg/python.exe，torch2.8+cu128、RTX5070Ti16GB，8核16线程/31GiB RAM。pytest 可禁 cacheprovider，临时产物放 outputs。
- 论文 Krause/Politis/Mesaros EUSIPCO2024 arXiv2403.11827：3CNN128→2BiGRU128→2×8头MHA→FC；外部 ADPIT3+self3，1,064,231参数。5秒非因果输入→50×100ms，GRU 不跨窗，100ms 不是端到端延迟；未加空间分支/长上下文。

## F041：置信度规则接入下一版生成与训练
- v6.1 spec新增confidence_policy_file=configs/confidence_v1.json，配置构建器同步。生成器保留原二元真值/mask，不造软置信标签；规则嵌入training_targets.json和generation_identity，完整生成后写calibration_plan.json。仅规划无完整清单时不写计划，未生成实际v6.1数据。
- tools/synthesis/confidence_policy.py校验规则/哈希/完整场景互斥；验证场景固定seed2026090811，50/50拟合/审计（至少4场景、各2），不读测试。最少20正/20负有效帧可配置，不足概率null；保留自然帧比例，无声且mask有效仍负例，mask0既不作正也不作负。
- verify_light_dataset.py核验目标定义与身份、计划/清单哈希；calibrate_presence.py读取固定计划，拒绝更换seed/篡改，并记录策略哈希。旧数据无策略仍支持原独立seed参数。
- 新版run_spatial_training.py对带策略的数据冻结校准器及helper，主导出model_localization.pt之后自动产presence_calibration.json，并带--calibration做CPU检查；支持120/240秒，不再固定1200帧。仅主模型自动校准，其他导出另校准；现有旧冻结v6运行不追溯生效。校准只拟合存在概率、不改骨干/损失/门限/选轮，也不代表实战验收。
- 71相关测试通过，流水线模拟训练/导出、校准使用测试夹具；当前冻结源码/配置/清单及旧数据策略不变。报告outputs/reports/confidence_generation_policy_20260908.json。真实模型概率仍未拟合；下一次生成等用户指令。详细规则docs/confidence.md。

## F040：置信度接口已实现，实际校准未拟合
- 当前外部 activity_vector_norm 是可大于1的模长；self_probabilities 是BCE训练后的未校准sigmoid，均非已验证正确概率。已解释正确0/1标签不等于模型能辨认全部声音；无需另标置信度，验证标签拟合分数→存在比例，固定输入输出确定，不代表随机猜测。详见docs/confidence.md原理段。
- src/pubg_audio/confidence.py + predict.py 新增每帧/每类 external_class_predictions/self_class_predictions：presence_score、score_kind、presence_probability、probability_status。未校准/样本不足概率 null；外部同类取三个轨道最大分数，不累计 ADPIT 重复槽。多类可重叠；external 非敌友判断。13类论文模型使用通用类名。
- 原始字段保留；每轨 direction_confidence/distance_confidence 暂 null，没有实现定位不确定性、逐轨概率、事件合并。没有改网络/损失/阈值。
- tools/calibrate_presence.py 绑定确切推理文件 SHA256；验证集完整场景各半拟合/审计单调 sigmoid 映射，沿用 mask，自然正负比例，至少各20帧；Brier/ECE/可靠性分箱报告，不读取测试标签。审计仍是曾用于模型选轮的验证数据，不是独立最终模型测试或真实游戏校准。概率随模型/分布改变须重审，v6.1不能沿用v6。参考及命令 docs/confidence.md。
- 23项针对测试通过；第50轮冻结诊断权重+现有120秒验证音频，新旧1200帧所有原始值完全一致，新概率明确 null。凭据 outputs/verify-confidence-20260908/verification.json。只验证接口，未拟合实际校准或证明精度改善。最终 model_localization.pt 等文件冻结后需执行校准/审计；当前训练冻结源码不包含本次接口，原自动推理也不会自动用它。

## F039：下一版生成规则已改，仍等待生成指令
- 用户要求本人开车/强车时通常不出现外部脚步，增加外部脚步及长1–2人纯脚步曲线、偏远和频繁转头。仅改脚本，没有新v6.1数据/训练。配置 configs/generation_v6_1.json、dataset_v6_1.json；configure_footsteps_v6_1.py 只写配置，详表 docs/external_footsteps_v6_1.md。
- 混合/纯外部脚步75/25%，120/240秒；720段期望30h但未生成，不能报已有30h。纯1/2人65/35%，第二人延迟8–24秒；固定地面/步态，走/跑/冲刺25/41.67/33.33%，移动6–18秒，45%停1–4秒。
- 纯轨迹为有界弧长 oval/figure_eight/bezier_loop 均匀选，远/中/近/往返60/10/10/20%，半径分数.58–.85/.35–.58/.15–.35/.15–.85；防长轨迹走出可听半径。纯视角固定/自然5/95%，小晃/观察/大转55/15/30%，短停.15–.7秒、5%长停2–5秒，角速≤140°/s。
- 混合脚步倍率.6→1，角色可选时自身50→30%，此时外部候选约2.33倍、自身率基本不变；外部并发≤2。混合视角、车/枪及声音电平/衰减不变。
- vehicle_footstep_guard.py 先渲染全段所有车（含后到/滑行/刹车），再筛整条脚步。本人车功率≥1e-8且重叠≥.5秒保留1%；总车≥1e-6、比脚步高6dB、重叠≥.5秒且占活动≥5%保留2%。每轨一次抽样，不裁断单步，筛后保存配方/决策。保留冲突帧 track_vehicle_masked=1、训练mask0，新增 vehicle_power。这是合成策略，非实测阈值。
- 52项生成相关测试通过，假信号仅内存；旧v6首配方统一换行后文本一致。以后生成须审计实际可听帧/概率/远近/重叠、枪型角色及四象限覆盖。存在性loss正负平衡/训练采样调整尚未实施，未解决零检出。

## v6 正式训练与诊断
- F037的400轮已全部完成：outputs/train-spatial-v6-400epochs-20260908，W&B235fa5787d18；pipeline_status为completed，2026-09-08 14:44+08完成流水线，2026-09-09只读核验。导出主定位371轮、检测360轮、loss367轮；export_verification通过，120秒CPU1200帧。保留冻结源码/配置/模型，未实际校准置信概率、未验证实战；勿重复启动。详见outputs/reports/spatial_v6_training_20260908.md。
- 配置400轮/batch16/workers4/threads4/Adam.001，无早停；600场景每轮12随机5秒窗=7200窗/450更新，总180000更新。50%镜像同步x/ILD/sinIPD，无噪声/增益/变速；归一化训练均匀1200窗，训练不存大特征缓存、验证缓存。单轮约36–40秒，初估4–5h。
- F038第50轮冻结权重全验证1440窗：外部脚步阈值.5 TP0/FP37/FN2601；真脚步模长中位.2379/p95 .3595，无脚步中位.0220/p95 .2813。诊断改.2 recall67.28%但precision19.79%（TP1750/FP7092），未改正式门限。训练脚步正例27428/703094=3.901%，车21.5%/枪8.85%；方向目标正常，presence模长MSE未平衡正负，是否主因待对照。报告/快照 outputs/reports/footstep_diagnosis_v6_20260908.md 及同名目录；未用测试或停训。
- ADPIT spatial_balanced 权重activity1/direction2/distance.2/self.5，方向1−cos按可定位帧平衡；100ms多源class-aware localized F1采用20°角门限/15°去重，按验证macro选 best_localization.pt，另存best.pt/best_detection.pt。非官方DCASE完整分数。MAE只统计检出可定位同类单源帧（可混其他类）；无帧null且W&B可能保留旧点，需配检测/recall/样本数。

## 当前数据与历史保留
- datasets/spatial-v6-24h-20260908：720×120s=24h，train/val/test600/60/60=20/2/2h，seed2026090800，19.312GiB；registry/spatial_v6_24h_20260908冻结配置/清单/代码；报告 outputs/reports/spatial_v6_release_20260908.md。
- 49枪声组、8完整启停车型、5地面、682源；排除仅怠速Mirado和RocketLancher。known_assets_new_scenes：同原声可跨分区，场景/种子/轨迹独立，非独立录音或未见枪型泛化。训练49组/90枪型角色对，val47/77、test47/82均在训练实际可听；全部外部类四象限覆盖。缺角色素材不伪造。
- v6 RMS脚步.025/枪.12、master.35；走/跑/冲刺35/45/50合成单位末段平滑静音，未实测米数。外部远70%，其余近过10/中过10/靠近5/远离5；枪远70%，100–180，近8–60。枪/车/脚步权重1.2/1.2/.6。物理近似空气吸收129tap（64sample共同延迟），低油门截止6kHz；KEMAR非游戏原生HRTF，实录标定待做。
- 自动枪单点/短扫/持续20/30/50%，短3–6发/长8–24且受弹匣限制；DMR含MK14/VSS及栓狙逐发，M16/Mk47单发40%/3或2连发60%。枪型/模式/发序/弹量写配方，射速/弹量为近似非补丁精确参数，无假换弹/无限扫射。
- 原10ms详细标签、32ms功率窗口；track_localization_observable为可听且方向有效且SIR≥−10dB。100ms按各源多数可定位投影；完整功率窗跨5秒边界则监督屏蔽。仅难定位不删分类正例。
- 主集全音频/标签/几何/投影及3精确重放/覆盖通过；清理后2880新文件、5旧模型hash不变。24min预检试听127.0.0.1:8770，75测试/39裁剪对照/GPU及CPU流程通过，非真实精度结论。
- 历史F034 outputs/train-light-v5_3-gunvehicle-aug400-20260907 已400轮/28800更新，W&B e6f7b62b7263 finished。检测339轮test外部脚步/车/枪F1 .8851/.9310/.5591；loss229轮 .8594/.909/.68；末轮val枪.7827不是同指标。旧250/30轮模型/报告/导出保留。旧集按整枪型留出（train AK47/M762/MP5K/AWM，val M16A4，test Kar98k），不与v6当同条件对比。
- 旧方向诊断车774帧MAE67.40°，40.6%前后混淆样式；近<100的46帧10.15°，远728帧71.02°，有车型场景混杂，最差固定视角，不能仅归因转头/短窗。详见 outputs/reports/direction_diagnosis_20260908.md；诊断不用于测试选模。
- 旧25h已先删，元数据v4_retired_generation_metadata_20260908.zip，净15.026GiB；旧2h/3.6h（包含关系）、对应缓存、scenario-small/smoke共6目录也归档删除，registry/v5_retired_training_metadata_20260908.zip逐成员校验，净12.375GiB。凭据 outputs/reports/v6_replacement_cleanup_20260908.json；保留原声/全部旧模型/试听，不清回收站、不重复计释放量。

## 数据不变量与未完能力
- 本机客户端只读提取/复制，不下载替代/不读游戏进程；AIAutoAim记忆仅历史。先轨迹状态再声音，同角色固定地面，不能混地面充左右脚；步态连续可走停。
- self按直接来源而非距离，原点方向无效；本人拉环/准备self，离手爆炸/燃烧不标投掷者。车内遮蔽外脚步/远车，旧候选接纳.1/.05，隐藏正例mask0；范围外静音无正例。
- 轻量版仅站立walk/run/sprint、车/枪+self/少量碰撞，无天气/飞机/身体动作/投掷/C4/损伤链。完整试听 outputs/body-action-preview-20260907-v5_3-final/试听与轨迹.html（8769，10段480s/815源），未做素材隔离，不混训练，人工反馈未填。
- 完整版前向投掷绑定出手yaw；自然视角90%，小晃/观察/大转60/18/22%，车加速2.5/巡航.6/减速1/刹车.5/滑车.2。只有水平转头，无玩家平移/俯仰。
- 未做实录标定、实时采集/跨音量声卡矩阵、完整RTPC/REV/遮挡混响多普勒、因果/长上下文；WASAPI默认pre-volume仍受上游游戏/会话增益/DSP影响。31类未全接入，暴风雪无唯一解码，C4节拍/燃烧10s待标定。
