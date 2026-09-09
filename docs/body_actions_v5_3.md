# v5.3：减少普通行走，增加跑步与身体动作

2026-09-07，反馈 F029。用户明确减少的是普通行走，不是所有脚步角色的总生成率。实际输入是 `configs/generation_v5_3.json`；[完整概率表](generation_probability_v5_3.md)由配置生成。

## 概率与自然连续性

| 项目 | v5.2 | v5.3 |
|---|---:|---:|
| 普通行走权重 | 1 | 0.75 |
| 跑步权重 | 1 | 1.25 |
| 冲刺、蹲行、匍匐各自权重 | 1 | 1 |
| 每合法调度步脚步角色生成概率（低/中/高） | 0.8% / 1.8% / 3.5% | 保持 |

五种步态都可用时，行走选择概率20%→15%，跑步20%→25%。不同地面的可用步态可能不同，先过滤再归一化；这些不是音频时长或脚步次数比例。

站立步态停步后，若距上次动作结束至少7秒、剩余至少11秒，附加候选为：不追加动作62%、蹲下—保持—起身25%、趴下—保持—站起8%、翻越5%。不满足前提或缺素材时删除相应候选，重新归一化。选择不追加动作后再抽下一步态；若下一步态需要另一姿态，单独安排必要转换，不把它计入附加循环概率。

当前步态最短3秒后每0.5秒以12.5%概率结束，最长12秒；不会每一步随机走跑。转身、远近、车辆状态沿用v5.2；同一个人物始终保持原地面。明显停顿仍为50%概率、1–3秒；另等待上一脚步原声结束，避免身体动作开始时还在连续踩步。蹲/趴后保持2–3.5秒再起身，转换覆盖原声全长。翻越约3.6秒，速度先升后降，沿同一曲线跨过规划障碍；正常走跑脚步暂停，使用抓扶、攀越、脚接触和衣物原声。障碍材质与脚下地面分开记录，不用换地面模拟左右脚。

## 素材能证明什么

新池 `datasets/sources-v5_3-final` 共815个WAV，原776个完整保留，新增39个。新增池单独在 `datasets/sources-body-v5_3-final`；只读本机客户端提取，没有下载音效。

| 原声 | 数量 | 证据与限制 |
|---|---:|---|
| 共享转换衣物声 | 9 | CharacterBank事件1103459313 → 动作334636408 → 容器500554510 → 9个实际media ID。Vaulting_Rustle也引用这9条。按ID选择，排除同名不同ID版本 |
| 翻越抓扶、攀越接触、脚接触 | 各9 | CharacterBank_Vaulting内具名素材，Concrete/Metal/Wood各3个；不是蹲行/匍匐或另一种地面脚步 |
| 翻越长衣物声 | 3 | 同一翻越Bank内具名原声 |

`Prone_Ruslte`在此次CharacterBank中没有关联动作。**未证实蹲下、蹲起、趴下、站起各有独有音色**。本批用真正的共享转换衣物声配合合成姿态计划，`semantic_evidence`明确记录这个边界；不能把细姿态标签当成仅凭该原声就能区分的类别。具体游戏动画触发、混音增益和时序仍待实录核对。HIRC只读解析使用[wwiser](https://github.com/bnnm/wwiser)；本地可复核报告为 `outputs/reports/body_source_audit_20260907.json`。

共享转换事件关联衰减对象515586057，距离轴终点3000；厘米/缩放1假设下暂取30m。中间衰减继续使用试听近似。翻越各分层暂以该30m作代理，并明确标为待校准，未宣称它们的原生可听距离已确认。新共享衣物声目标RMS为0.035、翻越衣物0.04，固定总增益0.6；均不是实录标定值。

## 标签与试听

每10ms保留原有距离、世界/耳相对方位、玩家朝向和可观测性，新增 `track_actor_id`、`track_posture`、`track_body_action`、`track_semantic_evidence`。动作声轨通过`actor_parent`关联原人物，不另占角色名额，自身/外部标签随人物继承。自身仍绑定听者原点、方向无效；这里只模拟相对听者的自身动作，不模拟玩家世界平移。翻越另存障碍位置/材质与因果链。

动作声轨为`action`；默认脚步/车/枪训练投影将它们视作干扰，不伪装成脚步正例。姿态真值与声学活动分别保存，静音、远处或被遮蔽不强造正例。

最终试听：`outputs/body-action-preview-20260907-v5_3-final/试听与轨迹.html`，10段、共8分钟。C01/C02为外部/自身蹲起，P01/P02为外部/自身趴起，V01/V02为外部/自身翻越，各40秒；R01–R04为独立60秒随机场景。专项显式提高指定动作，不能用于验证自然抽样占比。实际随机密度为3高1低，少量样本不强制配额。

旧25小时、v5/v5.1/v5.2文件与反馈保留。本批为试听，未划分训练/验证/测试，没有开始新训练。无`-final`后缀的v5.3目录是同名素材筛选核对前的中间产物，不能用于训练或作为本次交付。

## 复现入口

在子模块目录、已有本机原声池与HRIR条件下：

```powershell
python tools/synthesis/generate_body_preview.py --output outputs/new-body-audition
python tools/synthesis/verify_causal_preview.py outputs/new-body-audition --sources datasets/sources-v5_3-final --replay C01 P02 V01 R02
python tools/synthesis/build_causal_preview.py outputs/new-body-audition
python tools/synthesis/serve_preview.py outputs/new-body-audition --port 8769
python tools/synthesis/write_probability_reference.py --config configs/generation_v5_3.json --output docs/generation_probability_v5_3.md
```

现有目录会被拒绝覆盖。来源准备为`prepare_body_sources.py`和`merge_source_pools.py`；审计为`audit_body_actions.py --bank-xml <wwiser输出的CharacterBank XML>`。核验文件和试听反馈随每个输出目录独立保留。
