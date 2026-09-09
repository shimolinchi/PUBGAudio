# v5.3 枪声与车辆扩充登记

实际数据位于本地 `datasets/light-v5_3-gunvehicle-20260907`，音频、标签及素材不上传。

保留父版全部60段，并追加48段训练场景；108段核验和4段精确重放通过。验证/测试与父版相同。

400轮训练、两类选模测试、权重导出及W&B核验均完成，元数据见 `training_400epochs/`；概率和协议见 `docs/light_augmentation_gunvehicle_v5_3.md`。
