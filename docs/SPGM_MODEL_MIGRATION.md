# SPGM 多模型迁移指南

本文档用于在新的 Codex 对话中，把当前 YOLO26-SPGM 参考实现迁移到另一个 Ultralytics 模型。每个对话只处理一个明确的目标模型。

## 参考实现

- 服务器：`Server102_4090`
- 参考 worktree：`/home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo26`
- 参考分支：`feature/yolo26-spgm`
- 参考模型：`ultralytics/cfg/models/26/yolo26n-SPGM.yaml`
- SPGM 实现：`ultralytics/nn/modules/scene_prior_guided.py`
- 模块导出：`ultralytics/nn/modules/__init__.py`
- 模型解析和辅助损失：`ultralytics/nn/tasks.py`
- 训练脚本：`x_train_yolo26n_spgm_aux_trend_4090.py`

当前参考实现使用 P3/P4/P5 三个 `ScenePriorGuidedModule`，输入尺寸 640 时对应约 80x80、40x40、20x20。辅助监督使用 BCE + Dice prior loss，`lambda_prior=0.05`，P3/P4/P5 权重为 0.5/1.0/1.0。

## 分支和 worktree

每个目标模型使用独立分支和 worktree：

- 分支：`feature/spgm-<model>`
- worktree：`/home/liumengdong/xProjects/GP02/ultralytics84104-spgm-<model>`

新分支从已提交的 YOLO26-SPGM 参考提交创建，以复用 SPGM 实现、模块注册、模型解析和辅助损失。不要在正在执行正式训练的 worktree 中开发其他模型。创建分支前先检查参考 worktree 的未提交修改；需要继承的内容必须先得到用户确认并提交。

每个新 worktree 都要有项目级 `AGENTS.md`，至少写明目标模型、分支、目录、GP02 环境、测试命令、训练脚本、tmux 规则和禁止操作。未经确认不 push。

## 新对话首条消息模板

复制下面内容，并替换 `<目标模型>` 和 `<目标YAML>`：

```text
连接服务器 Server102_4090。

我要进行 SPGM 对比实验，把现有 YOLO26-SPGM 实现迁移到一个新模型。这个对话只处理下面这一个模型。

参考实现：
- 仓库：/home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo26
- 迁移指南：
  /home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo26/docs/SPGM_MODEL_MIGRATION.md
- 参考 YAML：
  /home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo26/ultralytics/cfg/models/26/yolo26n-SPGM.yaml
- SPGM 实现：
  ultralytics/nn/modules/scene_prior_guided.py
- 模块注册和辅助损失：
  ultralytics/nn/modules/__init__.py
  ultralytics/nn/tasks.py
- 参考训练脚本：
  x_train_yolo26n_spgm_aux_trend_4090.py

本次目标：
- 模型名称：<目标模型>
- 原始 YAML：<目标YAML>
- 任务类型：目标检测
- 创建对应的 SPGM 模型配置和训练脚本。
- 保留官方原始 YAML，不能直接修改它。
- SPGM 核心实现、lambda_prior=0.05、BCE+Dice 和当前调制方式默认保持不变。
- 根据目标模型真实的 backbone/neck 连接关系确定 P3/P4/P5 插入点，不能照抄 YOLO26 层号。
- SPGM 输出通道必须与被替换的原始特征连接兼容。
- 更新所有受新增层影响的 from、Concat 和 Detect 索引。
- 不修改无关代码，不覆盖已有实验结果。

Git：
- 创建独立分支 feature/spgm-<目标模型>。
- 使用独立 worktree：
  /home/liumengdong/xProjects/GP02/ultralytics84104-spgm-<目标模型>
- 不修改或停止当前正在训练的任务。
- 未经确认不 push。

环境和运行：
- 使用 GP02 环境和 RTX 4090。
- 训练及训练相关测试尽量在 tmux 中运行。
- tmux 名称使用 xt01～xt04 中最小的空闲名称，同时不超过 4 个。
- 不关闭来源不明或正在运行的 tmux 会话。

请先：
1. 完整阅读参考仓库 AGENTS.md 和迁移指南。
2. 检查 Git、worktree、tmux、GPU 和目标 YAML。
3. 对比原模型与 YOLO26-SPGM 的特征流。
4. 明确建议的 SPGM 插入位置、尺度、通道及所有层索引变化。
5. 判断现有辅助损失是否能直接适用。
6. 只给出实施计划，暂时不要修改文件。
```

确认结构方案后发送：

```text
确认按方案实施。先创建独立分支和 worktree，再修改。完成静态检查和 GPU smoke test，但不要启动正式长时训练，也不要 push。
```

## 实现约束

- 新增独立的 `<model>-SPGM.yaml`，不得修改官方基线 YAML。
- 根据目标模型特征流选择插入点，不能机械复制 YOLO26 的层号。
- SPGM 输出通道必须与原连接兼容，并更新新增层影响的全部索引。
- 首轮迁移保持 SPGM 实现、调制方式、辅助损失和超参数不变。
- 新建独立训练脚本和运行名，不覆盖其他实验。
- 基线与 SPGM 实验固定数据集、输入尺寸、seed、batch、优化器、学习率和增强参数。

## 验收标准

对标准 P3/P4/P5 检测模型至少验证：

1. 原始 YAML 和 SPGM YAML 均能构建。
2. SPGM 模型恰好包含 3 个 `ScenePriorGuidedModule`。
3. 640 输入下 P3/P4/P5 形状约为 80x80、40x40、20x20。
4. Detect 输入尺度、通道和层索引正确。
5. auxiliary prior loss 进入总损失。
6. 每个 SPGM 参数均获得有限、非零梯度。
7. 先完成结构、前向、反向和 1 epoch smoke test，再申请正式长时训练。

对比实验记录参数量、GFLOPs、峰值显存、训练耗时、Precision、Recall、mAP50、mAP50-95 和 SPGM prior 趋势。

## 特殊模型

- YOLO11、YOLOv8 等标准三尺度检测模型可优先迁移。
- P2/P6 模型不能默认使用 3 个 SPGM，必须先确定监督尺度和权重。
- 分类模型没有检测框，当前 prior 辅助监督不能直接使用。
- 分割、姿态、OBB、YOLOE 和 RT-DETR 必须先检查 batch 标注格式、损失入口和特征拓扑。
- 遇到结构不兼容时先报告，不自行改变 SPGM 算法定义。
