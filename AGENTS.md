# AGENTS.md

本文档适用于当前 SPGM 功能 worktree 及其子目录。该仓库是基于 Ultralytics 8.4.104 的个人研究项目，用于在 Server102_4090 上集成和验证 YOLO11n-SPGM，不执行 Ultralytics 官方 PR 工作流。

## 主机、目录与环境

- SSH 别名：`Server102_4090`。
- Git 主分支 worktree：`/home/liumengdong/xProjects/GP02/ultralytics84104`，默认只读。
- SPGM 功能 worktree：`/home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo11`。
- Conda：`/home/liumengdong/miniconda3`。
- 固定环境：`GP02`。
- 当前基线：Python 3.12.12、PyTorch 2.5.1+cu124、Ultralytics 8.4.104、单张 NVIDIA GeForce RTX 4090。

交互终端使用：

```bash
source /home/liumengdong/miniconda3/etc/profile.d/conda.sh
conda activate GP02
```

自动化命令使用：

```bash
/home/liumengdong/miniconda3/bin/conda run --no-capture-output -n GP02 python ...
```

执行 Python、安装依赖或测试前，先确认解释器和 editable 源码路径：

```bash
python -c "import sys, torch, ultralytics; print(sys.executable); print(torch.__version__, torch.version.cuda); print(ultralytics.__version__, ultralytics.__file__)"
python -m pip --version
```

依赖只能安装到 GP02，统一使用 `python -m pip install ...`。不得向 Conda base、系统 Python 或其他环境安装依赖。未经用户明确确认，不删除、重建或清理 Conda 环境。

## SPGM 范围

- SPGM 实现：`ultralytics/nn/modules/scene_prior_guided.py`。
- 模块导出：`ultralytics/nn/modules/__init__.py`。
- 模型解析和辅助损失：`ultralytics/nn/tasks.py`。
- 模型配置：`ultralytics/cfg/models/11/yolo11n-SPGM.yaml`。
- 训练脚本：`x_train_yolo11n_spgm_aux_trend_4090.py`。
- 默认输入尺寸：640。
- P3/P4/P5 各包含一个 `ScenePriorGuidedModule`，对应 80x80、40x40、20x20。
- 辅助监督采用 BCE + Dice prior loss，默认 `lambda_prior=0.05`，尺度权重为 P3/P4/P5 = 0.5/1.0/1.0。
- `torch-dct` 只用于默认关闭的 legacy DCT 分支；不要为默认 SPGM 路径额外安装它。

结构修改后至少验证：

1. YAML 能够构建模型。
2. 恰好存在 3 个 `ScenePriorGuidedModule`。
3. 640 输入下三尺度前向形状正确。
4. auxiliary prior loss 进入总损失。
5. 当前 fg_bg 调制路径中的 SPGM 参数能够获得有限、非零梯度；不把仅供 foreground 模式使用的 alpha 计入。
6. 原始 YOLO11 配置仍能构建。

## 多模型对比实验

- 多模型迁移规范和新对话模板见 `docs/SPGM_MODEL_MIGRATION.md`。
- 每个新模型使用独立分支 `feature/spgm-<model>` 和独立 worktree `/home/liumengdong/xProjects/GP02/ultralytics84104-spgm-<model>`。
- 新分支从已提交的 YOLO26-SPGM 参考提交创建；创建前先检查并处理参考 worktree 的未提交修改。
- 不在正在执行正式训练的 worktree 中开发其他模型，不修改或停止已有训练任务。
- 每次新对话只处理一个明确的目标 YAML；先比较目标模型特征流并形成计划，确认后再实施。
- 保留官方原始 YAML，新增独立的 `<model>-SPGM.yaml` 和对应训练脚本。
- 不照抄 YOLO26 的层号；根据目标模型真实的 P3/P4/P5 特征流更新所有 `from`、`Concat` 和检测头索引。
- 首轮迁移保持 SPGM 核心实现、`lambda_prior=0.05`、BCE+Dice 和调制方式不变。
- 基线与 SPGM 实验固定数据集、输入尺寸、seed、batch、优化器、学习率和增强参数。
- P2/P6、分类、分割、姿态、OBB、YOLOE 或 RT-DETR 必须先检查监督尺度、标注格式、损失入口和特征拓扑，不默认直接复用三尺度方案。
- smoke test 通过后才能申请正式长时训练；未经确认不 push。

## Git 约定

- 日常开发只在 `feature/spgm-yolo11` 对应 worktree 中进行；`main` 和 YOLO26-SPGM worktree 默认只读。
- 保留用户已有改动，只修改当前任务涉及的文件。
- 未经明确请求不 push，不 force push，不执行 `git reset --hard`。
- 不提交数据集、runs、权重、缓存、日志和临时文件。
- 提交前检查 `git status --short --branch`、`git diff --check` 和完整 diff。
- 当前远端 URL 为 `git@github.com:xecnaeo/ultralytics84104.git`，已配置仓库专用 Deploy Key；未经用户明确请求不推送。

## tmux 运行约定

- 所有训练脚本和训练相关测试，包括 smoke test，尽量在 tmux 会话中运行。
- 会话名称按 `xt01`、`xt02`、`xt03`、`xt04` 递增，选择当前最小的未占用名称。
- 同时存在的 tmux 会话不得超过 4 个；创建前必须先执行 `tmux list-sessions` 并统计数量。
- 不复用仍存在的会话名称，不为腾出名额自动终止正在训练、测试、监控或来源不明的会话。
- 命令使用绝对路径，并通过 `tee` 保存独立日志；训练结束后检查退出状态、日志、权重、results.csv 和 SPGM CSV。

常用命令：

```bash
tmux list-sessions
tmux new-session -d -s xt01 '<command>'
tmux capture-pane -pt xt01 -S -100
tmux attach-session -t xt01
```

## 验证原则

- 修改保持简单、局部、可验证，不顺手重构或格式化无关代码。
- 修改 Python 后至少执行 `python -m py_compile` 和针对性导入检查。
- 修改模型或损失后先做单卡结构、前向与梯度 smoke test，再考虑长时训练。
- 长时训练前检查 `nvidia-smi`、磁盘、进程和 tmux 会话；不终止来源不明的任务。
- 遇到 OOM、CUDA、数据或损失异常时保留完整输出，不静默修改 batch、输入尺寸或损失权重。

常用检查：

```bash
git -C /home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo11 status --short --branch
nvidia-smi
/home/liumengdong/miniconda3/bin/conda run --no-capture-output -n GP02 python -c "import torch, ultralytics; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(ultralytics.__version__, ultralytics.__file__)"
```