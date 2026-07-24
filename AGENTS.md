# AGENTS.md

本文档适用于当前 SPGM 功能 worktree 及其子目录。该仓库是基于 Ultralytics 8.4.104 的个人研究项目，用于在 Server102_4090 上集成和验证 YOLO26-SPGM，不执行 Ultralytics 官方 PR 工作流。

## 主机、目录与环境

- SSH 别名：`Server102_4090`。
- Git 主分支 worktree：`/home/liumengdong/xProjects/GP02/ultralytics84104`，默认只读。
- SPGM 功能 worktree：`/home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo26`。
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
- 模型配置：`ultralytics/cfg/models/26/yolo26n-SPGM.yaml`。
- 默认输入尺寸：640。
- P3/P4/P5 各包含一个 `ScenePriorGuidedModule`，对应 80x80、40x40、20x20。
- 辅助监督采用 BCE + Dice prior loss，默认 `lambda_prior=0.05`，尺度权重为 P3/P4/P5 = 0.5/1.0/1.0。
- `torch-dct` 只用于默认关闭的 legacy DCT 分支；不要为默认 SPGM 路径额外安装它。

结构修改后至少验证：

1. YAML 能够构建模型。
2. 恰好存在 3 个 `ScenePriorGuidedModule`。
3. 640 输入下三尺度前向形状正确。
4. auxiliary prior loss 进入总损失。
5. SPGM 参数能够获得有限、非零梯度。
6. 原始 YOLO26 配置仍能构建。

## Git 约定

- 日常开发只在 `feature/yolo26-spgm` worktree 中进行；`main` worktree 默认只读。
- 保留用户已有改动，只修改当前任务涉及的文件。
- 未经明确请求不 push，不 force push，不执行 `git reset --hard`。
- 不提交数据集、runs、权重、缓存、日志和临时文件。
- 提交前检查 `git status --short --branch`、`git diff --check` 和完整 diff。
- 当前远端 URL 为 `git@github.com:xecnaeo/ultralytics84104.git`；Server102_4090 尚未配置 GitHub 主机信任和仓库认证，未经用户确认不修改相关密钥或推送。

## 验证原则

- 修改保持简单、局部、可验证，不顺手重构或格式化无关代码。
- 修改 Python 后至少执行 `python -m py_compile` 和针对性导入检查。
- 修改模型或损失后先做单卡结构、前向与梯度 smoke test，再考虑长时训练。
- 长时训练前检查 `nvidia-smi`、磁盘、进程和 tmux 会话；不终止来源不明的任务。
- 遇到 OOM、CUDA、数据或损失异常时保留完整输出，不静默修改 batch、输入尺寸或损失权重。

常用检查：

```bash
git -C /home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo26 status --short --branch
nvidia-smi
/home/liumengdong/miniconda3/bin/conda run --no-capture-output -n GP02 python -c "import torch, ultralytics; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(ultralytics.__version__, ultralytics.__file__)"
```