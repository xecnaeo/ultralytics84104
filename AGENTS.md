# AGENTS.md

本文档适用于 YOLO26n 原始模型对照实验 worktree 及其子目录。该分支只用于在 Server102_4090 上复现未添加 SPGM 的官方 YOLO26n，不执行 Ultralytics 官方 PR 工作流。

## 主机、目录与环境

- SSH 别名：`Server102_4090`。
- worktree：`/home/liumengdong/xProjects/GP02/ultralytics84104-yolo26-baseline`。
- 分支：`experiment/yolo26-baseline`。
- 基线提交：`main@d8f2cad`。
- Conda：`/home/liumengdong/miniconda3`。
- 固定环境：`GP02`。
- Python 3.12.12、PyTorch 2.5.1+cu124、Ultralytics 8.4.104、单张 NVIDIA GeForce RTX 4090。

交互终端使用：

```bash
source /home/liumengdong/miniconda3/etc/profile.d/conda.sh
conda activate GP02
```

自动化命令使用：

```bash
/home/liumengdong/miniconda3/bin/conda run --no-capture-output -n GP02 python ...
```

不得向 Conda base、系统 Python 或其他环境安装依赖。未经用户确认，不删除、重建或清理 Conda 环境。

## 对照实验约束

- 模型必须使用官方 `ultralytics/cfg/models/26/yolo26.yaml`，通过 `yolo26n.yaml` 别名选择 `n` scale。
- 不得加入 `ScenePriorGuidedModule`、SPGM 缓存、SPGM auxiliary prior loss 或 SPGM 趋势回调。
- 官方模型 YAML 保持不变。
- 训练入口：`x_train_yolo26n_baseline_4090.py`。
- 数据集：`/home/liumengdong/xProjects/GP01/yolo26/dataset/pigData2025.yaml`。
- 输出目录：`/home/liumengdong/xProjects/GP02/yolo26/runs/train/yolo26n_baseline_4090_100e`。
- 对照参数必须与 `yolo26n_spgm_stage6_aux_trend_4090_100e/args.yaml` 一致；仅允许 `model`、`name` 和 `save_dir` 因实验组不同而变化。
- 未经用户明确要求，不读取、比较或评价实验指标。

## Git 约定

- 只在当前 worktree 和 `experiment/yolo26-baseline` 分支工作，不修改主 worktree 或 SPGM worktree。
- 保留用户已有改动，只修改当前任务涉及的文件。
- 未经明确请求不 push，不 force push，不执行 `git reset --hard`。
- 不提交数据集、runs、权重、缓存、日志和临时文件。
- 提交前检查 `git status --short --branch`、`git diff --check` 和完整 diff。

## tmux 运行约定

- 所有训练脚本和训练相关测试尽量在 tmux 中运行。
- 会话名称使用 `xt01`、`xt02`、`xt03`、`xt04` 中最小的未占用名称，同时存在的会话不得超过 4 个。
- 不复用仍存在的会话名称，不终止正在训练、测试、监控或来源不明的会话。
- 使用绝对路径，通过 `tee` 保存独立日志，并在训练结束后输出 `TRAIN_EXIT_CODE`。

## 验证原则

- 修改 Python 后执行 `python -m py_compile` 和针对性导入检查。
- 正式训练前确认导入的 `ultralytics` 来自当前基线 worktree。
- 构建模型后确认使用 `n` scale、三尺度 Detect，且不存在 `ScenePriorGuidedModule`。
- 启动长时训练前检查 GPU、磁盘、进程、输出目录和 tmux 会话。
- 启动后仅进行一次启动验收；除非用户要求，不持续监控。
