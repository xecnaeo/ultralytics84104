# AGENTS.md

本文档适用于共享官方模型对照实验 worktree 及其子目录。该分支统一承载未添加 SPGM 的 Ultralytics 官方 baseline，只增加实验训练入口和运行规范，不执行 Ultralytics 官方 PR 工作流，也不再为每个模型创建单独 baseline 分支或 worktree。

## 主机、目录与环境

- SSH 别名：`Server102_4090`。
- 共享 worktree：`/home/liumengdong/xProjects/GP02/ultralytics84104-yolo26-baseline`。
- 共享分支：`experiment/yolo26-baseline`（保留既有名称用于历史结果溯源）。
- 官方源码基线：`main@d8f2cad2ca798701875c5ef91fd5a6a4189781ca`。
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

## 官方 Baseline 原则

- `ultralytics/`、官方模型 YAML、损失函数和模型解析器必须与 `main@d8f2cad` 完全一致。
- 每个官方模型只新增独立的根目录训练脚本，不创建模型专用 baseline 分支或 worktree。
- 模型通过带 scale 的官方别名选择具体尺寸，并同时校验统一官方 YAML：
  - YOLO26n：`models/26/yolo26n.yaml` → `models/26/yolo26.yaml`。
  - YOLO11n：`models/11/yolo11n.yaml` → `models/11/yolo11.yaml`。
  - YOLOv8s：`models/v8/yolov8s.yaml` → `models/v8/yolov8.yaml`。
- 不得加入 `ScenePriorGuidedModule`、SPGM 缓存、SPGM auxiliary prior loss 或 SPGM 趋势回调。
- 官方模型 YAML 保持不变。
- 当前训练入口：
  - `x_train_yolo26n_baseline_4090.py`
  - `x_train_yolo11n_baseline_4090.py`
  - `x_train_yolov8s_baseline_4090.py`
- 当前测试入口：
  - `x_eval_yolov8s_baseline_test_coco.py`
- 数据集统一使用 `/home/liumengdong/xProjects/GP01/yolo26/dataset/pigData2025.yaml`。
- baseline 与对应 SPGM 实验的有效训练参数必须一致；只允许 `model`、`name`、`save_dir` 和 SPGM 独有 prior callback 不同。
- 当前实验固定 `seed=0`、`deterministic=True`、`multi_scale=0.0`。
- `pretrained` 必须与对应 SPGM 实验逐组匹配：YOLO26n 当前使用本地 COCO `yolo26n.pt`，YOLO11n 和 YOLOv8s 当前训练入口使用 `pretrained=False`；不得跨越不同初始化设置直接比较。
- 每次运行记录官方源码提交、runner 提交、官方 YAML SHA256、源码导入路径、Python/PyTorch/CUDA/Ultralytics 版本和完整 `args.yaml`。
- 未经用户明确要求，不读取、比较或评价正式实验指标。

## 源码完整性门禁

正式或 smoke 训练前必须同时满足：

```bash
git diff --quiet main..HEAD -- ultralytics
git diff --quiet -- ultralytics
git status --short -- ultralytics
```

其中前两条退出码必须为 0，第三条不得输出任何内容。构建模型后确认使用目标 `n` scale、三尺度 Detect，且不存在 `ScenePriorGuidedModule`。

## Git 约定

- 只在当前共享 worktree 和 `experiment/yolo26-baseline` 分支工作，不修改 main 或 SPGM worktree。
- 保留用户已有改动，只修改当前任务涉及的 baseline runner 和本说明。
- 未经明确请求不 push，不 force push，不执行 `git reset --hard`。
- 不提交数据集、runs、权重、缓存、日志和临时文件。
- 提交前检查 `git status --short --branch`、`git diff --check` 和完整 diff。

## tmux 运行约定

- 所有训练脚本和训练相关测试尽量在 tmux 中运行。
- 会话名称使用 `xt01`、`xt02`、`xt03`、`xt04` 中最小的未占用名称，同时存在的会话不得超过 4 个。
- 不复用仍存在的会话名称，不终止正在训练、测试、监控或来源不明的会话。
- baseline 与 SPGM 不在同一张 RTX 4090 上并行训练。
- 使用绝对路径，通过 `tee` 保存独立日志，并在训练结束后输出退出码。

## 验证原则

- 修改 Python 后执行 `python -m py_compile` 和针对性导入检查。
- 运行前确认 `ultralytics.__file__` 来自当前共享 baseline worktree。
- 每个新 baseline 先完成模型构建、结构检查和独立 1 epoch smoke，再申请正式长时训练。
- 启动长时训练前检查 GPU、磁盘、进程、输出目录和 tmux 会话。
- 不覆盖已有 baseline、SPGM、smoke 或正式实验结果；除非用户要求，不持续监控。
