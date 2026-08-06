# YOLO26s-SPGM 实现、训练与评估说明

本文档记录 `feature/yolo26-spgm` 分支中 YOLO26s-SPGM 的代码结构、训练入口、权重迁移、统一测试流程、已有实验结果和当前限制。目标是让后续实验能够复现当前版本，并避免将修复前后的结果混为一谈。

## 1. 版本和运行环境

| 项目 | 当前配置 |
| --- | --- |
| 官方源码基线 | Ultralytics 8.4.104，`main=d8f2cad2ca798701875c5ef91fd5a6a4189781ca` |
| SPGM 分支 | `feature/yolo26-spgm` |
| 服务器 | `Server102_4090` |
| Conda 环境 | `GP02` |
| Python | 3.12.12 |
| PyTorch / CUDA | 2.5.1+cu124 / CUDA 12.4 |
| GPU | NVIDIA GeForce RTX 4090 24 GB |
| 数据配置 | `/home/liumengdong/xProjects/GP01/yolo26/dataset/pigData2025.yaml` |
| 数据根目录 | `/home/liumengdong/xData/pigdata2025_all` |

数据集保持在仓库外：训练集 2,431 张、验证集 250 张、测试集 250 张。测试集包含 5,436 个目标，平均每张约 21.7 个目标，明显比验证集更密集。

## 2. 仓库中的实验文件

本实验直接相关的文件如下：

- `ultralytics/nn/modules/scene_prior_guided.py`：SPGM、弱前景 mask、辅助损失、运行时诊断和旧版 DCT 兼容实现。
- `ultralytics/nn/modules/__init__.py`：导出 `ScenePriorGuidedModule`。
- `ultralytics/nn/tasks.py`：模型解析器接入 SPGM，并在检测损失中加入辅助 prior loss。
- `ultralytics/cfg/models/26/yolo26n-SPGM.yaml`：YOLO26n-SPGM 三尺度模型。
- `ultralytics/cfg/models/26/yolo26s-SPGM.yaml`：YOLO26s-SPGM 三尺度模型。
- `x_train_yolo26n_spgm_aux_trend_4090.py`：YOLO26n-SPGM 训练和 prior 趋势记录入口。
- `x_train_yolo26s_spgm_optimized_4090.py`：YOLO26s-SPGM 正式 300 轮训练入口。
- `x_eval_yolo26s_test_coco.py`：YOLO26s baseline/SPGM 统一 test split 与 COCOeval 入口。
- `docs/SPGM_MODEL_MIGRATION.md`：模型迁移阶段和历史实施约定。

数据、预训练权重、训练输出、TensorBoard 文件和预测 JSON 不提交到 Git。

## 3. SPGM 结构

YOLO26s-SPGM 在 P5、P4、P3 三个骨干特征进入检测头融合之前各插入一个 `ScenePriorGuidedModule`。模块不改变输入输出通道数，因此检测头仍接收 `[128, 256, 512]` 三个尺度。

每个 SPGM 的默认路径为：

1. 根据特征图尺寸生成二维坐标编码，范围为 `[-1, 1]`。
2. 将坐标编码和输入特征拼接。
3. 通过 `1×1 Conv -> GroupNorm -> SiLU -> DW 3×3 Conv -> GroupNorm -> SiLU -> 1×1 Conv` 预测单通道 `prior_logits`。
4. 使用 Sigmoid 得到前景 prior mask。
5. 对原始特征执行前景增强和背景抑制。

默认双向调制公式为：

```text
Y = X * (1 + a_fg * M - a_bg * (1 - M))
```

其中：

- `M` 是可学习的前景 prior mask；
- `a_fg` 默认从 0.10 开始，最大限制为 0.30；
- `a_bg` 默认从 0.05 开始，最大限制为 0.20；
- 两个尺度通过 Sigmoid 参数化，始终保持非负和有界。

三尺度 SPGM 对 YOLO26s 增加约 102,412 个单类别模型参数，约为基线的 1.03%。单类别 fine-tuned checkpoint 的参数量分别为：

- YOLO26s baseline：9,948,638；
- YOLO26s-SPGM：10,051,050。

## 4. 辅助 prior loss

训练时，SPGM 将带计算图的 `prior_logits` 临时写入文件级运行时缓存。这样可以让检测模型在完成前向传播后统一计算辅助损失，同时避免把带计算图的 Tensor 长期保存到 `nn.Module` 属性中，防止 EMA 或 checkpoint 的 `deepcopy` 失败。

弱监督目标由 YOLO 格式的 GT boxes 生成矩形前景 mask。当前实现采用向量化二维差分图合并同一图像中的所有矩形，避免旧版逐框 `.item()` 引发大量 GPU 到 CPU 同步。

每个尺度的辅助损失为：

```text
L_prior_scale = BCEWithLogits(prior_logits, target) + Dice(sigmoid(prior_logits), target)
```

总辅助损失为加权尺度平均后乘以 `lambda_prior=0.05`：

```text
scale_weights = {P3: 0.5, P4: 1.0, P5: 1.0}
L_prior = 0.05 * weighted_mean(L_prior_scale)
```

最后将 `L_prior` 加入检测损失并参与反向传播。检测 loss 的原有日志字段保持不变，详细 prior 指标由训练脚本写入 `spgm_prior_trend/` 下的 CSV。

## 5. 960 输入尺度修复

旧实现通过特征图绝对尺寸识别尺度：

```text
80×80 -> P3
40×40 -> P4
20×20 -> P5
```

该写法只适用于 640 输入。正式 YOLO26s-SPGM 训练使用 960 输入，真实 prior 尺寸为 `120×120、60×60、30×30`，旧代码会把它们记录为普通尺寸字符串，并通过 `scale_weights.get(name, 1.0)` 静默退化为三个尺度权重均为 1.0。因此，已经完成的首次 960 训练没有实际应用 `P3=0.5`。

当前版本改为根据输入图像与 prior 特征的步长动态推断：

```text
960 / 120 = 8  -> P3
960 / 60  = 16 -> P4
960 / 30  = 32 -> P5
```

同一逻辑同时支持 640、960 和矩形输入。若高度/宽度步长不一致、步长不是2的幂，或 `scale_weights` 缺少推断出的尺度，代码会直接报错，不再静默使用默认权重。

已完成的修复验证包括：

- 640、960、矩形输入共 9 组尺度推断；
- 非法步长拒绝；
- 缺失尺度权重拒绝；
- 合成 auxiliary loss 前向和反向传播；
- 真实 `yolo26s-SPGM.yaml` 在 960 输入下完整前向、损失和反向传播。

真实模型验证得到：

```text
P5: 30×30,  weight=1.0
P4: 60×60,  weight=1.0
P3: 120×120, weight=0.5
```

重要：当前保存的 YOLO26s-SPGM 正式训练结果来自尺度修复前。修复后的精度必须通过重新训练验证，不能把旧 checkpoint 当作修复后的结果。

## 6. COCO 预训练权重迁移

SPGM 在检测头中增加了三个层，导致官方 YOLO26s checkpoint 与 SPGM 模型的层编号不再完全一致。训练脚本使用显式 `COCO_LAYER_MAP` 将官方层映射到 SPGM 层，并验证兼容状态项数量为 708。

流程为：

1. 从 SPGM YAML 构建目标模型并保存随机初始化状态。
2. 加载官方 COCO YOLO26s checkpoint。
3. 读取 EMA 或 model state dict。
4. 按层映射和 Tensor shape 选择兼容权重。
5. 恢复 SPGM 模型结构并加载映射后的官方权重。
6. 保留新增 SPGM 层和不兼容检测层的初始化参数。

训练脚本同时检查模型尺度、参数量、SPGM 数量、调制方式和检测头通道，防止错误模型被静默训练。

## 7. 正式训练配置

YOLO26s baseline 与 SPGM 使用相同的主要训练参数：

| 参数 | 设置 |
| --- | --- |
| epochs | 300 |
| batch / nbs | 16 / 16 |
| imgsz | 960 |
| optimizer | SGD |
| lr0 / lrf | 0.0025 / 0.01 |
| momentum | 0.937 |
| weight_decay | 0.0005 |
| warmup_epochs | 3 |
| box / cls / dfl | 7.5 / 0.5 / 1.5 |
| mosaic / close_mosaic | 1.0 / 20 |
| translate / scale / fliplr | 0.1 / 0.5 / 0.5 |
| AMP | 开启 |
| seed / deterministic | 0 / True |
| early stopping | 关闭，完整训练 300 轮 |

正式 SPGM 训练命令：

```bash
source /home/liumengdong/miniconda3/etc/profile.d/conda.sh
conda activate GP02
cd /home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo26
python -u x_train_yolo26s_spgm_optimized_4090.py
```

训练脚本会拒绝源码目录存在未提交修改、权重缺失、输出目录已存在或模型结构不符合预期的情况。

## 8. 独立测试与统一 COCOeval

`x_eval_yolo26s_test_coco.py` 通过 `--variant baseline|spgm` 分别锁定官方 baseline 和 SPGM 源码。两个 variant 必须在不同 Python 进程中运行，避免同一进程的模块缓存污染源码来源。

统一测试设置：

```text
split=test
imgsz=960
batch=16
iou=0.7
max_det=300
save_json=True
```

脚本执行以下步骤：

1. 校验源码路径、Git状态、权重哈希、输出目录和模型结构。
2. 使用 Ultralytics 在独立 test split 上推理并导出 `predictions.json`。
3. 将 YOLO test labels 确定性转换为 COCO GT JSON。
4. 将预测中的文件名映射为固定整数 image ID。
5. 使用 `pycocotools.COCOeval` 计算标准12项 bbox指标。
6. 保存源码commit、权重哈希、GT/预测哈希和指标汇总。

运行命令：

```bash
cd /home/liumengdong/xProjects/GP02/ultralytics84104-spgm-yolo26
python x_eval_yolo26s_test_coco.py --variant baseline
python x_eval_yolo26s_test_coco.py --variant spgm
```

输出包括：

- `predictions.json`：Ultralytics原始预测；
- `predictions_coco.json`：标准整数image ID预测；
- `test_ground_truth_coco.json`：由YOLO标签转换的GT；
- `cocoeval_summary.json`：复现信息和完整指标。

## 9. 已有结果

两次测试使用相同的250张图片、5,436个标注和相同GT哈希：

```text
8a9277adc877a180ee4c9fcb4497c84be795a24ac76bad41d62bc9707fccf5d1
```

| 指标 | YOLO26s baseline | YOLO26s-SPGM（修复前训练） | 差值 |
| --- | ---: | ---: | ---: |
| 最佳验证 mAP50-95 | 0.88845（epoch 82） | 0.88987（epoch 106） | +0.00142 |
| test COCO AP50-95 | 0.80475 | 0.80616 | +0.00140 |
| AP50 | 0.97943 | 0.97975 | +0.00032 |
| AP75 | 0.91731 | 0.91206 | -0.00525 |
| AP-small | 0.40266 | 0.41504 | +0.01238 |
| AP-medium | 0.77590 | 0.77506 | -0.00084 |
| AP-large | 0.83729 | 0.84133 | +0.00405 |
| AR@100 | 0.84877 | 0.84978 | +0.00101 |
| 预测数量 | 11,132 | 10,738 | -3.54% |
| 训练时间 | 7,319.20 秒 | 7,951.79 秒 | +8.64% |

当前结果表明 SPGM 倾向于减少预测数量、提高 Precision，但同时降低 Recall；总体 AP 提升只有约 0.14 个百分点。单随机种子不足以证明这一差异稳定，且该结果没有使用修复后的 P3 权重。

## 10. 当前限制和下一步

1. 用当前尺度修复版本重新训练 YOLO26s-SPGM，生成新的 run name，不能覆盖旧结果。
2. 至少运行 3 个随机种子，报告 mean ± std；只使用验证集做模型选择和调参。
3. 关闭非必要的 prior mask、feature delta 逐批统计，单独测量纯 SPGM 前向开销。
4. 分别消融特征调制、辅助监督以及两者组合，确定真实收益来源。
5. 小目标测试标注只有 68 个，AP-small 变化只能作为趋势，不应独立形成结论。
6. 当前训练和测试入口包含服务器绝对路径；迁移到其他机器时必须同步修改路径常量。

旧权重和旧测试结果应保留作为“修复前”基线。任何修复后实验都必须使用新的输出目录、记录源码commit和权重哈希，并继续使用相同的 test GT 和 COCOeval 口径。
