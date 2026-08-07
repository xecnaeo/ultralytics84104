# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SPGM Runtime Cache
# ------------------------------------------------------------
# 说明：
# 1. 阶段 3 默认不使用该缓存。
# 2. 阶段 6 添加辅助监督时，可以启用 collect_aux_logits=True，
#    将带计算图的 prior_logits 临时放入这个全局运行时缓存。
# 3. 不要把带计算图的 Tensor 保存为 nn.Module 的普通属性，
#    否则 Ultralytics 创建 EMA 或保存模型时 deepcopy(model) 可能报错。
# ============================================================

_SPGM_AUX_CACHE = []


def clear_spgm_aux_cache():
    """清空 SPGM 辅助监督运行时缓存。阶段 6 修改 loss 时会用到。"""
    _SPGM_AUX_CACHE.clear()


def get_spgm_aux_cache():
    """获取 SPGM 辅助监督运行时缓存。阶段 6 修改 loss 时会用到。"""
    return list(_SPGM_AUX_CACHE)


def _append_spgm_aux_cache(item):
    """内部函数：向运行时缓存中加入一项。"""
    _SPGM_AUX_CACHE.append(item)


# ============================================================
# Stage 6: SPGM Auxiliary Prior Loss
# ------------------------------------------------------------
# 用于将 GT boxes 转换为弱前景 mask，并计算
# BCEWithLogits + Dice auxiliary prior loss。
# ============================================================


def build_batch_weak_foreground_masks_torch(
    batch_idx,
    bboxes,
    batch_size,
    out_hw,
    mode="binary",
    center_ratio=0.7,
    device=None,
    dtype=torch.float32,
):
    """
    根据 batch 中的 GT boxes 生成弱监督前景 mask。

    Args:
        batch_idx (Tensor): [N] 或 [N,1]，每个 bbox 属于 batch 中第几张图。
        bboxes (Tensor): [N,4]，YOLO normalized xywh，范围通常为 [0,1]。
        batch_size (int): 当前 batch size。
        out_hw (tuple): 输出 mask 尺寸，格式为 (H, W)。
        mode (str): binary / center。快速验证阶段建议 binary。
        center_ratio (float): center 模式下 bbox 收缩比例。
        device: 输出 tensor 所在设备。
        dtype: 输出 tensor 类型。

    Returns:
        masks (Tensor): [B,1,H,W]，弱监督前景 mask。
    """
    out_h, out_w = out_hw

    if device is None:
        device = bboxes.device

    masks = torch.zeros(
        (batch_size, 1, out_h, out_w),
        device=device,
        dtype=dtype,
    )

    if bboxes is None or bboxes.numel() == 0:
        return masks

    batch_idx = batch_idx.view(-1).long().to(device)
    bboxes = bboxes.to(device=device, dtype=dtype)

    if mode not in ("binary", "center"):
        raise ValueError(f"快速验证阶段暂只支持 binary / center，但得到 mode={mode}")
    if not 0.0 < float(center_ratio) <= 1.0:
        raise ValueError(f"center_ratio must be in (0, 1], but got {center_ratio}")

    valid = (batch_idx >= 0) & (batch_idx < batch_size)
    batch_idx = batch_idx[valid]
    bboxes = bboxes[valid]

    shrink = center_ratio if mode == "center" else 1.0
    centers = bboxes[:, :2]
    sizes = bboxes[:, 2:] * shrink

    x1 = torch.floor((centers[:, 0] - sizes[:, 0] / 2.0) * out_w).long().clamp(0, out_w - 1)
    y1 = torch.floor((centers[:, 1] - sizes[:, 1] / 2.0) * out_h).long().clamp(0, out_h - 1)
    x2 = torch.ceil((centers[:, 0] + sizes[:, 0] / 2.0) * out_w).long().clamp(0, out_w)
    y2 = torch.ceil((centers[:, 1] + sizes[:, 1] / 2.0) * out_h).long().clamp(0, out_h)

    # 保证至少占一个网格，与逐框切片赋值保持一致。
    x2 = torch.where(x2 <= x1, (x1 + 1).clamp(max=out_w), x2)
    y2 = torch.where(y2 <= y1, (y1 + 1).clamp(max=out_h), y2)

    # 用二维差分图一次性合并同一图像内的所有矩形，避免逐框 .item() 触发 GPU-CPU 同步。
    difference = torch.zeros((batch_size, out_h + 1, out_w + 1), device=device, dtype=torch.int32)
    ones = torch.ones_like(batch_idx, dtype=difference.dtype)
    difference.index_put_((batch_idx, y1, x1), ones, accumulate=True)
    difference.index_put_((batch_idx, y2, x1), -ones, accumulate=True)
    difference.index_put_((batch_idx, y1, x2), -ones, accumulate=True)
    difference.index_put_((batch_idx, y2, x2), ones, accumulate=True)
    covered = difference.cumsum(dim=1).cumsum(dim=2)[:, :out_h, :out_w] > 0
    masks[:, 0] = covered.to(dtype=dtype)

    return masks


def dice_loss_from_logits(logits, targets, eps=1e-6):
    """
    根据 logits 和 target mask 计算 Dice Loss。

    Args:
        logits: [B,1,H,W]
        targets: [B,1,H,W]

    Returns:
        dice_loss: scalar
    """
    probs = torch.sigmoid(logits)

    probs = probs.flatten(1)
    targets = targets.flatten(1)

    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def bce_dice_prior_loss(
    logits,
    targets,
    bce_weight=1.0,
    dice_weight=1.0,
):
    """
    BCEWithLogits + Dice prior loss。
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)

    loss = bce_weight * bce + dice_weight * dice

    return loss, {
        "bce": float(bce.detach()),
        "dice": float(dice.detach()),
        "loss": float(loss.detach()),
        "target_mean": float(targets.mean().detach()),
        "pred_mean": float(torch.sigmoid(logits).mean().detach()),
    }


def _infer_pyramid_scale_name(logits, imgs, aux_name=None):
    """Infer a pyramid level from input and feature strides, independent of image size."""
    if aux_name is not None:
        return str(aux_name)

    feature_h, feature_w = logits.shape[-2:]
    image_h, image_w = imgs.shape[-2:]
    if image_h % feature_h != 0 or image_w % feature_w != 0:
        raise ValueError(
            f"Cannot infer SPGM scale: image={image_h}x{image_w}, feature={feature_h}x{feature_w}"
        )

    stride_h = image_h // feature_h
    stride_w = image_w // feature_w
    if stride_h != stride_w or stride_h <= 0 or stride_h & (stride_h - 1):
        raise ValueError(
            f"Expected an equal power-of-two SPGM stride, got {stride_h}x{stride_w} "
            f"for image={image_h}x{image_w}, feature={feature_h}x{feature_w}"
        )
    return f"P{int(math.log2(stride_h))}"


def compute_spgm_aux_prior_loss(
    batch,
    lambda_prior=0.05,
    scale_weights=None,
    mask_mode="binary",
    center_ratio=0.7,
    bce_weight=1.0,
    dice_weight=1.0,
):
    """
    计算所有 SPGM prior logits 的辅助监督损失。

    使用前提：
    - forward 前已经 clear_spgm_aux_cache()
    - SPGM forward 中 collect_aux_logits=True
    - forward 后 get_spgm_aux_cache() 中能拿到带计算图的 prior_logits

    Args:
        batch: Ultralytics 训练 batch，通常包含：
            batch['img']       [B,3,H,W]
            batch['batch_idx'] [N]
            batch['bboxes']    [N,4], normalized xywh
        lambda_prior: 总 prior loss 权重。
        scale_weights: dict。尺度名根据输入图像与 prior logits 的步长动态推断，例如：
            {
                "P3": 0.5,
                "P4": 1.0,
                "P5": 1.0,
            }
        mask_mode: binary / center。
        center_ratio: center 模式下 bbox 收缩比例。
        bce_weight, dice_weight: BCE 和 Dice 内部权重。

    Returns:
        weighted_loss: scalar Tensor
        info: dict
    """
    cache = get_spgm_aux_cache()

    if len(cache) == 0:
        # 没有 SPGM logits 时，返回 0 loss
        device = batch["img"].device
        return torch.zeros((), device=device), {
            "enabled": False,
            "num_items": 0,
            "loss_raw": 0.0,
            "loss_weighted": 0.0,
        }

    if scale_weights is None:
        scale_weights = {
            "P3": 0.5,
            "P4": 1.0,
            "P5": 1.0,
        }

    imgs = batch["img"]
    batch_size = imgs.shape[0]
    device = imgs.device

    batch_idx = batch.get("batch_idx", None)
    bboxes = batch.get("bboxes", None)

    if batch_idx is None or bboxes is None:
        return torch.zeros((), device=device), {
            "enabled": False,
            "reason": "batch has no batch_idx or bboxes",
            "num_items": len(cache),
            "loss_raw": 0.0,
            "loss_weighted": 0.0,
        }

    total_loss = torch.zeros((), device=device)
    total_weight = 0.0

    detail = []

    for item in cache:
        logits = item["logits"]

        if logits is None:
            continue

        _, _, h, w = logits.shape
        scale_name = _infer_pyramid_scale_name(logits, imgs, item.get("aux_name"))
        if scale_name not in scale_weights:
            raise KeyError(
                f"Missing SPGM scale weight for {scale_name}; configured scales={sorted(scale_weights)}"
            )
        scale_weight = float(scale_weights[scale_name])

        targets = build_batch_weak_foreground_masks_torch(
            batch_idx=batch_idx,
            bboxes=bboxes,
            batch_size=batch_size,
            out_hw=(h, w),
            mode=mask_mode,
            center_ratio=center_ratio,
            device=logits.device,
            dtype=logits.dtype,
        )

        loss_i, info_i = bce_dice_prior_loss(
            logits=logits,
            targets=targets,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
        )

        total_loss = total_loss + scale_weight * loss_i
        total_weight += scale_weight

        info_i.update({
            "scale": scale_name,
            "shape": (h, w),
            "scale_weight": scale_weight,
        })
        detail.append(info_i)

    if total_weight > 0:
        loss_raw = total_loss / total_weight
    else:
        loss_raw = total_loss

    weighted_loss = lambda_prior * loss_raw

    info = {
        "enabled": True,
        "num_items": len(cache),
        "lambda_prior": lambda_prior,
        "mask_mode": mask_mode,
        "center_ratio": center_ratio,
        "loss_raw": float(loss_raw.detach()),
        "loss_weighted": float(weighted_loss.detach()),
        "detail": detail,
    }

    return weighted_loss, info

# ============================================================
# DCT import
# ============================================================

try:
    import torch_dct as DCT
except ImportError:
    DCT = None


__all__ = [
    "ScenePriorGuidedModule",
    "CoordEncoding",
    "ExplicitForegroundPriorHead",
    "DctSpatialInteraction",
    "DctChannelInteraction",
    "HFP",
    "clear_spgm_aux_cache",
    "get_spgm_aux_cache",
    "compute_spgm_aux_prior_loss",
    "iter_spgm_modules",
    "set_all_spgm_runtime_mode",
    "reset_all_spgm_feature_stats",
    "collect_all_spgm_feature_stats",
]


def make_group_norm(num_channels, max_groups=32):
    """
    构造稳定的 GroupNorm。
    GroupNorm 要求 num_channels 能被 num_groups 整除。
    """
    groups = min(max_groups, num_channels)
    while groups > 1:
        if num_channels % groups == 0:
            break
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class CoordEncoding(nn.Module):
    """
    坐标编码模块。

    根据输入特征图 x 的空间尺寸 H×W，动态生成坐标图。

    默认输出：
        coord = [x_coord, y_coord]
        shape = [B, 2, H, W]

    include_radius=True 时：
        coord = [x_coord, y_coord, r_coord]
        shape = [B, 3, H, W]
    """

    def __init__(self, coord_range="-1_1", include_radius=False):
        super().__init__()

        assert coord_range in ("-1_1", "0_1"), \
            f"coord_range 只支持 '-1_1' 或 '0_1'，但得到 {coord_range}"

        self.coord_range = coord_range
        self.include_radius = include_radius
        self.out_channels = 3 if include_radius else 2

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, C, H, W]

        Returns:
            coord (Tensor): [B, 2, H, W] 或 [B, 3, H, W]
        """
        b, _, h, w = x.shape
        device = x.device
        dtype = x.dtype

        if self.coord_range == "-1_1":
            x_range = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype)
            y_range = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype)
        else:
            x_range = torch.linspace(0.0, 1.0, steps=w, device=device, dtype=dtype)
            y_range = torch.linspace(0.0, 1.0, steps=h, device=device, dtype=dtype)

        x_coord = x_range.view(1, 1, 1, w).expand(b, 1, h, w)
        y_coord = y_range.view(1, 1, h, 1).expand(b, 1, h, w)

        if self.include_radius:
            if self.coord_range == "-1_1":
                r_coord = torch.sqrt(x_coord ** 2 + y_coord ** 2)
            else:
                r_coord = torch.sqrt((x_coord - 0.5) ** 2 + (y_coord - 0.5) ** 2)

            coord = torch.cat([x_coord, y_coord, r_coord], dim=1)
        else:
            coord = torch.cat([x_coord, y_coord], dim=1)

        return coord


class ExplicitForegroundPriorHead(nn.Module):
    """
    显式前景先验预测头。

    输入：
        X:     [B, C, H, W]
        Coord: [B, C_coord, H, W]

    输出：
        prior_logits: [B, 1, H, W]

    结构：
        Concat[X, Coord]
            -> 1×1 Conv
            -> DW 3×3 Conv
            -> GroupNorm + SiLU
            -> 1×1 Conv
            -> prior_logits
    """

    def __init__(
        self,
        in_channels,
        coord_channels=2,
        hidden_channels=None,
        act=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.coord_channels = coord_channels

        if hidden_channels is None:
            hidden_channels = max(16, in_channels // 4)

        self.hidden_channels = hidden_channels

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels + coord_channels, hidden_channels, kernel_size=1, bias=False),
            make_group_norm(hidden_channels),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )

        self.dwconv = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            make_group_norm(hidden_channels),
            nn.SiLU(inplace=True) if act else nn.Identity(),
        )

        self.pred = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)

    def forward(self, x, coord):
        feat = torch.cat([x, coord], dim=1)
        feat = self.proj(feat)
        feat = self.dwconv(feat)
        logits = self.pred(feat)
        return logits


class ScenePriorGuidedModule(nn.Module):
    """
    Scene-Prior Guided Module, SPGM.

    阶段 3 默认功能：
    - 生成坐标编码；
    - 通过 ExplicitForegroundPriorHead 预测 prior_logits；
    - Sigmoid 得到 foreground prior mask；
    - 使用残差式前景增强：
        Y = X + alpha * X * M_fg

    当前阶段：
    - 不修改检测 loss；
    - 不加入辅助监督；
    - 模块属性里只缓存 detach 后的 Tensor，避免 deepcopy 报错。

    阶段 6 预留：
    - collect_aux_logits=True 时，可以把带计算图的 prior_logits
      放入文件级运行时缓存 _SPGM_AUX_CACHE，用于辅助损失计算。
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        ratio=(0.25, 0.25),
        patch_size=(8, 8),
        isdct=True,

        # 显式 prior 分支
        use_coord=True,
        coord_range="-1_1",
        include_radius=False,
        use_explicit_prior=True,
        prior_hidden_channels=None,
        alpha_init=0.1,

        # ------------------------------------------------------------
        # 新版调制方式：
        # foreground : 原始单向前景调制 Y = X + alpha * X * M
        # fg_bg      : 前景增强 + 背景抑制
        #              Y = X * (1 + a_fg * M - a_bg * (1 - M))
        # bidirectional: fg_bg 的别名，便于实验命名
        # ------------------------------------------------------------
        modulation_type="fg_bg",
        fg_scale_init=0.10,
        bg_scale_init=0.05,
        fg_scale_max=0.30,
        bg_scale_max=0.20,

        # 阶段 6 预留：是否收集带计算图的 logits
        collect_aux_logits=True,
        aux_name=None,

        # 旧版 DCT 分支，默认关闭
        use_legacy_branches=False,
        use_spatial=True,
        use_channel=True,
        use_out_fuse=True,
        legacy_fusion="add",

        debug=False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.ratio = self._normalize_ratio(ratio)
        self.patch_size = self._normalize_patch_size(patch_size)
        self.isdct = isdct

        self.use_coord = use_coord
        self.coord_range = coord_range
        self.include_radius = include_radius
        self.use_explicit_prior = use_explicit_prior

        valid_modulation_types = ("foreground", "fg_bg", "bidirectional")
        if modulation_type not in valid_modulation_types:
            raise ValueError(
                f"Unsupported modulation_type={modulation_type}, "
                f"valid={valid_modulation_types}"
            )
        self.modulation_type = modulation_type
        self.fg_scale_init = float(fg_scale_init)
        self.bg_scale_init = float(bg_scale_init)
        self.fg_scale_max = float(fg_scale_max)
        self.bg_scale_max = float(bg_scale_max)

        self.collect_aux_logits = collect_aux_logits
        self.aux_name = aux_name

        self.use_legacy_branches = use_legacy_branches
        self.use_spatial = use_spatial
        self.use_channel = use_channel
        self.use_out_fuse = use_out_fuse
        self.legacy_fusion = legacy_fusion

        self.debug = debug

        if self.out_channels != self.in_channels and debug:
            print(
                f"[SPGM] 当前 out_channels={self.out_channels} 未参与实际输出通道变换，"
                f"模块输出仍为 in_channels={self.in_channels}。"
            )

        # ------------------------------------------------------------
        # 仅用于调试/可视化的缓存。
        # 注意：这些属性不要保存带计算图的 Tensor。
        # ------------------------------------------------------------
        self.coord_map = None
        self.coord_feature_shape = None
        self.prior_logits = None
        self.prior_mask = None
        self.prior_mask_stats = None

        # ------------------------------------------------------------
        # 诊断/运行时消融开关。
        # normal       : 正常使用 SPGM
        # identity     : 强制输出 Y = X，用于判断主干是否依赖 SPGM
        # shuffle_mask : 打乱 prior mask 空间位置，保留数值分布
        # zero_mask    : prior mask 全 0，理论上等价于 identity
        # one_mask     : prior mask 全 1，测试全局均匀增强
        # ------------------------------------------------------------
        self.runtime_mode = "normal"
        self.collect_feature_stats = True
        self.feature_delta_stats = self._empty_feature_stats()

        # ------------------------------------------------------------
        # 坐标编码
        # ------------------------------------------------------------
        if self.use_coord:
            self.coord_encoder = CoordEncoding(
                coord_range=coord_range,
                include_radius=include_radius,
            )
            self.coord_channels = self.coord_encoder.out_channels
        else:
            self.coord_encoder = None
            self.coord_channels = 0

        # ------------------------------------------------------------
        # 显式前景 Prior Head
        # ------------------------------------------------------------
        if self.use_explicit_prior:
            if not self.use_coord:
                raise ValueError(
                    "Explicit Prior Head 默认依赖 CoordEncoding，请设置 use_coord=True。"
                )

            self.prior_head = ExplicitForegroundPriorHead(
                in_channels=in_channels,
                coord_channels=self.coord_channels,
                hidden_channels=prior_hidden_channels,
            )

            # --------------------------------------------------------
            # 原始单向前景调制参数。
            # 注意：
            # - foreground 模式继续使用该参数，兼容旧版公式；
            # - fg_bg / bidirectional 模式不直接使用 alpha 做调制，
            #   而是使用受限的正向 fg_scale / bg_scale，避免 alpha 学成负值
            #   后出现“前景抑制”和论文表述冲突。
            # --------------------------------------------------------
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

            # --------------------------------------------------------
            # 新版双向调制参数，使用 sigmoid 限幅，保证：
            #   0 < fg_scale < fg_scale_max
            #   0 < bg_scale < bg_scale_max
            #
            # 初始有效尺度接近：
            #   fg_scale_init, bg_scale_init
            # --------------------------------------------------------
            self.fg_scale_raw = nn.Parameter(
                torch.tensor(
                    self._inverse_sigmoid_init(
                        value=self.fg_scale_init,
                        max_value=self.fg_scale_max,
                    )
                )
            )
            self.bg_scale_raw = nn.Parameter(
                torch.tensor(
                    self._inverse_sigmoid_init(
                        value=self.bg_scale_init,
                        max_value=self.bg_scale_max,
                    )
                )
            )
        else:
            self.prior_head = None
            self.alpha = None
            self.fg_scale_raw = None
            self.bg_scale_raw = None

        # ------------------------------------------------------------
        # 旧版 DCT 空间/通道分支
        # 默认不启用，只作为兼容和消融使用。
        # ------------------------------------------------------------
        if self.use_legacy_branches:
            self.spatial = (
                DctSpatialInteraction(
                    in_channels=in_channels,
                    ratio=self.ratio,
                    isdct=isdct,
                    debug=debug,
                )
                if use_spatial
                else nn.Identity()
            )

            self.channel = (
                DctChannelInteraction(
                    in_channels=in_channels,
                    patch=self.patch_size,
                    ratio=self.ratio,
                    isdct=isdct,
                    debug=debug,
                )
                if use_channel
                else nn.Identity()
            )

            if use_out_fuse:
                self.out = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
                    make_group_norm(in_channels),
                )
            else:
                self.out = nn.Identity()

            self.legacy_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.spatial = None
            self.channel = None
            self.out = None
            self.legacy_scale = None

    def clear_cache(self):
        """
        清空调试/可视化缓存。

        注意：
        这些缓存不参与模型参数保存，也不应保存带计算图的 Tensor。
        """
        self.coord_map = None
        self.coord_feature_shape = None
        self.prior_logits = None
        self.prior_mask = None
        self.prior_mask_stats = None

    def __getstate__(self):
        """
        防止 deepcopy(model) 时复制缓存 Tensor。

        Ultralytics 在创建 EMA 或保存模型时可能 deepcopy 整个 model。
        这里确保缓存字段在 deepcopy/pickle 时被置空。
        """
        state = self.__dict__.copy()
        state["coord_map"] = None
        state["coord_feature_shape"] = None
        state["prior_logits"] = None
        state["prior_mask"] = None
        state["prior_mask_stats"] = None
        return state

    @staticmethod
    def _empty_feature_stats():
        """构造 SPGM 特征扰动统计容器。"""
        return {
            "num_batches": 0,
            "alpha": 0.0,          # 兼容旧字段；fg_bg 模式下记录有效前景尺度
            "fg_scale": 0.0,
            "bg_scale": 0.0,
            "modulation_type": "",
            "x_abs_mean": 0.0,
            "delta_abs_mean": 0.0,
            "delta_ratio": 0.0,
            "mask_mean": 0.0,
            "mask_min": 0.0,
            "mask_max": 0.0,
        }

    def set_runtime_mode(self, mode="normal"):
        """
        设置 SPGM 推理/诊断模式。

        Args:
            mode (str):
                normal       : 正常模式
                identity     : 直接返回 X
                shuffle_mask : 打乱 prior mask 空间位置
                zero_mask    : prior mask 全 0
                one_mask     : prior mask 全 1
        """
        valid_modes = ["normal", "identity", "shuffle_mask", "zero_mask", "one_mask"]
        if mode not in valid_modes:
            raise ValueError(f"Unsupported SPGM runtime mode: {mode}, valid={valid_modes}")
        self.runtime_mode = mode

    def reset_feature_stats(self):
        """重置 SPGM 特征扰动统计。"""
        self.feature_delta_stats = self._empty_feature_stats()

    @staticmethod
    def _inverse_sigmoid_init(value, max_value, eps=1e-4):
        """
        将期望的有效尺度 value 映射为 sigmoid 原始参数。

        effective = max_value * sigmoid(raw)

        这样可以让 fg_scale_raw / bg_scale_raw 在初始化时对应
        用户指定的 fg_scale_init / bg_scale_init。
        """
        max_value = float(max_value)
        if max_value <= 0:
            raise ValueError(f"max_value must be positive, got {max_value}")

        p = float(value) / max_value
        p = max(eps, min(1.0 - eps, p))
        return float(math.log(p / (1.0 - p)))

    def _effective_fg_bg_scales(self):
        """
        返回当前有效的前景增强尺度和背景抑制尺度。

        Returns:
            fg_scale, bg_scale: Tensor 标量，可参与反向传播。
        """
        if self.fg_scale_raw is None or self.bg_scale_raw is None:
            device = self.alpha.device if self.alpha is not None else torch.device("cpu")
            zero = torch.zeros((), device=device)
            return zero, zero

        fg_scale = self.fg_scale_max * torch.sigmoid(self.fg_scale_raw)
        bg_scale = self.bg_scale_max * torch.sigmoid(self.bg_scale_raw)

        return fg_scale, bg_scale

    def _effective_fg_bg_scales_float(self):
        """返回 Python float，用于日志和统计。"""
        if self.use_explicit_prior and self.modulation_type in ("fg_bg", "bidirectional"):
            fg, bg = self._effective_fg_bg_scales()
            return float(fg.detach().cpu()), float(bg.detach().cpu())

        alpha = getattr(self, "alpha", None)
        alpha_value = 0.0 if alpha is None else float(alpha.detach().cpu())
        return alpha_value, 0.0

    def _apply_feature_modulation(self, x, used_mask):
        """
        根据 modulation_type 对输入特征进行调制。

        foreground:
            旧版单向前景调制：
                Y = X + alpha * X * M

        fg_bg / bidirectional:
            新版前景增强 + 背景抑制：
                Y = X * (1 + a_fg * M - a_bg * (1 - M))

            其中 a_fg, a_bg 通过 sigmoid 限幅为正数，
            避免 alpha 学成负值导致语义混乱。
        """
        if self.modulation_type == "foreground":
            delta = self.alpha * x * used_mask
            return x + delta, delta

        if self.modulation_type in ("fg_bg", "bidirectional"):
            fg_scale, bg_scale = self._effective_fg_bg_scales()

            # 前景区域 scale > 1，背景区域 scale < 1。
            # bg_scale_max 默认 0.20，因此最低约为 0.80，比较保守。
            scale = 1.0 + fg_scale * used_mask - bg_scale * (1.0 - used_mask)
            out = x * scale
            delta = out - x
            return out, delta

        raise ValueError(f"Unsupported modulation_type={self.modulation_type}")

    def _update_feature_stats(self, x, delta, mask):
        """
        统计 SPGM 对主路径特征的实际扰动幅度。

        关键指标：
            delta_ratio = mean(|Y-X|) / mean(|X|)

        Args:
            x (Tensor): 输入特征 X。
            delta (Tensor): SPGM 造成的特征变化量。
            mask (Tensor): 实际用于调制的 prior mask。
        """
        if not self.collect_feature_stats:
            return

        with torch.no_grad():
            x_abs_mean = x.detach().abs().mean()
            delta_abs_mean = delta.detach().abs().mean()
            delta_ratio = delta_abs_mean / (x_abs_mean + 1e-6)
            mask_detach = mask.detach()

            n = int(self.feature_delta_stats.get("num_batches", 0))
            n_new = n + 1

            def update_avg(old, new):
                return (float(old) * n + float(new)) / n_new

            fg_scale_value, bg_scale_value = self._effective_fg_bg_scales_float()

            self.feature_delta_stats["num_batches"] = n_new
            self.feature_delta_stats["alpha"] = fg_scale_value
            self.feature_delta_stats["fg_scale"] = fg_scale_value
            self.feature_delta_stats["bg_scale"] = bg_scale_value
            self.feature_delta_stats["modulation_type"] = self.modulation_type
            self.feature_delta_stats["x_abs_mean"] = update_avg(
                self.feature_delta_stats.get("x_abs_mean", 0.0),
                x_abs_mean,
            )
            self.feature_delta_stats["delta_abs_mean"] = update_avg(
                self.feature_delta_stats.get("delta_abs_mean", 0.0),
                delta_abs_mean,
            )
            self.feature_delta_stats["delta_ratio"] = update_avg(
                self.feature_delta_stats.get("delta_ratio", 0.0),
                delta_ratio,
            )
            self.feature_delta_stats["mask_mean"] = update_avg(
                self.feature_delta_stats.get("mask_mean", 0.0),
                mask_detach.mean(),
            )
            self.feature_delta_stats["mask_min"] = update_avg(
                self.feature_delta_stats.get("mask_min", 0.0),
                mask_detach.min(),
            )
            self.feature_delta_stats["mask_max"] = update_avg(
                self.feature_delta_stats.get("mask_max", 0.0),
                mask_detach.max(),
            )

    def _shuffle_prior_mask(self, mask):
        """
        打乱 prior mask 的空间位置，但保留每张图的 mask 数值分布。

        Args:
            mask (Tensor): [B, 1, H, W]

        Returns:
            Tensor: shuffle 后的 mask，shape 不变。
        """
        b, c, h, w = mask.shape
        flat = mask.flatten(2)  # [B, 1, H*W]

        # 每张图单独随机打乱，避免 batch 内不同样本共享同一个排列。
        shuffled = []
        for bi in range(b):
            perm = torch.randperm(h * w, device=mask.device)
            shuffled.append(flat[bi:bi + 1, :, perm])

        return torch.cat(shuffled, dim=0).view(b, c, h, w)

    def _apply_runtime_mask_mode(self, prior_mask):
        """根据 runtime_mode 调整实际用于特征调制的 mask。"""
        if self.runtime_mode == "shuffle_mask":
            return self._shuffle_prior_mask(prior_mask)

        if self.runtime_mode == "zero_mask":
            return torch.zeros_like(prior_mask)

        if self.runtime_mode == "one_mask":
            return torch.ones_like(prior_mask)

        return prior_mask

    def forward(self, x):
        """
        Args:
            x (Tensor): [B, C, H, W]

        Returns:
            Tensor: [B, C, H, W]
        """
        if self.debug:
            print(f"[SPGM] input shape: {tuple(x.shape)}")

        # 每次 forward 前清空缓存，避免读取上一批次结果
        self.clear_cache()

        # ------------------------------------------------------------
        # 1. 坐标编码
        # ------------------------------------------------------------
        coord = None
        if self.use_coord and self.coord_encoder is not None:
            coord = self.coord_encoder(x)

            # 这里只缓存 detach 后的坐标图，避免 deepcopy 和显存问题。
            self.coord_map = coord.detach()

            b, c, h, w = x.shape
            self.coord_feature_shape = (b, c + self.coord_channels, h, w)

            if self.debug and not hasattr(self, "_coord_debug_printed"):
                print("\n[SPGM Coord Debug]")
                print("input shape:", tuple(x.shape))
                print("coord shape:", tuple(coord.shape))
                print("future concat shape:", self.coord_feature_shape)
                print("coord min:", float(coord.min().detach()))
                print("coord max:", float(coord.max().detach()))
                print("==================\n")
                self._coord_debug_printed = True

        # ------------------------------------------------------------
        # 2. 显式前景先验分支
        # ------------------------------------------------------------
        if self.use_explicit_prior:
            prior_logits = self.prior_head(x, coord)
            prior_mask = torch.sigmoid(prior_logits)

            # ========================================================
            # 阶段 3：模块属性中只保存 detach 后的结果。
            # 注意：
            # - explicit_out 使用的是未 detach 的 prior_mask；
            # - self.prior_logits/self.prior_mask 只用于调试和可视化。
            # ========================================================
            self.prior_logits = prior_logits.detach()
            self.prior_mask = prior_mask.detach()

            self.prior_mask_stats = {
                "min": float(prior_mask.min().detach()),
                "max": float(prior_mask.max().detach()),
                "mean": float(prior_mask.mean().detach()),
            }

            # ========================================================
            # 阶段 6 预留：
            # 如果 collect_aux_logits=True，则把带计算图的 logits
            # 放入文件级运行时缓存，而不是保存到模块属性中。
            #
            # 阶段 3 默认 collect_aux_logits=False，不会启用。
            # ========================================================
            if self.collect_aux_logits and self.training:
                _append_spgm_aux_cache({
                    "logits": prior_logits,
                    "mask": prior_mask,
                    "shape": tuple(prior_logits.shape),
                    "aux_name": self.aux_name,
                    "module_id": id(self),
                })

            if self.debug and not hasattr(self, "_prior_debug_printed"):
                print("\n[SPGM Prior Debug]")
                print("prior_logits shape:", tuple(prior_logits.shape))
                print("prior_mask shape:", tuple(prior_mask.shape))
                print("prior_mask min:", self.prior_mask_stats["min"])
                print("prior_mask max:", self.prior_mask_stats["max"])
                print("prior_mask mean:", self.prior_mask_stats["mean"])
                if self.modulation_type == "foreground":
                    print("alpha:", float(self.alpha.detach()))
                else:
                    fg_scale, bg_scale = self._effective_fg_bg_scales_float()
                    print("modulation_type:", self.modulation_type)
                    print("fg_scale:", fg_scale)
                    print("bg_scale:", bg_scale)
                print("==================\n")
                self._prior_debug_printed = True

            # ------------------------------------------------------------
            # 运行时诊断模式：
            # - identity：强制模块退化为恒等映射，用于判断检测主干是否依赖 SPGM
            # - shuffle_mask：打乱 mask 空间位置，验证空间先验是否真正被利用
            # - zero_mask/one_mask：辅助判断 SPGM 是空间调制还是全局缩放
            # ------------------------------------------------------------
            if self.runtime_mode == "identity":
                zero_delta = torch.zeros_like(x)
                self._update_feature_stats(x, zero_delta, prior_mask)
                explicit_out = x
            else:
                used_mask = self._apply_runtime_mask_mode(prior_mask)

                # 这里必须使用未 detach 的 used_mask，保证 prior_head 能通过检测损失反传。
                # normal 模式下 used_mask == prior_mask；其他模式主要用于 eval 诊断。
                explicit_out, delta = self._apply_feature_modulation(x, used_mask)

                self._update_feature_stats(x, delta, used_mask)
        else:
            if self.runtime_mode == "identity":
                explicit_out = x
            else:
                explicit_out = x

        # ------------------------------------------------------------
        # 3. 可选旧版 DCT 分支
        # 默认不启用。
        # ------------------------------------------------------------
        if self.use_legacy_branches:
            spatial_feat = self.spatial(x)
            channel_feat = self.channel(x)
            legacy_out = self.out(spatial_feat + channel_feat)

            if self.use_explicit_prior:
                if self.legacy_fusion == "replace":
                    out = legacy_out
                elif self.legacy_fusion == "add":
                    out = explicit_out + self.legacy_scale * legacy_out
                else:
                    raise ValueError(f"不支持的 legacy_fusion: {self.legacy_fusion}")
            else:
                out = legacy_out
        else:
            out = explicit_out

        if self.debug:
            print(f"[SPGM] output shape: {tuple(out.shape)}")

        return out

    @staticmethod
    def _normalize_ratio(ratio):
        if isinstance(ratio, (int, float)):
            ratio = (float(ratio), float(ratio))
        elif isinstance(ratio, list):
            ratio = tuple(ratio)

        assert isinstance(ratio, tuple) and len(ratio) == 2, \
            f"ratio 应为 float 或长度为 2 的 tuple/list，但得到 {ratio}"

        return ratio

    @staticmethod
    def _normalize_patch_size(patch_size):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, list):
            patch_size = tuple(patch_size)

        assert isinstance(patch_size, tuple) and len(patch_size) == 2, \
            f"patch_size 应为 int 或长度为 2 的 tuple/list，但得到 {patch_size}"

        return patch_size


class DctSpatialInteraction(nn.Module):
    """
    旧版 DCT 空间分支。

    功能：
    - isdct=True:
        X -> DCT -> 高频掩码 -> IDCT -> X * 高频响应
    - isdct=False:
        X -> 1×1 Conv -> Sigmoid -> X * spatial_mask
    """

    def __init__(self, in_channels, ratio=(0.25, 0.25), isdct=True, debug=False):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = self._normalize_ratio(ratio)
        self.isdct = isdct
        self.debug = debug

        if not self.isdct:
            self.spatial1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        _, _, h, w = x.size()

        if not self.isdct:
            mask = torch.sigmoid(self.spatial1x1(x))
            return x * mask

        if DCT is None:
            if self.debug:
                print("[DctSpatialInteraction] DCT is None, return identity.")
            return x

        try:
            x_dct = DCT.dct_2d(x, norm="ortho")
        except Exception as e:
            if self.debug:
                print(f"[DctSpatialInteraction] DCT failed, return identity. Error: {e}")
            return x

        weight = self._compute_frequency_mask(h, w, self.ratio, device=x.device, dtype=x.dtype)
        weight = weight.view(1, 1, h, w).expand_as(x_dct)

        x_dct_high = x_dct * weight

        try:
            x_high = DCT.idct_2d(x_dct_high, norm="ortho")
        except Exception as e:
            if self.debug:
                print(f"[DctSpatialInteraction] IDCT failed, return identity. Error: {e}")
            return x

        return x * x_high

    @staticmethod
    def _compute_frequency_mask(h, w, ratio, device=None, dtype=None):
        r_h, r_w = ratio
        h0 = int(h * r_h)
        w0 = int(w * r_w)

        weight = torch.ones((h, w), device=device, dtype=dtype, requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

    @staticmethod
    def _normalize_ratio(ratio):
        if isinstance(ratio, (int, float)):
            ratio = (float(ratio), float(ratio))
        elif isinstance(ratio, list):
            ratio = tuple(ratio)

        assert isinstance(ratio, tuple) and len(ratio) == 2, \
            f"ratio 应为 float 或长度为 2 的 tuple/list，但得到 {ratio}"

        return ratio


class DctChannelInteraction(nn.Module):
    """
    旧版 DCT 通道分支。

    功能：
    - isdct=True:
        X -> DCT -> 高频掩码 -> IDCT
          -> Adaptive MaxPool / Adaptive AvgPool
          -> ReLU -> spatial sum
          -> grouped 1×1 Conv -> Sigmoid
          -> X * channel_weight

    - isdct=False:
        X -> Adaptive MaxPool / Adaptive AvgPool
          -> ReLU -> grouped 1×1 Conv -> Sigmoid
          -> X * channel_weight
    """

    def __init__(self, in_channels, patch=(8, 8), ratio=(0.25, 0.25), isdct=True, debug=False):
        super().__init__()

        self.in_channels = in_channels
        self.patch = self._normalize_patch_size(patch)
        self.h = self.patch[0]
        self.w = self.patch[1]
        self.ratio = self._normalize_ratio(ratio)
        self.isdct = isdct
        self.debug = debug

        self.groups = 32
        if in_channels % self.groups != 0:
            raise ValueError(
                f"in_channels={in_channels} 不能被 groups={self.groups} 整除，"
                f"请调整 groups 或通道数。"
            )

        self.channel1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=self.groups)
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()

        if not self.isdct:
            return self._channel_attention_from_feature(x)

        if DCT is None:
            if self.debug:
                print("[DctChannelInteraction] DCT is None, use non-DCT channel attention.")
            return self._channel_attention_from_feature(x)

        try:
            x_dct = DCT.dct_2d(x, norm="ortho")
        except Exception as e:
            if self.debug:
                print(f"[DctChannelInteraction] DCT failed, use non-DCT channel attention. Error: {e}")
            return self._channel_attention_from_feature(x)

        weight = self._compute_frequency_mask(h, w, self.ratio, device=x.device, dtype=x.dtype)
        weight = weight.view(1, 1, h, w).expand_as(x_dct)

        x_dct_high = x_dct * weight

        try:
            x_high = DCT.idct_2d(x_dct_high, norm="ortho")
        except Exception as e:
            if self.debug:
                print(f"[DctChannelInteraction] IDCT failed, use non-DCT channel attention. Error: {e}")
            return self._channel_attention_from_feature(x)

        return self._channel_attention_from_feature(x, attention_source=x_high)

    def _channel_attention_from_feature(self, x, attention_source=None):
        if attention_source is None:
            attention_source = x

        n, c, _, _ = x.size()

        amaxp = F.adaptive_max_pool2d(attention_source, output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(attention_source, output_size=(self.h, self.w))

        amaxp = torch.sum(self.relu(amaxp), dim=[2, 3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2, 3]).view(n, c, 1, 1)

        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp)
        channel_weight = torch.sigmoid(channel)

        return x * channel_weight

    @staticmethod
    def _compute_frequency_mask(h, w, ratio, device=None, dtype=None):
        r_h, r_w = ratio
        h0 = int(h * r_h)
        w0 = int(w * r_w)

        weight = torch.ones((h, w), device=device, dtype=dtype, requires_grad=False)
        weight[:h0, :w0] = 0
        return weight

    @staticmethod
    def _normalize_ratio(ratio):
        if isinstance(ratio, (int, float)):
            ratio = (float(ratio), float(ratio))
        elif isinstance(ratio, list):
            ratio = tuple(ratio)

        assert isinstance(ratio, tuple) and len(ratio) == 2, \
            f"ratio 应为 float 或长度为 2 的 tuple/list，但得到 {ratio}"

        return ratio

    @staticmethod
    def _normalize_patch_size(patch_size):
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, list):
            patch_size = tuple(patch_size)

        assert isinstance(patch_size, tuple) and len(patch_size) == 2, \
            f"patch_size 应为 int 或长度为 2 的 tuple/list，但得到 {patch_size}"

        return patch_size


# ============================================================
# SPGM Global Diagnostic Helpers
# ------------------------------------------------------------
# 这些函数用于外部验证脚本统一设置所有 SPGM 模块，
# 例如 normal / identity / shuffle_mask / zero_mask / one_mask。
# ============================================================


def _unwrap_model_for_spgm(model):
    """兼容 DDP/DataParallel/Ultralytics 外层对象，返回实际 torch module。"""
    if hasattr(model, "model") and isinstance(getattr(model, "model"), nn.Module):
        model = model.model

    while hasattr(model, "module"):
        model = model.module

    return model


def iter_spgm_modules(model):
    """遍历模型中的所有 ScenePriorGuidedModule。"""
    model = _unwrap_model_for_spgm(model)

    for name, m in model.named_modules():
        if m.__class__.__name__ == "ScenePriorGuidedModule":
            yield name, m


def set_all_spgm_runtime_mode(model, mode="normal"):
    """统一设置所有 SPGM 模块的 runtime_mode。"""
    for _, m in iter_spgm_modules(model):
        if hasattr(m, "set_runtime_mode"):
            m.set_runtime_mode(mode)


def reset_all_spgm_feature_stats(model):
    """统一重置所有 SPGM 模块的特征扰动统计。"""
    for _, m in iter_spgm_modules(model):
        if hasattr(m, "reset_feature_stats"):
            m.reset_feature_stats()


def collect_all_spgm_feature_stats(model):
    """
    收集所有 SPGM 模块的 alpha 和 feature_delta_ratio。

    Returns:
        rows (list[dict]): 每个 SPGM 模块一行。
    """
    rows = []

    for name, m in iter_spgm_modules(model):
        stats = getattr(m, "feature_delta_stats", None)
        alpha = getattr(m, "alpha", None)
        alpha_value = None if alpha is None else float(alpha.detach().cpu())

        fg_scale_value = None
        bg_scale_value = None
        if hasattr(m, "_effective_fg_bg_scales_float"):
            try:
                fg_scale_value, bg_scale_value = m._effective_fg_bg_scales_float()
            except Exception:
                fg_scale_value, bg_scale_value = None, None

        row = {
            "module": name,
            "alpha": alpha_value,
            "fg_scale": fg_scale_value,
            "bg_scale": bg_scale_value,
            "runtime_mode": getattr(m, "runtime_mode", None),
            "modulation_type": getattr(m, "modulation_type", None),
        }

        if stats is None:
            row.update({
                "num_batches": 0,
                "x_abs_mean": None,
                "delta_abs_mean": None,
                "delta_ratio": None,
                "mask_mean": None,
                "mask_min": None,
                "mask_max": None,
            })
        else:
            row.update({
                "num_batches": stats.get("num_batches", 0),
                "stats_fg_scale": stats.get("fg_scale", None),
                "stats_bg_scale": stats.get("bg_scale", None),
                "stats_modulation_type": stats.get("modulation_type", None),
                "x_abs_mean": stats.get("x_abs_mean", None),
                "delta_abs_mean": stats.get("delta_abs_mean", None),
                "delta_ratio": stats.get("delta_ratio", None),
                "mask_mean": stats.get("mask_mean", None),
                "mask_min": stats.get("mask_min", None),
                "mask_max": stats.get("mask_max", None),
            })

        rows.append(row)

    return rows


class HFP(ScenePriorGuidedModule):
    """
    兼容旧名称 HFP。

    HFP 默认走旧版 DCT 空间/通道分支逻辑，
    不启用阶段 3 的 explicit prior head。
    """

    def __init__(self, in_channels, ratio=(0.25, 0.25), patch=(8, 8), isdct=True):
        super().__init__(
            in_channels=in_channels,
            out_channels=in_channels,
            ratio=ratio,
            patch_size=patch,
            isdct=isdct,
            use_coord=False,
            use_explicit_prior=False,
            use_legacy_branches=True,
            use_spatial=True,
            use_channel=True,
            use_out_fuse=True,
        )
