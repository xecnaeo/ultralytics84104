"""Focused tests for SPGM weak foreground masks and auxiliary prior loss."""

import unittest

import torch

from ultralytics import YOLO
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.nn.modules.scene_prior_guided import (
    ScenePriorGuidedModule,
    _infer_pyramid_scale_name,
    bce_dice_prior_loss,
    build_batch_weak_foreground_masks_torch,
)


def build_mask(bboxes, out_hw=(20, 20), mode="binary", center_ratio=1.0, batch_idx=None, batch_size=1):
    bboxes = torch.tensor(bboxes, dtype=torch.float32)
    if batch_idx is None:
        batch_idx = torch.zeros(len(bboxes), dtype=torch.long)
    else:
        batch_idx = torch.tensor(batch_idx, dtype=torch.long)
    return build_batch_weak_foreground_masks_torch(
        batch_idx=batch_idx,
        bboxes=bboxes,
        batch_size=batch_size,
        out_hw=out_hw,
        mode=mode,
        center_ratio=center_ratio,
    )


class SPGMMaskTest(unittest.TestCase):
    def test_binary_mask_fills_full_box(self):
        mask = build_mask([[0.5, 0.5, 0.5, 0.5]])
        expected = torch.zeros((1, 1, 20, 20))
        expected[:, :, 5:15, 5:15] = 1
        self.assertTrue(torch.equal(mask, expected))

    def test_center_mask_shrinks_box_about_center(self):
        mask = build_mask([[0.5, 0.5, 0.5, 0.5]], mode="center", center_ratio=0.7)
        expected = torch.zeros((1, 1, 20, 20))
        expected[:, :, 6:14, 6:14] = 1
        self.assertTrue(torch.equal(mask, expected))

    def test_multiple_boxes_are_merged_per_image(self):
        combined = build_mask(
            [[0.25, 0.5, 0.25, 0.5], [0.75, 0.5, 0.25, 0.5]],
            out_hw=(12, 12),
        )
        first = build_mask([[0.25, 0.5, 0.25, 0.5]], out_hw=(12, 12))
        second = build_mask([[0.75, 0.5, 0.25, 0.5]], out_hw=(12, 12))
        self.assertTrue(torch.equal(combined.bool(), first.bool() | second.bool()))

    def test_boundary_box_is_clipped_to_feature_map(self):
        mask = build_mask([[0.0, 0.0, 0.4, 0.4]], out_hw=(10, 10))
        expected = torch.zeros((1, 1, 10, 10))
        expected[:, :, 0:2, 0:2] = 1
        self.assertTrue(torch.equal(mask, expected))

    def test_tiny_box_occupies_at_least_one_cell(self):
        mask = build_mask([[0.5, 0.5, 0.0, 0.0]], out_hw=(10, 10), mode="center", center_ratio=0.7)
        self.assertEqual(mask.sum().item(), 1)
        self.assertEqual(mask[0, 0, 5, 5].item(), 1)

    def test_invalid_center_ratio_is_rejected(self):
        for center_ratio in (0.0, -0.1, 1.1):
            with self.subTest(center_ratio=center_ratio):
                with self.assertRaisesRegex(ValueError, "center_ratio"):
                    build_mask([[0.5, 0.5, 0.5, 0.5]], mode="center", center_ratio=center_ratio)

    def test_invalid_mask_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "binary / center"):
            build_mask([[0.5, 0.5, 0.5, 0.5]], mode="gaussian")

    def test_bce_dice_prior_loss_backpropagates_finite_nonzero_gradients(self):
        logits = torch.randn((2, 1, 8, 8), requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[:, :, 2:6, 2:6] = 1
        loss, info = bce_dice_prior_loss(logits, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum(), 0)
        self.assertGreater(info["loss"], 0)

    def test_scale_inference_at_960(self):
        images = torch.empty((1, 3, 960, 960))
        for feature_hw, expected in (((120, 120), "P3"), ((60, 60), "P4"), ((30, 30), "P5")):
            with self.subTest(feature_hw=feature_hw):
                logits = torch.empty((1, 1, *feature_hw))
                self.assertEqual(_infer_pyramid_scale_name(logits, images), expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA integration smoke test")
    def test_yolo26s_spgm_structure_aux_loss_and_gradients(self):
        spgm_yaml = "ultralytics/cfg/models/26/yolo26s-SPGM.yaml"
        official_yaml = "ultralytics/cfg/models/26/yolo26.yaml"
        model = YOLO(spgm_yaml).model.cuda().train()
        model.args = DEFAULT_CFG
        model.spgm_aux_config = {
            "lambda_prior": 0.05,
            "scale_weights": {"P3": 0.5, "P4": 1.0, "P5": 1.0},
            "mask_mode": "center",
            "center_ratio": 0.7,
            "bce_weight": 1.0,
            "dice_weight": 1.0,
        }
        spgm_modules = [module for module in model.modules() if isinstance(module, ScenePriorGuidedModule)]
        self.assertEqual(len(spgm_modules), 3)

        batch = {
            "img": torch.rand((2, 3, 640, 640), device="cuda"),
            "batch_idx": torch.tensor([0, 1], device="cuda"),
            "cls": torch.zeros((2, 1), device="cuda"),
            "bboxes": torch.tensor(
                [[0.5, 0.5, 0.25, 0.3], [0.35, 0.6, 0.2, 0.2]],
                device="cuda",
            ),
        }
        loss, _ = model.loss(batch)
        loss.sum().backward()

        details = model.spgm_prior_info["detail"]
        self.assertEqual({item["scale"] for item in details}, {"P3", "P4", "P5"})
        self.assertEqual({tuple(item["shape"]) for item in details}, {(80, 80), (40, 40), (20, 20)})
        self.assertEqual(model.spgm_prior_info["mask_mode"], "center")
        self.assertEqual(model.spgm_prior_info["center_ratio"], 0.7)
        for module in spgm_modules:
            gradients = [parameter.grad for parameter in module.prior_head.parameters()]
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
            self.assertGreater(sum(gradient.abs().sum() for gradient in gradients), 0)

        official_model = YOLO(official_yaml).model
        self.assertFalse(any(isinstance(module, ScenePriorGuidedModule) for module in official_model.modules()))


if __name__ == "__main__":
    unittest.main()
