"""
test_train.py — Smoke & Functional tests for the training pipeline
===================================================================
"""
import pytest
import argparse
import torch
import torch.nn as nn
from src.train import FocalLoss, profile_model
from src.model import get_model


# ============================================================
# SMOKE TESTS
# ============================================================
class TestSmoke:
    def test_focal_loss_instantiates(self):
        """FocalLoss can be created."""
        loss = FocalLoss(gamma=2.0)
        assert loss is not None

    def test_profile_model_runs(self):
        """profile_model returns valid dict on CPU."""
        model = get_model('SDR_Custom_CoordASPP_Focal')
        result = profile_model(model, torch.device('cpu'))
        assert 'Total_Parameters' in result
        assert 'Inference_Time_ms' in result


# ============================================================
# FUNCTIONAL TESTS
# ============================================================
class TestFunctional:
    def test_focal_loss_forward(self):
        """FocalLoss computes a scalar loss."""
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 6)
        targets = torch.randint(0, 6, (4,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_focal_loss_gamma_effect(self):
        """Higher gamma should penalize easy examples less -> different loss values."""
        logits = torch.randn(8, 6)
        targets = torch.randint(0, 6, (8,))
        loss_g1 = FocalLoss(gamma=1.0)(logits, targets).item()
        loss_g3 = FocalLoss(gamma=3.0)(logits, targets).item()
        # They should be different (not exactly equal)
        assert loss_g1 != loss_g3

    def test_profile_metrics_are_positive(self):
        """All profiling metrics should be positive numbers."""
        model = get_model('SDR_Custom_CoordASPP_Focal')
        result = profile_model(model, torch.device('cpu'))
        assert result['Total_Parameters'] > 0
        assert result['Trainable_Parameters'] > 0
        assert result['Model_Size_MB'] > 0
        assert result['Inference_Time_ms'] > 0

    def test_focal_loss_matches_ce_at_gamma_zero(self):
        """When gamma=0, FocalLoss should equal CrossEntropyLoss."""
        torch.manual_seed(42)
        logits = torch.randn(8, 6)
        targets = torch.randint(0, 6, (8,))
        focal = FocalLoss(gamma=0.0)(logits, targets).item()
        ce = nn.CrossEntropyLoss()(logits, targets).item()
        assert abs(focal - ce) < 1e-4, f"FocalLoss(gamma=0)={focal} != CE={ce}"
