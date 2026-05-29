"""
test_pipeline.py — Integration tests for the end-to-end pipeline
=================================================================
Tests that run_ablation, evaluate, and generate_figures modules
can be imported and their key functions invoked without crashes.
"""
import pytest
import os
import json
import argparse
import torch
from unittest.mock import patch, MagicMock


# ============================================================
# SMOKE TESTS — Can we import all modules?
# ============================================================
class TestSmoke:
    def test_import_run_ablation(self):
        from src.run_ablation import phase_backbone, phase_hparam, CUSTOM_MODEL
        assert CUSTOM_MODEL == 'SDR_Custom_CoordASPP_Focal'

    def test_import_evaluate(self):
        from src.evaluate import evaluate_model
        assert callable(evaluate_model)

    def test_import_generate_figures(self):
        from src.generate_figures import (
            plot_training_curves, plot_radar_chart,
            plot_pareto, plot_hparam_importance, plot_accuracy_bars
        )
        assert callable(plot_training_curves)

    def test_import_benchmark_edge(self):
        from src.benchmark_edge import benchmark_pytorch, run_benchmarks
        assert callable(benchmark_pytorch)


# ============================================================
# FUNCTIONAL TESTS
# ============================================================
class TestFunctional:
    def test_generate_figures_with_dummy_data(self, tmp_path):
        """generate_figures reads JSON and produces images without crashing."""
        from src.generate_figures import plot_training_curves

        # Create dummy history JSON
        os.makedirs(tmp_path / 'results' / 'logs', exist_ok=True)
        os.makedirs(tmp_path / 'results' / 'figures', exist_ok=True)

        dummy_history = {
            "model": "TestModel",
            "profile": {"Total_Parameters": 1000, "Inference_Time_ms": 5.0, "Model_Size_MB": 1.0},
            "best_val_acc": 90.0,
            "history": [
                {"epoch": i, "train_loss": 1/(i+1), "val_loss": 1.1/(i+1),
                 "train_acc": 50+i*5, "val_acc": 48+i*5}
                for i in range(5)
            ]
        }

        log_path = tmp_path / 'results' / 'logs' / 'history_TestModel.json'
        with open(log_path, 'w') as f:
            json.dump(dummy_history, f)

        # patch the hardcoded path inside the function
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            plot_training_curves("TestModel")
            assert os.path.exists(tmp_path / 'results' / 'figures' / 'training_curves_TestModel.png')
        finally:
            os.chdir(original_cwd)

    def test_benchmark_single_model(self):
        """benchmark_pytorch returns valid timing dict."""
        from src.benchmark_edge import benchmark_pytorch
        from src.model import get_model

        model = get_model('SDR_Custom_CoordASPP_Focal')
        result = benchmark_pytorch(model, torch.device('cpu'), n_runs=5)

        assert 'mean_ms' in result
        assert 'min_ms' in result
        assert result['mean_ms'] > 0


# ============================================================
# INTEGRATION TESTS
# ============================================================
class TestIntegration:
    def test_ablation_backbone_list_is_complete(self):
        """Verify the backbone phase includes our custom model + 18 baselines."""
        from src.run_ablation import CUSTOM_MODEL
        from src.model import TIMM_MODEL_MAP

        models = [CUSTOM_MODEL] + list(TIMM_MODEL_MAP.keys())
        assert len(models) == 19
        assert models[0] == 'SDR_Custom_CoordASPP_Focal'

    def test_evaluate_model_structure(self):
        """evaluate_model accepts a Namespace with expected attributes."""
        from src.evaluate import evaluate_model
        args = argparse.Namespace(
            data_dir='nonexistent_dir',
            model='SDR_Custom_CoordASPP_Focal',
            batch_size=16
        )
        # Should fail gracefully because data_dir doesn't exist
        with pytest.raises((FileNotFoundError, RuntimeError)):
            evaluate_model(args)

    def test_full_forward_backward_on_all_custom_modules(self):
        """Full gradient check: input -> model -> loss -> backward on custom arch."""
        from src.model import SDR_Custom_CoordASPP_Focal
        from src.train import FocalLoss

        model = SDR_Custom_CoordASPP_Focal(num_classes=6, pretrained=False)
        model.train()
        criterion = FocalLoss(gamma=2.0)

        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 6, (2,))

        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        # Check gradients flow through all custom modules
        for name, param in model.named_parameters():
            if param.requires_grad and 'bn' not in name:
                assert param.grad is not None, f"No gradient for {name}"
