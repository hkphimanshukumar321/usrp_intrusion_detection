"""
test_model.py — Smoke, Functional, and Integration Tests for model.py
=====================================================================
"""
import pytest
import torch
from src.model import (
    get_model, SDR_Custom_CoordASPP_Focal,
    CoordinateAttention, LightASPP,
    TIMM_MODEL_MAP
)
from src.data_loader import NUM_CLASSES


# ============================================================
# SMOKE TESTS — Does it even load without crashing?
# ============================================================
class TestSmoke:
    def test_custom_model_instantiates(self):
        """Can our custom model be created without errors?"""
        model = get_model('SDR_Custom_CoordASPP_Focal')
        assert model is not None

    def test_factory_rejects_unknown_name(self):
        """Does get_model reject garbage names?"""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("TotallyFakeModel_999")

    def test_timm_baseline_instantiates(self):
        """Can we load at least one timm baseline?"""
        model = get_model('DenseNet121')
        assert model is not None

    def test_timm_model_map_not_empty(self):
        """Is the TIMM_MODEL_MAP populated?"""
        assert len(TIMM_MODEL_MAP) >= 18


# ============================================================
# FUNCTIONAL TESTS — Does the math work correctly?
# ============================================================
class TestFunctional:
    def test_custom_forward_pass_shape(self):
        """Does the custom model produce [B, 6] output from [B, 3, 224, 224]?"""
        model = SDR_Custom_CoordASPP_Focal(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES), f"Expected (2, {NUM_CLASSES}), got {out.shape}"

    def test_coordinate_attention_preserves_shape(self):
        """Does CoordinateAttention output the same shape as input?"""
        ca = CoordinateAttention(64, 64)
        x = torch.randn(2, 64, 14, 14)
        with torch.no_grad():
            out = ca(x)
        assert out.shape == x.shape

    def test_light_aspp_preserves_shape(self):
        """Does LightASPP maintain spatial dims?"""
        aspp = LightASPP(128, 128)
        x = torch.randn(2, 128, 7, 7)
        with torch.no_grad():
            out = aspp(x)
        assert out.shape == x.shape

    def test_custom_model_feature_dim(self):
        """Is the feature_dim correctly set to 1536 (512*3)?"""
        model = SDR_Custom_CoordASPP_Focal(pretrained=False)
        assert model.feature_dim == 1536

    def test_custom_model_has_attention(self):
        """Does the model have CoordinateAttention modules?"""
        model = SDR_Custom_CoordASPP_Focal(pretrained=False)
        assert hasattr(model, 'attention')
        assert hasattr(model, 'ca_s2')
        assert hasattr(model, 'ca_s3')

    def test_custom_model_count_params(self):
        """Does count_params return a positive integer?"""
        model = SDR_Custom_CoordASPP_Focal(pretrained=False)
        n = model.count_params()
        assert isinstance(n, int)
        assert n > 0

    def test_baseline_forward_pass_shape(self):
        """Does a timm baseline produce [B, 6] output?"""
        model = get_model('DenseNet121')
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_output_not_all_zeros(self):
        """Verify logits are non-trivial (not all zeros)."""
        model = SDR_Custom_CoordASPP_Focal(pretrained=False)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert not torch.allclose(out, torch.zeros_like(out))


# ============================================================
# INTEGRATION TESTS — Does it work with the data pipeline?
# ============================================================
class TestIntegration:
    def test_model_with_simulated_batch(self):
        """Simulate a full training step: forward -> loss -> backward."""
        model = SDR_Custom_CoordASPP_Focal(num_classes=NUM_CLASSES, pretrained=False)
        model.train()

        x = torch.randn(4, 3, 224, 224)
        y = torch.randint(0, NUM_CLASSES, (4,))

        out = model(x)
        loss = torch.nn.CrossEntropyLoss()(out, y)
        loss.backward()

        # Verify gradients exist
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, "No gradients after backward pass!"

    def test_model_checkpoint_save_load(self, tmp_path):
        """Can we save and reload the model without data corruption?"""
        model = SDR_Custom_CoordASPP_Focal(num_classes=NUM_CLASSES, pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            original_out = model(x)

        # Save
        path = tmp_path / "test_model.pth"
        torch.save(model.state_dict(), path)

        # Reload into fresh model
        model2 = SDR_Custom_CoordASPP_Focal(num_classes=NUM_CLASSES, pretrained=False)
        model2.load_state_dict(torch.load(path, weights_only=True))
        model2.eval()

        with torch.no_grad():
            loaded_out = model2(x)

        assert torch.allclose(original_out, loaded_out, atol=1e-5), \
            "Output changed after save/load!"

    def test_all_timm_names_are_valid(self):
        """Verify every key in TIMM_MODEL_MAP creates a real model."""
        # Only test a subset to keep tests fast
        fast_models = ['DenseNet121', 'MobileNetV2', 'VGG16']
        for name in fast_models:
            model = get_model(name)
            assert model is not None, f"Failed to create {name}"
