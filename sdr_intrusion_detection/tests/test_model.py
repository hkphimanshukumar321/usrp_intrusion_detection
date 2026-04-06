import torch
from src.model import (
    ComplexEncoder, 
    CBAM1D, 
    CrossModalAttention, 
    DualBranchFusionCNN, 
    NUM_CLASSES, 
    WINDOW_SIZE
)

def test_complex_encoder(dummy_iq_batch):
    """Verify the Math: [B, W, 2] -> [B, 4, W] with correct magnitudes and phases"""
    encoder = ComplexEncoder()
    out = encoder(dummy_iq_batch)
    
    B, W, _ = dummy_iq_batch.shape
    assert out.shape == (B, 4, W), "ComplexEncoder output shape mismatch"
    
    # Assert amplitude is mathematically correct: sqrt(I^2 + Q^2)
    # the encoder outputs [I, Q, amp, phase]
    # out[:, 0:1] is I, out[:, 1:2] is Q, out[:, 2:3] is amp
    I = out[:, 0, :]
    Q = out[:, 1, :]
    amp = out[:, 2, :]
    
    expected_amp = torch.sqrt(I**2 + Q**2)
    torch.testing.assert_close(amp, expected_amp, rtol=1e-5, atol=1e-5)

def test_cbam1d():
    """Verify CBAM channel and spatial attention logic preserves shapes."""
    batch_size = 4
    channels = 32
    length = 128
    dummy_input = torch.randn(batch_size, channels, length)
    
    cbam = CBAM1D(channels=channels, reduction=8)
    out = cbam(dummy_input)
    assert out.shape == dummy_input.shape, "CBAM1D altered the input tensor shape"

def test_cross_modal_attention():
    """Verify cross modality attention logic preserves dimension sizes."""
    batch_size = 2
    dim = 64
    seq_len = 16
    freq_time_flat = 36  # F * T equivalent
    
    # dummy feature maps representing outputs of CNNs before pooling
    iq_maps = torch.randn(batch_size, dim, seq_len)
    spec_maps = torch.randn(batch_size, dim, 6, 6) # [B, dim, F', T'] where 6*6=36
    
    cross_attn = CrossModalAttention(dim=dim, num_heads=2)
    out_iq, out_spec = cross_attn(iq_maps, spec_maps)
    
    assert out_iq.shape == iq_maps.shape, "CrossModalAttention messed up IQ branch shape"
    assert out_spec.shape == spec_maps.shape, "CrossModalAttention messed up Spectrogram branch shape"

def test_dual_branch_fusion_forward(dummy_iq_batch, dummy_spec_batch):
    """Smoke Test: Verify end-to-end forward pass of the SOTA model natively."""
    model = DualBranchFusionCNN(
        window_size=WINDOW_SIZE, 
        num_classes=NUM_CLASSES, 
        branch_dim=64, # smaller dim for faster testing
        fusion_dim=64
    )
    
    # Just to confirm the inputs match what the Data Loader yields
    assert dummy_iq_batch.shape[1] == WINDOW_SIZE
    assert dummy_iq_batch.shape[2] == 2
    
    logits = model(dummy_iq_batch, dummy_spec_batch)
    
    B = dummy_iq_batch.shape[0]
    assert logits.shape == (B, NUM_CLASSES), "Output logits must be shape [B, num_classes]"

def test_dual_branch_fusion_get_features(dummy_iq_batch, dummy_spec_batch):
    """Smoke Test: Verify feature extraction works for t-SNE evaluation."""
    model = DualBranchFusionCNN(branch_dim=64, fusion_dim=64)
    features = model.get_features(dummy_iq_batch, dummy_spec_batch)
    
    B = dummy_iq_batch.shape[0]
    assert features.shape == (B, 64), "get_features must return exactly the fusion_dim size [B, 64] before the final layer"
