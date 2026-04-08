import torch

from src.train import prepare_model_inputs


def test_prepare_model_inputs_routes_iq_batch(dummy_iq_batch, dummy_spec_batch):
    batch = {
        "iq": dummy_iq_batch,
        "spectrogram": dummy_spec_batch,
        "label": torch.tensor([0, 1, 2, 3]),
    }

    inputs, labels = prepare_model_inputs(batch, torch.device("cpu"), "iq")

    assert len(inputs) == 1
    assert inputs[0].shape == dummy_iq_batch.shape
    assert labels.shape[0] == dummy_iq_batch.shape[0]


def test_prepare_model_inputs_routes_spectrogram_batch(dummy_iq_batch, dummy_spec_batch):
    batch = {
        "iq": dummy_iq_batch,
        "spectrogram": dummy_spec_batch,
        "label": torch.tensor([0, 1, 2, 3]),
    }

    inputs, labels = prepare_model_inputs(batch, torch.device("cpu"), "spectrogram")

    assert len(inputs) == 1
    assert inputs[0].shape == dummy_spec_batch.shape
    assert labels.shape[0] == dummy_spec_batch.shape[0]


def test_prepare_model_inputs_routes_dual_batch(dummy_iq_batch, dummy_spec_batch):
    batch = {
        "iq": dummy_iq_batch,
        "spectrogram": dummy_spec_batch,
        "label": torch.tensor([0, 1, 2, 3]),
    }

    inputs, labels = prepare_model_inputs(batch, torch.device("cpu"), "dual")

    assert len(inputs) == 2
    assert inputs[0].shape == dummy_iq_batch.shape
    assert inputs[1].shape == dummy_spec_batch.shape
    assert labels.shape[0] == dummy_iq_batch.shape[0]
