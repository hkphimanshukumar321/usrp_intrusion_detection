import pytest
import os
import torch
from src.data_loader import get_dataloaders, CLASS_NAMES

def test_class_names():
    assert len(CLASS_NAMES) == 6
    assert 'Machine' in CLASS_NAMES
    assert 'Benign' in CLASS_NAMES

def test_dataloader_fails_on_missing_dir():
    with pytest.raises(FileNotFoundError):
        get_dataloaders(dataset_dir="fake_dir_12345")

# Note: We cannot natively test get_dataloaders() without a mock dataset 
# on disk because ImageFolder requires physical directories. 
# We rely on the FileNotFoundError test to prove the directory logic triggers.
