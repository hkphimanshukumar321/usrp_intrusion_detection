import os
import shutil
import pytest
from src.run_ablation import run_full_pipeline

def test_pipeline_smoke(tmp_path):
    """
    Run a full end-to-end test of the pipeline inside a temporary directory.
    Uses 'quick=True' which limits data generation to 2 seconds and epochs to 10.
    """
    data_dir = tmp_path / "dataset"
    results_dir = tmp_path / "results"
    figures_dir = tmp_path / "figures"
    
    # We expect these directories to be created automatically or by the script without crashing
    try:
        run_full_pipeline(
            data_dir=str(data_dir),
            results_dir=str(results_dir),
            figures_dir=str(figures_dir),
            duration=0.1, # extremely small for test speed
            snr_db=20.0,
            epochs=1,
            batch_size=32,
            quick=True,
            tune=False,
            n_folds=2    # Overwrite default 5 for speed
        )
    except Exception as e:
        pytest.fail(f"End-to-end pipeline failed with exception: {e}")
    
    # Verify outputs were actually generated
    assert os.path.exists(data_dir), "Data generation failed"
    assert os.path.exists(results_dir), "Training failed to output models"
    assert os.path.exists(figures_dir), "Evaluate phase failed to generate figures"

    # Make sure we got the specific JSON ablation results
    assert os.path.exists(os.path.join(results_dir, "ablation_results.json")), "Ablation results JSON missing"
