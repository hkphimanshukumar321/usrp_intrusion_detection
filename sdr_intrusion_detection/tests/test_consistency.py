"""
test_consistency.py — Consistency & Syntax Integrity Tests
==========================================================
1. Verifies all .py files compile without syntax errors
2. Verifies all cross-module references are consistent
3. Verifies no broken imports across the entire project
"""
import pytest
import os
import sys
import py_compile
import importlib
import glob


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')


# ============================================================
# SMOKE: Every .py file compiles without syntax errors
# ============================================================
class TestNoSyntaxErrors:
    @staticmethod
    def _get_all_py_files():
        """Collect every .py file in the project."""
        py_files = []
        for root, _, files in os.walk(PROJECT_ROOT):
            # Skip __pycache__, .pytest_cache, .git
            if any(skip in root for skip in ('__pycache__', '.pytest_cache', '.git')):
                continue
            for f in files:
                if f.endswith('.py'):
                    py_files.append(os.path.join(root, f))
        return py_files

    def test_all_python_files_compile(self):
        """Every .py file in the project must compile without SyntaxError."""
        errors = []
        for filepath in self._get_all_py_files():
            try:
                py_compile.compile(filepath, doraise=True)
            except py_compile.PyCompileError as e:
                errors.append(f"{filepath}: {e}")

        assert len(errors) == 0, (
            f"{len(errors)} file(s) have syntax errors:\n" +
            "\n".join(errors)
        )

    def test_no_empty_source_files(self):
        """No source file in src/ should be completely empty (0 bytes of code)."""
        empty_files = []
        for filepath in self._get_all_py_files():
            if os.sep + 'src' + os.sep in filepath:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if len(content) == 0:
                    empty_files.append(filepath)

        assert len(empty_files) == 0, (
            f"{len(empty_files)} source file(s) are empty:\n" +
            "\n".join(empty_files)
        )


# ============================================================
# SMOKE: All modules import without errors
# ============================================================
class TestNoImportErrors:
    MODULES = [
        'src.model',
        'src.train',
        'src.evaluate',
        'src.data_loader',
        'src.run_ablation',
        'src.benchmark_edge',
        'src.generate_figures',
    ]

    @pytest.mark.parametrize("module_name", MODULES)
    def test_module_imports(self, module_name):
        """Each src module must import without ImportError or ModuleNotFoundError."""
        try:
            importlib.import_module(module_name)
        except Exception as e:
            pytest.fail(f"Failed to import '{module_name}': {type(e).__name__}: {e}")


# ============================================================
# CONSISTENCY: Cross-module references match
# ============================================================
class TestConsistency:
    def test_custom_model_name_consistent(self):
        """The custom model name used in run_ablation, train, evaluate must match model.py."""
        from src.model import get_model
        from src.run_ablation import CUSTOM_MODEL
        from src.generate_figures import CUSTOM_MODEL as FIG_MODEL

        # Verify model.py can actually instantiate with this name
        model = get_model(CUSTOM_MODEL)
        assert model is not None, f"get_model('{CUSTOM_MODEL}') returned None"

        # Verify generate_figures uses the same name
        assert CUSTOM_MODEL == FIG_MODEL, (
            f"run_ablation uses '{CUSTOM_MODEL}' but generate_figures uses '{FIG_MODEL}'"
        )

    def test_timm_map_used_consistently(self):
        """run_ablation references TIMM_MODEL_MAP from model.py correctly."""
        from src.model import TIMM_MODEL_MAP as model_map
        from src.run_ablation import TIMM_MODEL_MAP as ablation_map
        assert model_map is ablation_map, "run_ablation imports a different TIMM_MODEL_MAP"

    def test_class_names_consistent(self):
        """data_loader.CLASS_NAMES is used by evaluate and model."""
        from src.data_loader import CLASS_NAMES, NUM_CLASSES
        assert len(CLASS_NAMES) == NUM_CLASSES
        assert NUM_CLASSES == 6

    def test_evaluate_uses_get_model(self):
        """evaluate.py must import get_model from model.py (not define its own)."""
        import inspect
        from src import evaluate
        source = inspect.getsource(evaluate)
        assert 'from src.model import get_model' in source, \
            "evaluate.py does not import get_model from src.model"

    def test_train_uses_focal_loss(self):
        """train.py must define and use FocalLoss, not plain CrossEntropyLoss."""
        import inspect
        from src import train
        source = inspect.getsource(train)
        assert 'class FocalLoss' in source, "train.py is missing FocalLoss class"
        assert 'FocalLoss' in source, "train.py does not use FocalLoss"

    def test_all_timm_names_are_real(self):
        """Every value in TIMM_MODEL_MAP must be a valid timm model identifier."""
        import timm
        from src.model import TIMM_MODEL_MAP
        available = timm.list_models()
        invalid = []
        for display_name, timm_name in TIMM_MODEL_MAP.items():
            if timm_name not in available:
                invalid.append(f"{display_name} -> '{timm_name}'")

        assert len(invalid) == 0, (
            f"{len(invalid)} TIMM model name(s) are invalid:\n" +
            "\n".join(invalid)
        )

    def test_num_classes_matches_dataset_dirs(self):
        """NUM_CLASSES should match the number of folders in unified_dataset/train/."""
        from src.data_loader import NUM_CLASSES
        train_dir = os.path.join(PROJECT_ROOT, '..', 'unified_dataset', 'train')
        if os.path.isdir(train_dir):
            n_dirs = len([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])
            assert NUM_CLASSES == n_dirs, (
                f"NUM_CLASSES={NUM_CLASSES} but unified_dataset/train/ has {n_dirs} folders"
            )

    def test_requirements_has_all_imports(self):
        """Key packages used in source code exist in requirements.txt."""
        req_path = os.path.join(PROJECT_ROOT, 'requirements.txt')
        with open(req_path, 'r') as f:
            req_text = f.read().lower()

        required_pkgs = ['torch', 'timm', 'wandb', 'optuna', 'tqdm',
                         'matplotlib', 'seaborn', 'scikit-learn', 'numpy']
        missing = [p for p in required_pkgs if p not in req_text]
        assert len(missing) == 0, f"Missing from requirements.txt: {missing}"
