"""Release checks requiring only the standard library (and NumPy for scoring)."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cli_utils import str_to_bool, read_names
from train import build_parser
from eval_ood import detection_scores, load_predictions



class ReleaseChecks(unittest.TestCase):
    def test_all_python_compiles(self):
        for path in ROOT.rglob("*.py"):
            with self.subTest(path=str(path.relative_to(ROOT))):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_packaging_configuration(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest("tomllib requires Python 3.11 or later")
        with (ROOT / "pyproject.toml").open("rb") as handle:
            config = tomllib.load(handle)
        self.assertEqual(config["project"]["name"], "ultralytics")
        self.assertIn("scikit-learn>=1.0", config["project"]["dependencies"])
        self.assertTrue((ROOT / config["project"]["readme"]).is_file())

    def test_help_without_torch(self):
        for script in ["train.py", "predict.py", "val.py", "eval_ood.py", "negative_label_generation/generate.py"]:
            result = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)

    def test_fixed_recipe_cli(self):
        args = build_parser().parse_args(["--data", "dataset.yaml", "--names", "classes.names", "--weight", "world.pt"])
        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.batch, 160)
        self.assertEqual(args.lr0, 0.01)
        removed = {"using_fg_mask", "invert_fg_mask", "copy_neck", "using_noise_for_ood",
                   "noise_std", "ood_loss_weight", "ood_loss", "prompt_learner", "ctx_init_in_str",
                   "text_processor", "mcm_loss_type", "freeze", "stage"}
        self.assertFalse(removed.intersection(vars(args)))

    def test_cpu_training_forwards_total_batch(self):
        from unittest.mock import Mock, patch
        import types
        from train import main
        with tempfile.TemporaryDirectory() as tmp:
            names = Path(tmp) / "classes.names"
            classes = ["cat"] + [f"negative_{i}" for i in range(20)]
            names.write_text("\n".join(classes))
            model = Mock()
            fake_torch = types.ModuleType("torch")
            fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
            fake_ultralytics = types.ModuleType("ultralytics")
            fake_ultralytics.YOLOWorld = Mock(return_value=model)
            argv = ["train.py", "--data", "data.yaml", "--names", str(names), "--weight", "world.pt",
                    "--n_id_cls", "1", "--device", "cpu", "--batch", "7"]
            with patch.dict(sys.modules, {"torch": fake_torch, "ultralytics": fake_ultralytics}), patch.object(sys, "argv", argv):
                main()
            self.assertEqual(model.train.call_args.kwargs["batch"], 7)
            self.assertEqual(model.train.call_args.kwargs["device"], "cpu")
            self.assertEqual(model.train.call_args.kwargs["optimizer"], "AdamW")
            self.assertNotIn("myargs", model.train.call_args.kwargs)
            model.set_classes.assert_called_once_with(classes)

    def test_no_hidden_ablation_flags(self):
        recipe = json.loads((ROOT / "docs/paper_recipe.json").read_text())
        removed = {name.removeprefix("--") for name in recipe["removed_training_options"]}
        # Generic framework features may share names; restrict to the method implementation.
        banned = removed - {"freeze", "model_name", "profile", "optimizer", "lrf", "momentum", "prompt_learner",
                            "weight_decay", "warmup_epochs", "warmup_momentum", "patience", "close_mosaic",
                            "iou_ood", "multi_label", "cfg_or_weight", "save_period", "tal_topk"}
        for relative in ["ultralytics/nn/tasks.py", "ultralytics/utils/loss.py", "ultralytics/nn/modules/block.py"]:
            tree = ast.parse((ROOT / relative).read_text())
            method_classes = {"WorldModel", "NegASDetectionLoss", "C2fAttn", "MaxSigmoidAttnBlock"}
            roots = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in method_classes]
            attributes = {node.attr for root in roots for node in ast.walk(root) if isinstance(node, ast.Attribute)}
            self.assertFalse(attributes & banned, f"{relative}: {attributes & banned}")

    def test_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "names"
            path.write_text("cat\ndog\n")
            self.assertEqual(read_names(path), ["cat", "dog"])
            path.write_text("cat\ncat\n")
            with self.assertRaises(ValueError):
                read_names(path)

    def test_no_private_imports_or_historical_paths(self):
        for path in ROOT.rglob("*.py"):
            if path.parent.name == "tests":
                continue
            text = path.read_text()
            self.assertNotIn("from pyzyj", text, str(path))
            self.assertNotIn("/data/zyj", text, str(path))
            self.assertNotIn("/home/zyj", text, str(path))


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, "NumPy unavailable")
class ScoreChecks(unittest.TestCase):
    def test_negs_matches_original_formula(self):
        rng = np.random.default_rng(7)
        logits = rng.normal(size=(30, 40))
        expected = (1 / (1 + np.exp(-logits[:, :20]))).max(axis=1) - (1 / (1 + np.exp(-logits[:, 20:]))).max(axis=1)
        actual = detection_scores([{"logit": x.tolist()} for x in logits], 20)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_class_boundary_and_extreme_values(self):
        predictions = [{"logit": [1000, -1000, 0]}, {"logit": [-1000, 0, 1000]}]
        np.testing.assert_allclose(detection_scores(predictions, 2), [0.5, -0.5])
        for logits, boundary in [([], 1), ([0, 1], 2), ([0, float("nan")], 1)]:
            with self.assertRaises(ValueError):
                detection_scores([{"logit": logits}], boundary)
        with self.assertRaises(ValueError):
            detection_scores([], 2)

    def test_fpr95_convention(self):
        from detection_util import fpr_and_fdr_at_recall
        labels = np.array([1] * 20 + [0] * 20)
        self.assertEqual(fpr_and_fdr_at_recall(labels, np.array([1] * 20 + [0] * 20)), 0)
        self.assertEqual(fpr_and_fdr_at_recall(labels, np.ones(40)), 1)
        self.assertEqual(fpr_and_fdr_at_recall(labels, np.array([0] * 20 + [1] * 20)), 1)

    def test_prediction_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.json"
            path.write_text(json.dumps([{"score": 0.2}, {"conf": 0.7}, {"score": 0.5}]))
            kept, total = load_predictions(path, 0.5)
            self.assertEqual(kept, [{"conf": 0.7}])
            self.assertEqual(total, 3)


if __name__ == "__main__":
    unittest.main()
