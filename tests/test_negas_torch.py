"""Real CPU tensor tests; a small text encoder avoids downloading CLIP weights."""
import unittest
from unittest.mock import patch
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install PyTorch and the package dependencies for integration tests")
class NegASIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ultralytics.nn.tasks import WorldModel
        torch.set_num_threads(1)
        torch.manual_seed(13)

        class TextEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.token_embedding = nn.Embedding(128, 512)
                self.model.projection = nn.Linear(512, 512)

            def tokenize(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                values = torch.zeros(len(texts), 77, dtype=torch.long)
                for i, text in enumerate(texts):
                    values[i, 1:4] = torch.tensor([1, 2, 3])
                    values[i, 4:] = sum(map(ord, text)) % 127
                return values

            def encode_text_emb(self, embeddings, tokens):
                return torch.nn.functional.normalize(self.model.projection(embeddings.mean(1)), dim=-1)

        cls.TextEncoder = TextEncoder
        with patch("ultralytics.nn.text_model.build_text_model", return_value=TextEncoder()):
            cls.model = WorldModel("yolov8n-worldv2-promptlearner.yaml", nc=22, n_id_cls=2, verbose=False)
        cls.model.set_classes(["cat", "dog"] + [f"negative_{i}" for i in range(20)])
        from types import SimpleNamespace
        cls.model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    def test_only_prompt_parameters_and_frozen_statistics_update(self):
        model = self.model.train()
        self.assertEqual([name for name, p in model.named_parameters() if p.requires_grad], ["prompt_learner.prompt"])
        self.assertEqual(model.prompt_learner.prompt.shape, (3, 512))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        batch = dict(img=torch.randn(2, 3, 64, 96), batch_idx=torch.tensor([0., 1.]),
                     cls=torch.tensor([[0.], [1.]]), bboxes=torch.tensor([[.5, .5, .4, .4], [.5, .5, .3, .3]]))
        optimizer = torch.optim.AdamW(model.prompt_learner.parameters(), lr=.01)
        optimizer.zero_grad()
        losses, items = model.loss(batch)
        self.assertEqual(items.shape, (4,))
        self.assertTrue(torch.isfinite(losses).all())
        losses.sum().backward()
        self.assertGreater(model.prompt_learner.prompt.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for n, p in model.named_parameters() if n != "prompt_learner.prompt"))
        optimizer.step()
        for name, value in model.state_dict().items():
            if name != "prompt_learner.prompt":
                self.assertTrue(torch.equal(value, before[name]), name)
        self.assertFalse(torch.equal(model.prompt_learner.prompt, before["prompt_learner.prompt"]))

    def test_trainer_optimizer_contains_only_shared_prompt(self):
        from ultralytics.models.yolo.world.train import WorldTrainer
        trainer = WorldTrainer.__new__(WorldTrainer)
        optimizer = trainer.build_optimizer(self.model)
        parameters = [p for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(parameters), 1)
        self.assertIs(parameters[0], self.model.prompt_learner.prompt)

    def test_inference_runs_neck_once_training_twice(self):
        from ultralytics.nn.modules import C2fAttn
        model = self.model
        attention = [module for module in model.model if isinstance(module, C2fAttn)]
        counts = [0] * len(attention)
        def hook(index):
            def count(*args):
                counts[index] += 1
            return count
        handles = [module.register_forward_hook(hook(i)) for i, module in enumerate(attention)]
        try:
            image = torch.randn(1, 3, 64, 96)
            with torch.no_grad():
                model.train()
                train = model.predict(image)
                self.assertEqual(counts, [2] * len(attention))
                self.assertIsNotNone(train["ood"])
                counts[:] = [0] * len(attention)
                model.eval()
                prediction = model.predict(image)
                self.assertEqual(counts, [1] * len(attention))
                self.assertIsNone(prediction["ood"])
                self.assertEqual(prediction["preds"][0].shape[1], 6)  # four coordinates plus two ID scores
                self.assertEqual(prediction["logits"].shape[1], 22)
        finally:
            for handle in handles:
                handle.remove()

    def test_prediction_export_preserves_all_logits(self):
        import tempfile
        import json
        import numpy as np
        from pathlib import Path
        from ultralytics import YOLOWorld
        with patch("ultralytics.nn.text_model.build_text_model", return_value=self.TextEncoder()):
            wrapper = YOLOWorld("yolov8n-worldv2-promptlearner.yaml", n_id_cls=2, verbose=False)
        wrapper.set_classes(["cat", "dog"] + [f"negative_{i}" for i in range(20)])
        with tempfile.TemporaryDirectory() as tmp:
            results = wrapper.predict(source=np.zeros((64, 96, 3), dtype=np.uint8), device="cpu",
                                      conf=0., imgsz=64, project=tmp, dataset="test", model_name="negas",
                                      name="predict", save=False, save_json=True, verbose=False)
            records = json.loads((wrapper.predictor.save_dir / "predictions.json").read_text())
            self.assertTrue(records)
            self.assertTrue(all(len(row["logit"]) == 22 for row in records))
            self.assertTrue(all(row["class"] in (1, 2) for row in records))
            self.assertEqual(len(results), 1)

    def test_gt_background_mask_rectangles_and_empty_images(self):
        from ultralytics.utils.negas import background_mask
        features = [torch.zeros(2, 1, 2, 4), torch.zeros(2, 1, 4, 8)]
        boxes = torch.tensor([[[9., 1., 17., 9.]], [[0., 0., 0., 0.]]])
        valid = torch.tensor([[[True]], [[False]]])
        mask = background_mask(features, [8, 4], boxes, valid)
        self.assertEqual(mask.shape, (2, 40))
        expected = torch.tensor([True, False, False, True] * 2)
        self.assertTrue(torch.equal(mask[0, :8], expected))
        self.assertTrue(mask[1].all())
        empty = background_mask(features, [8, 4], boxes[:, :0], valid[:, :0])
        self.assertTrue(empty.all())

    def test_auxiliary_coefficient_and_no_foreground_gradient(self):
        from ultralytics.utils.loss import NegASDetectionLoss
        from types import SimpleNamespace
        criterion = NegASDetectionLoss.__new__(NegASDetectionLoss)
        criterion.reg_max = 1
        criterion.stride = [1]
        criterion.hyp = SimpleNamespace(cls=.5)
        criterion.bce = nn.BCEWithLogitsLoss(reduction="none")
        features = [torch.zeros(1, 5, 2, 2)]
        targets = torch.zeros(1, 4, 1)
        boxes = torch.tensor([[[0., 0., 1., 1.]]])
        valid = torch.ones(1, 1, 1, dtype=torch.bool)
        criterion._compute_loss = lambda *args: (torch.zeros(3), targets, 1., boxes, valid, features)
        auxiliary = torch.zeros(1, 5, 2, 2, requires_grad=True)
        loss, _ = criterion({"preds": features, "ood": [auxiliary]}, {})
        expected = 3 * .5 * 3 * torch.log(torch.tensor(2.))
        self.assertAlmostEqual(loss[3].item(), expected.item(), places=6)
        loss.sum().backward()
        self.assertEqual(auxiliary.grad[0, 4, 0, 0].item(), 0.)
        self.assertGreater(auxiliary.grad[0, 4, 1, 1].item(), 0.)
        boxes[:] = torch.tensor([0., 0., 2., 2.])
        loss, _ = criterion({"preds": features, "ood": [auxiliary]}, {})
        self.assertEqual(loss[3].item(), 0.)


if __name__ == "__main__":
    unittest.main()
