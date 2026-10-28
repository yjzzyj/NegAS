# Ultralytics AGPL-3.0; see LICENSE.
"""Prompt-only closed-set NegAS trainer."""
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import de_parallel


class WorldTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True, **kwargs):
        model = WorldModel(cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
                           ch=self.data["channels"], nc=self.data["nc"],
                           n_id_cls=self.args.n_id_cls, verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        names = [value.split("/", 1)[0] for value in self.data["names"].values()]
        model.set_classes(names)
        model.names = self.data["names"]
        return model

    def get_validator(self):
        validator = super().get_validator()
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "ood_loss"
        return validator

    def build_optimizer(self, model, name="AdamW", lr=0.01, momentum=0.937, decay=0.0005, iterations=1e5):
        import torch
        detector = de_parallel(model)
        detector.freeze_detector()
        return torch.optim.AdamW(detector.prompt_learner.parameters(), lr=lr,
                                 betas=(momentum, 0.999), weight_decay=decay)
