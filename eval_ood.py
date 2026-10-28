"""Evaluate ID-positive AUROC, AUPR and FPR95 from exported detection logits."""
import argparse
import json
from pathlib import Path


def detection_scores(predictions, n_id_cls):
    """Return ID confidence; NegS is max ID sigmoid minus max negative sigmoid."""
    import numpy as np
    if not predictions:
        raise ValueError("No detections remain after confidence filtering")
    logits = np.asarray([p["logit"] for p in predictions], dtype=np.float64)
    if logits.ndim != 2 or not 0 < n_id_cls <= logits.shape[1]:
        raise ValueError("Logits must be a rectangular matrix with a valid ID class count")
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite")
    # Stable sigmoid, including extreme logits.
    exp_abs = np.exp(-np.abs(logits))
    probabilities = np.where(logits >= 0, 1 / (1 + exp_abs), exp_abs / (1 + exp_abs))
    known = probabilities[:, :n_id_cls].max(axis=1)
    if logits.shape[1] == n_id_cls:
        raise ValueError("NegS requires negative-label logits after the ID logits")
    return known - probabilities[:, n_id_cls:].max(axis=1)


def load_predictions(path, threshold):
    with Path(path).open(encoding="utf-8") as handle:
        predictions = json.load(handle)
    if not isinstance(predictions, list):
        raise ValueError("Expected a list of detection dictionaries")
    filtered = []
    for prediction in predictions:
        confidence = prediction.get("conf", prediction.get("score"))
        if confidence is None:
            raise ValueError("Each detection must contain conf or score")
        if confidence > threshold:
            filtered.append(prediction)
    return filtered, len(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-json", required=True, help="Predictions on the ID dataset")
    parser.add_argument("--ood-json", required=True, help="Predictions on the OOD dataset")
    parser.add_argument("--n-id-cls", type=int, required=True)
    parser.add_argument("--id-conf", type=float, required=True, help="ID validation threshold, in [0, 1)")
    parser.add_argument("--ood-conf", type=float, default=None, help="Defaults to the ID threshold")
    parser.add_argument("--output", default=None, help="Optional metrics JSON")
    args = parser.parse_args()
    if args.ood_conf is None:
        args.ood_conf = args.id_conf
    if not all(0 <= threshold < 1 for threshold in [args.id_conf, args.ood_conf]):
        parser.error("Confidence thresholds must be in [0, 1)")
    from detection_util import get_measures
    id_predictions, id_total = load_predictions(args.id_json, args.id_conf)
    ood_predictions, ood_total = load_predictions(args.ood_json, args.ood_conf)
    known = detection_scores(id_predictions, args.n_id_cls)
    unknown = detection_scores(ood_predictions, args.n_id_cls)
    auroc, aupr, fpr = get_measures(known, unknown)
    metrics = dict(auroc=float(auroc), aupr=float(aupr), fpr95=float(fpr),
                   id_kept=len(known), id_total=id_total, ood_kept=len(unknown), ood_total=ood_total,
                   method="negs", n_id_cls=args.n_id_cls, id_conf=args.id_conf,
                   ood_conf=args.ood_conf, positive_class="ID")
    result = json.dumps(metrics, indent=2, allow_nan=False)
    print(result)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
