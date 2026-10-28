"""Predict and export per-detection logits with a trained checkpoint."""
import argparse
from cli_utils import read_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--source", required=True, help="Image, folder, or image-list file")
    parser.add_argument("--names", required=True, help="ID names followed by negative names")
    parser.add_argument("--n-id-cls", type=int, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=20, help="Total inference batch size")
    parser.add_argument("--conf", type=float, default=0.001, help="Export threshold, below evaluation threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--name", default="predict")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    names = read_names(args.names)
    if args.n_id_cls <= 0 or len(names) != args.n_id_cls + 20:
        parser.error("Names must contain ID classes followed by exactly 20 negative labels")
    import torch
    from ultralytics import YOLOWorld
    device = args.device if torch.cuda.is_available() else "cpu"
    model = YOLOWorld(args.weight)
    model.set_classes(names)
    model.model.n_id_cls = args.n_id_cls
    model.predict(source=args.source, device=device, batch=args.batch, names=names,
                  imgsz=args.imgsz, conf=args.conf, project=args.project, name=args.name,
                  dataset=args.dataset, model_name="negas", n_id_cls=args.n_id_cls,
                  save=args.save, save_json=True)
    print(f"Predictions: {model.predictor.save_dir / 'predictions.json'}")


if __name__ == "__main__":
    main()
