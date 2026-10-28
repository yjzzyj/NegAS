"""Validate a trained YOLO-World checkpoint on a YOLO dataset."""
import argparse
from cli_utils import read_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--names", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=20, help="Total validation batch size")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--dataset", default="custom")
    parser.add_argument("--name", default="val")
    parser.add_argument("--save-json", action="store_true")
    args = parser.parse_args()
    names = read_names(args.names)
    import torch
    from ultralytics import YOLOWorld
    model = YOLOWorld(args.weight)
    model.set_classes(names)
    model.val(data=args.data, device=args.device if torch.cuda.is_available() else "cpu",
              batch=args.batch, names=names, project=args.project, name=args.name,
              dataset=args.dataset, save_json=args.save_json)


if __name__ == "__main__":
    main()
