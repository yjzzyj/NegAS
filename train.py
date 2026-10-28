"""Train the fixed NegAS paper recipe. Batch is the total across devices."""
import argparse
from cli_utils import read_names, str_to_bool


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--names", required=True, help="ID labels followed by the 20 curated negative labels")
    parser.add_argument("--n_id_cls", type=int, default=20, help="Dataset ID category count; VOC=20, BDD=10")
    parser.add_argument("--weight", required=True, help="Pretrained YOLO-World-v2 or cleaned last.pt for resume")
    parser.add_argument("--cfg", default="yolov8x-worldv2-promptlearner.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=160, help="Total batch size (paper: 160 on 4 GPUs)")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", type=str_to_bool, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--dataset", default="negas")
    parser.add_argument("--name", default="train")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    names = read_names(args.names)
    if args.n_id_cls <= 0 or len(names) != args.n_id_cls + 20:
        parser.error("Provide the dataset's ID labels followed by exactly 20 negative labels")
    if args.batch <= 0:
        parser.error("Batch size must be positive")
    import torch
    from ultralytics import YOLOWorld
    device = args.device if torch.cuda.is_available() else "cpu"
    model = YOLOWorld(args.weight if args.resume else args.cfg, n_id_cls=args.n_id_cls)
    if not args.resume:
        model.load(args.weight)
    model.set_classes(names)
    model.train(data=args.data, names=names, n_id_cls=args.n_id_cls,
                device=device, epochs=args.epochs, batch=args.batch,
                imgsz=args.imgsz, workers=args.workers, amp=args.amp,
                optimizer="AdamW", lr0=args.lr0, resume=args.resume,
                project=args.project, dataset=args.dataset, name=args.name)


if __name__ == "__main__":
    main()
