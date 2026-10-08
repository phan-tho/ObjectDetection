"""Fine-tune YOLO-OBB nano trên dataset do build_dataset.py tạo.

    .venv/Scripts/python code/train/train.py --data fold_red --model yolo26n-obb.pt --name red_y26n_aug

Augment hình học làm online ở đây (ảnh nhìn từ trên xuống nên xoay tự do, lật cả hai chiều).
Màu/ánh sáng chủ yếu đã làm offline, online chỉ thêm HSV nhẹ.
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="tên thư mục trong datasets.nosync/")
    ap.add_argument("--model", default="yolo26n-obb.pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="GPU id, vd 0 hoặc 1 (mặc định tự chọn)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-geo", action="store_true", help="tắt augment hình học online (ablation)")
    args = ap.parse_args()

    geo = dict(degrees=180, flipud=0.5, fliplr=0.5, scale=0.5, translate=0.15,
               perspective=0.0003, mosaic=1.0, close_mosaic=10)
    if args.no_geo:
        geo = dict(degrees=0, flipud=0, fliplr=0, scale=0, translate=0, perspective=0, mosaic=0)
    model = YOLO(args.model)
    model.train(data=str(ROOT / "datasets.nosync" / args.data / "data.yaml"), epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device,
                project=str(ROOT / "runs.nosync"), name=args.name, exist_ok=True, patience=0, seed=0, deterministic=False,
                hsv_h=0.015, hsv_s=0.5, hsv_v=0.3, plots=True, **geo)
    best = YOLO(ROOT / "runs.nosync" / args.name / "weights" / "best.pt")
    m = best.val(data=str(ROOT / "datasets.nosync" / args.data / "data.yaml"), imgsz=args.imgsz,
                 split="val", device=args.device, project=str(ROOT / "runs.nosync"),
                 name=args.name + "_val", exist_ok=True)
    result = {"run": args.name, "data": args.data, "model": args.model, "imgsz": args.imgsz,
              "mAP50": m.box.map50, "mAP50-95": m.box.map,
              "per_class_mAP50-95": dict(zip(m.names.values(), map(float, m.box.maps))),
              "precision": m.box.mp, "recall": m.box.mr}
    print(json.dumps(result, indent=2))
    with open(ROOT / "runs.nosync" / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
