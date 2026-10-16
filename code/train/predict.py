"""Dự đoán OBB cho các lượt "dùng được" chưa có nhãn, để người rà/sửa ở vòng sau.

    .venv/Scripts/python code/train/predict.py --weights runs/all_y26n_aug/weights/best.pt
    .venv/Scripts/python code/train/predict.py --weights a/last.pt b/last.pt --tta --imgsz 960

Nhiều --weights và/hoặc --tta: gộp các lần chạy bằng WBF như ensemble.py (TTA 8 hướng xoay/lật).
Kết quả trong runs/predict_<tên run>[_ens][_tta]/:
- predictions.json: mỗi ảnh một danh sách box cùng schema với annotations.json
  (cx, cy, w, h, angle, corners theo pixel gốc; góc theo quy ước le90 của labeler) + conf.
- vis/<id>.jpg: ảnh vẽ box để xem nhanh.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/obb_labeler"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import canonical, corners  # noqa: E402
from ensemble import TTA_SETS, predict_tta, wbf  # noqa: E402
from pipeline import to_xywhr  # noqa: E402

COLORS = {0: (0, 255, 255), 1: (255, 0, 255)}


def detect(models, img, imgsz, tta=False, conf=0.35, fuse_iou=0.55):
    """Box OBB theo quy ước labeler (cx, cy, w, h, angle le90, corners, class_id, conf). Nhiều model
    hoặc TTA thì gộp WBF như ensemble.py."""
    runs = [d for m in models for d in predict_tta(m, img, imgsz, TTA_SETS["tta8" if tta else "none"])]
    pred = runs[0] if len(runs) == 1 else wbf(runs, fuse_iou)
    keep = pred["conf"] >= conf
    boxes = []
    for (cx, cy, w, h, r), cls, c in zip(to_xywhr(pred["poly"][keep]), pred["cls"][keep], pred["conf"][keep]):
        box = canonical({"cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h),
                         "angle": float(np.degrees(r))})
        box.update(class_id=int(cls), conf=round(float(c), 4))
        box["corners"] = corners(box)
        boxes.append(box)
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path, nargs="+")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--tta", action="store_true", help="thêm 7 hướng xoay/lật, gộp WBF")
    ap.add_argument("--fuse-iou", type=float, default=0.55)
    ap.add_argument("--all", action="store_true", help="dự đoán cả 25 ảnh đã gán nhãn")
    args = ap.parse_args()

    selection = json.loads((ROOT / "code/obb_labeler/selection.json").read_text(encoding="utf-8"))
    labeled = {im["id"] for im in selection["images"]}
    with open(ROOT / "code/image_sorter/results/labels.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "usable"
                and (args.all or r["group_id"] not in labeled)]

    run = args.weights[0].resolve().parents[1].name
    run += "_ens" if len(args.weights) > 1 else ""
    run += "_tta" if args.tta else ""
    out = ROOT / "runs" / f"predict_{run}"
    (out / "vis").mkdir(parents=True, exist_ok=True)
    models = [YOLO(w) for w in args.weights]
    results = {}
    for row in rows:
        img = cv2.imread(str(ROOT / row["image_path"]))
        boxes = detect(models, img, args.imgsz, args.tta, args.conf, args.fuse_iou)
        for box in boxes:
            cv2.polylines(img, [np.array(box["corners"], np.int32)], True, COLORS[box["class_id"]], 4)
            if box["class_id"] == 1:
                cv2.putText(img, f"{box['angle']:.0f}", (int(box["cx"]) - 20, int(box["cy"])), 0, 1.2,
                            (0, 255, 0), 3)
        results[row["group_id"]] = {"image_path": row["image_path"], "objects": boxes}
        n_bag = sum(b["class_id"] == 1 for b in boxes)
        cv2.putText(img, f"{row['state']} | bag={n_bag}", (20, 60), 0, 2, (0, 255, 0), 4)
        cv2.imwrite(str(out / "vis" / (row["group_id"].replace("/", "__") + ".jpg")),
                    cv2.resize(img, None, fx=0.5, fy=0.5))
    (out / "predictions.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(results)} ảnh -> {out}")


if __name__ == "__main__":
    main()
