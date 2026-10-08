"""Dự đoán OBB cho các lượt "dùng được" chưa có nhãn, để người rà/sửa ở vòng sau.

    .venv/Scripts/python code/train/predict.py --weights runs.nosync/all_y26n_aug/weights/best.pt

Kết quả trong runs.nosync/predict_<tên run>/:
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
from app import canonical, corners  # noqa: E402

COLORS = {0: (0, 255, 255), 1: (255, 0, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--all", action="store_true", help="dự đoán cả 25 ảnh đã gán nhãn")
    args = ap.parse_args()

    selection = json.loads((ROOT / "code/obb_labeler/selection.json").read_text(encoding="utf-8"))
    labeled = {im["id"] for im in selection["images"]}
    with open(ROOT / "code/image_sorter/results.nosync/labels.csv", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "usable"
                and (args.all or r["group_id"] not in labeled)]

    run = args.weights.resolve().parents[1].name
    out = ROOT / "runs.nosync" / f"predict_{run}"
    (out / "vis").mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    results = {}
    for row in rows:
        img = cv2.imread(str(ROOT / row["image_path"]))
        pred = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        boxes = []
        for (cx, cy, w, h, r), cls, conf in zip(pred.obb.xywhr.tolist(), pred.obb.cls.tolist(),
                                                pred.obb.conf.tolist()):
            box = canonical({"cx": cx, "cy": cy, "w": w, "h": h, "angle": float(np.degrees(r))})
            box.update(class_id=int(cls), conf=round(conf, 4))
            box["corners"] = corners(box)
            boxes.append(box)
            cv2.polylines(img, [np.array(box["corners"], np.int32)], True, COLORS[int(cls)], 4)
            if cls == 1:
                cv2.putText(img, f"{box['angle']:.0f}", (int(cx) - 20, int(cy)), 0, 1.2, (0, 255, 0), 3)
        results[row["group_id"]] = {"image_path": row["image_path"], "objects": boxes}
        n_bag = sum(b["class_id"] == 1 for b in boxes)
        cv2.putText(img, f"{row['state']} | bag={n_bag}", (20, 60), 0, 2, (0, 255, 0), 4)
        cv2.imwrite(str(out / "vis" / (row["group_id"].replace("/", "__") + ".jpg")),
                    cv2.resize(img, None, fx=0.5, fy=0.5))
    (out / "predictions.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(results)} ảnh -> {out}")


if __name__ == "__main__":
    main()
