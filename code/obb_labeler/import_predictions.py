"""Nạp dự đoán của model vào labeler làm nhãn nháp, để người sửa thay vì vẽ lại từ đầu.

    .venv/Scripts/python code/obb_labeler/import_predictions.py runs/predict_<run>/predictions.json --dry-run
    .venv/Scripts/python code/obb_labeler/import_predictions.py runs/predict_<run>/predictions.json

- Thêm ảnh chưa có vào selection.json (thông tin lấy từ image_sorter/results/labels.csv).
- sequence_group: nhóm của ảnh đã có nhãn gần nhất về thời gian, cùng nguồn. Các lượt chụp mới đều
  nằm xen giữa các lượt đã có nhãn (cách nhau vài giây), nên phải chung nhóm để chia fold không lọt.
- Record nháp (reviewed = false) qua validate() của app.py: bao gắn với pallet chứa tâm của nó (pallet
  nhỏ nhất nếu nhiều), pallet mục tiêu = pallet có tâm gần tâm ảnh nhất.
- Không đụng ảnh đã có record (kể cả nháp). Chạy lại an toàn.
"""
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app import HERE, ROOT, validate

FLAGS = ["overlap", "rotation", "deformed", "lighting", "bag_color", "pallet_offset", "overhang", "bottom_visible"]


def nearest_group(row, labeled):
    t = datetime.fromisoformat(row["captured_at"])
    same = [im for im in labeled if im["id"].split("/")[0] == row["source"]]
    best = min(same, key=lambda im: abs((datetime.fromisoformat(im["captured_at"]) - t).total_seconds()))
    return best["sequence_group"]


def draft(objects, width, height):
    pallets = sorted((o for o in objects if o["class_id"] == 0), key=lambda o: -o["conf"])
    bags = sorted((o for o in objects if o["class_id"] == 1), key=lambda o: -o["conf"])
    boxes = []
    for i, o in enumerate(pallets + bags, 1):
        boxes.append({"id": i, "class_id": o["class_id"], **{k: o[k] for k in ("cx", "cy", "w", "h", "angle")},
                      "parent_id": None, "target": False, "occluded": False, "truncated": False})
    pal = [b for b, o in zip(boxes, pallets + bags) if b["class_id"] == 0]
    poly = {b["id"]: np.float32(o["corners"]) for b, o in zip(boxes, pallets + bags)}
    for b in boxes:
        if b["class_id"] == 1:
            inside = [p for p in pal if cv2.pointPolygonTest(poly[p["id"]], (b["cx"], b["cy"]), False) >= 0]
            if inside:
                b["parent_id"] = min(inside, key=lambda p: p["w"] * p["h"])["id"]
    if pal:
        min(pal, key=lambda p: math.hypot(p["cx"] - width / 2, p["cy"] - height / 2))["target"] = True
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="chỉ in thống kê, không ghi file")
    args = ap.parse_args()

    preds = json.loads(args.predictions.read_text(encoding="utf-8"))
    sel_path = HERE / "selection.json"
    ann_path = HERE / "results" / "annotations.json"
    selection = json.loads(sel_path.read_text(encoding="utf-8"))
    doc = json.loads(ann_path.read_text(encoding="utf-8"))
    with open(ROOT / "code/image_sorter/results/labels.csv", encoding="utf-8-sig") as f:
        rows = {r["group_id"]: r for r in csv.DictReader(f)}
    labeled = list(selection["images"])
    known = {im["id"] for im in labeled} | set(doc["records"])
    note = f"Nháp từ {args.predictions.parent.name}, chưa kiểm tra."

    added = []
    for image_id, pred in preds.items():
        if image_id in known:
            continue
        row = rows[image_id]
        h, w = cv2.imread(str(ROOT / row["image_path"])).shape[:2]
        image = {"id": image_id, "image_path": row["image_path"], "depth_path": row["depth_path"],
                 "width": w, "height": h, "state": row["state"],
                 "tags": [k for k in FLAGS if row.get(k) == "1"], "reason": row.get("note", ""),
                 "captured_at": row["captured_at"], "sequence_group": nearest_group(row, labeled),
                 "old_sequence_hint": "", "purpose": "draft_import"}
        record = validate({"objects": draft(pred["objects"], w, h), "reviewed": False, "note": note}, image)
        added.append((image, record))

    by_group = {}
    for image, record in added:
        by_group.setdefault(image["sequence_group"], []).append(len(record["objects"]))
    for g, n in sorted(by_group.items()):
        print(f"{g:26s} +{len(n):3d} ảnh, TB {sum(n) / len(n):.1f} box/ảnh")
    print(f"Tổng: thêm {len(added)} ảnh nháp; giữ nguyên {len(doc['records'])} record đã có.")
    if args.dry_run or not added:
        return
    selection["images"] += [image for image, _ in added]
    doc["images"] = selection["images"]
    doc["records"].update({image["id"]: record for image, record in added})
    sel_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ann_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã ghi {sel_path.name} và {ann_path.name}. Mở labeler: python code/obb_labeler/app.py")


if __name__ == "__main__":
    main()
