"""Ước lượng nhiễu nhãn: so nhãn của cùng một vật qua các frame của một chuỗi chụp camera đứng yên.

    .venv/Scripts/python code/train/label_noise.py

Phần lớn frame trong một chuỗi đã khác cảnh (dỡ bao, xe nâng dời pallet), nên chỉ dùng cặp ảnh gần
như trùng nhau (phase correlation > MIN_SIMILAR). Mỗi box ghép với box cùng lớp có IoU cao nhất ở ảnh
kia (IoU > 0.5 coi là cùng vật).

In hai loại IoU: IoU đa giác (hình học thường) và probiou (Ultralytics dùng để chấm mAP OBB, cao hơn
IoU đa giác với cùng một cặp box). Trần mAP phải đọc theo probiou.
- "nhãn-nhãn": coi lần khoanh này là đáp án của lần kia (hai lần nhiễu).
- "model hoàn hảo": model đoán đúng vật thật, chỉ còn nhiễu của một lần khoanh; ước lượng bằng
  1 - (1 - IoU) / sqrt(2), vì sai lệch giữa hai nhãn độc lập lớn hơn sai lệch nhãn-thật khoảng sqrt(2) lần.
"""
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.utils.metrics import batch_probiou

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import to_xywhr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = ROOT / "code/obb_labeler/results/annotations.json"
NAMES = {0: "pallet_region", 1: "bag"}
THRESHOLDS = np.arange(0.5, 0.96, 0.05)
MIN_SIMILAR = 0.3


def iou(a, b):
    a, b = np.float32(a), np.float32(b)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0


def probiou(a, b):
    return float(batch_probiou(torch.tensor(to_xywhr([a])), torch.tensor(to_xywhr([b])))[0, 0])


def ceiling(vals):
    """Trung bình tỷ lệ cặp đạt từng ngưỡng 0.5..0.95 ~ trần AP50-95."""
    return np.mean([(vals >= t).mean() for t in THRESHOLDS])


def main():
    doc = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for image in doc["images"]:
        record = doc["records"].get(image["id"])
        if record and record.get("reviewed"):
            gray = cv2.cvtColor(cv2.imread(str(ROOT / image["image_path"])), cv2.COLOR_BGR2GRAY)
            groups[image.get("sequence_group", "?")].append((np.float32(gray), record["objects"]))

    ious = {"poly": defaultdict(list), "probiou": defaultdict(list)}
    for group, items in sorted(groups.items()):
        n_pairs = defaultdict(int)
        for (img_a, objs_a), (img_b, objs_b) in itertools.combinations(items, 2):
            if cv2.phaseCorrelate(img_a, img_b)[1] < MIN_SIMILAR:
                continue
            for obj in objs_a:
                cands = [o["corners"] for o in objs_b if o["class_id"] == obj["class_id"]]
                if not cands:
                    continue
                best = max(cands, key=lambda c: iou(obj["corners"], c))
                if iou(obj["corners"], best) > 0.5:
                    ious["poly"][obj["class_id"]].append(iou(obj["corners"], best))
                    ious["probiou"][obj["class_id"]].append(probiou(obj["corners"], best))
                    n_pairs[obj["class_id"]] += 1
        for cls, n in n_pairs.items():
            print(f"{group:28s} {NAMES[cls]:14s} cặp={n:4d}")

    print("\nTổng hợp (cặp ảnh gần trùng, box IoU > 0.5):")
    for kind, by_cls in ious.items():
        for cls, vals in sorted(by_cls.items()):
            vals = np.array(vals)
            perfect = 1 - (1 - vals) / np.sqrt(2)
            pct = np.percentile(vals, [10, 50, 90])
            print(f"  {kind:8s} {NAMES[cls]:14s} n={len(vals):3d} IoU p10/p50/p90 = "
                  f"{pct[0]:.3f}/{pct[1]:.3f}/{pct[2]:.3f} | ≥0.85: {(vals >= 0.85).mean():.0%} "
                  f"≥0.95: {(vals >= 0.95).mean():.0%} | trần AP50-95: nhãn-nhãn {ceiling(vals):.3f}, "
                  f"model hoàn hảo ≈ {ceiling(perfect):.3f}")


if __name__ == "__main__":
    main()
