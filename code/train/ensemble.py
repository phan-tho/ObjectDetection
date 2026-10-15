"""TTA (xoay 90°/lật) và ensemble nhiều weights, gộp bằng WBF cho box xoay; chấm như pipeline.py.

    .venv/Scripts/python code/train/ensemble.py --fold red --val-groups august_red september_pallet_4 \
        --models runs/red_a/weights/last.pt runs/red_b/weights/last.pt --imgsz 960 --tag v3_960

Camera nhìn từ trên và model đã train với xoay 0–180° + lật, nên 8 phép đối xứng của hình vuông đều
nằm trong phân phối train. Mỗi phép: dự đoán trên ảnh biến đổi rồi đổi box về toạ độ gốc.

Gộp (WBF): box cùng lớp xếp theo conf, box nào IoU (probiou) ≥ --fuse-iou với cụm thì nhập cụm.
Toạ độ cụm = trung bình có trọng số conf của (cx, cy, cạnh dài, cạnh ngắn, góc); góc trung bình
theo vector (cos 2θ, sin 2θ) vì box đối xứng 180°. Conf cụm = tổng conf / số lượt dự đoán, nên box
chỉ một lượt thấy bị hạ điểm. Kết quả nối vào runs/pipeline.jsonl với method = tên cấu hình.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import load_items, mask_unlabeled  # noqa: E402
from pipeline import ROOT, concat, nms, predict, score, take, to_xywhr  # noqa: E402

# (tên, biến đổi ảnh, đổi điểm (N,2) trên ảnh biến đổi về ảnh gốc kích thước w×h)
TRANSFORMS = [
    ("id", lambda im: im, lambda p, w, h: p),
    ("hflip", lambda im: im[:, ::-1], lambda p, w, h: np.c_[w - p[:, 0], p[:, 1]]),
    ("vflip", lambda im: im[::-1], lambda p, w, h: np.c_[p[:, 0], h - p[:, 1]]),
    ("rot180", lambda im: im[::-1, ::-1], lambda p, w, h: np.c_[w - p[:, 0], h - p[:, 1]]),
    ("rot90cw", lambda im: cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), lambda p, w, h: np.c_[p[:, 1], h - p[:, 0]]),
    ("rot90ccw", lambda im: cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE),
     lambda p, w, h: np.c_[w - p[:, 1], p[:, 0]]),
    ("transpose", lambda im: cv2.transpose(im), lambda p, w, h: np.c_[p[:, 1], p[:, 0]]),
    ("antitranspose", lambda im: cv2.rotate(cv2.transpose(im), cv2.ROTATE_180),
     lambda p, w, h: np.c_[w - p[:, 1], h - p[:, 0]]),
]
TTA_SETS = {"none": ["id"], "tta4": ["id", "hflip", "vflip", "rot180"], "tta8": [t[0] for t in TRANSFORMS]}


def predict_tta(model, img, imgsz, names):
    h, w = img.shape[:2]
    outs = []
    for name, fwd, inv in TRANSFORMS:
        if name in names:
            d = predict(model, np.ascontiguousarray(fwd(img)), imgsz)
            d["poly"] = inv(d["poly"].reshape(-1, 2), w, h).reshape(-1, 4, 2).astype(np.float32)
            outs.append(d)
    return outs


def canonical(xywhr):
    """Cạnh dài là w, góc trong [-pi/2, pi/2)."""
    x = xywhr.copy()
    swap = x[:, 3] > x[:, 2]
    x[swap, 2], x[swap, 3] = xywhr[swap, 3], xywhr[swap, 2]
    x[swap, 4] += np.pi / 2
    x[:, 4] = (x[:, 4] + np.pi / 2) % np.pi - np.pi / 2
    return x


def to_poly(xywhr):
    return np.array([cv2.boxPoints(((x, y), (w, h), np.degrees(r))) for x, y, w, h, r in xywhr],
                    np.float32).reshape(-1, 4, 2)


def wbf(preds, fuse_iou):
    n_runs = len(preds)
    d = concat(*preds)
    if not len(d["conf"]):
        return d
    out = []
    for c in np.unique(d["cls"]):
        sub = take(d, d["cls"] == c)
        order = np.argsort(-sub["conf"])
        boxes, conf = canonical(to_xywhr(sub["poly"][order])), sub["conf"][order]
        iou = batch_probiou(torch.tensor(boxes), torch.tensor(boxes)).numpy()
        cluster = -np.ones(len(boxes), int)
        heads = []
        for i in range(len(boxes)):
            if heads:
                j = np.argmax(iou[i, heads])
                if iou[i, heads[j]] >= fuse_iou:
                    cluster[i] = j
                    continue
            cluster[i] = len(heads)
            heads.append(i)
        for k in range(len(heads)):
            m = cluster == k
            wgt = conf[m] / conf[m].sum()
            b = boxes[m]
            fused = (wgt[:, None] * b[:, :4]).sum(0)
            ang = np.arctan2((wgt * np.sin(2 * b[:, 4])).sum(), (wgt * np.cos(2 * b[:, 4])).sum()) / 2
            out.append((np.r_[fused, ang], conf[m].sum() / n_runs, c))
    xywhr = np.array([o[0] for o in out], np.float32)
    return {"poly": to_poly(xywhr), "conf": np.array([o[1] for o in out], np.float32),
            "cls": np.array([o[2] for o in out], int)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--val-groups", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fuse-iou", type=float, default=0.55)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="pipeline.jsonl", help="file kết quả trong runs/")
    args = ap.parse_args()

    items = [it for it in load_items() if it["sequence_group"] in args.val_groups]
    models = [YOLO(m) for m in args.models]
    for m in models:
        m.to(f"cuda:{args.device}" if args.device.isdigit() else args.device)  # "0" như Ultralytics
    records = {}
    for it in items:
        img, objects = mask_unlabeled(cv2.imread(str(ROOT / it["image_path"])), it)
        gt = (np.array([o["corners"] for o in objects], np.float32).reshape(-1, 4, 2),
              np.array([o["class_id"] for o in objects], int))
        per_model = [predict_tta(m, img, args.imgsz, TTA_SETS["tta8"]) for m in models]
        idx = {t[0]: i for i, t in enumerate(TRANSFORMS)}
        for tta, names in TTA_SETS.items():
            for scope, ms in [("m0", per_model[:1]), (f"ens{len(models)}", per_model)]:
                if scope != "m0" and len(models) == 1:
                    continue
                runs = [p[idx[n]] for p in ms for n in names]
                key = f"{scope}_{tta}"
                if len(runs) == 1:
                    records.setdefault(key, []).append((runs[0], *gt))
                    continue
                records.setdefault(f"{key}_wbf", []).append((wbf(runs, args.fuse_iou), *gt))
                records.setdefault(f"{key}_nms", []).append((nms(concat(*runs), 0.7), *gt))
    with open(ROOT / "runs" / args.out, "a", encoding="utf-8") as f:
        for name, recs in records.items():
            r = {"fold": args.fold, "tag": args.tag, "method": name, "n_val": len(items), **score(recs)}
            print(json.dumps(r))
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
