"""Dataset cho giai đoạn 2 (two-stage): crop vùng pallet ở độ phân giải gốc, chỉ nhãn bao.

    .venv/Scripts/python code/train/build_crops.py --name red_crop \
        --val-groups august_red september_pallet_4 --aug 15 --heavy

Ở ảnh 640 cả khung, bao nhỏ chỉ còn vài chục pixel. Crop pallet ở 1920x1080 làm bao to hơn
2–3 lần. Train: crop quanh mỗi pallet với lề ngẫu nhiên 5–30% mỗi cạnh (bù sai số khi giai
đoạn 1 khoanh pallet), cộng một cửa sổ ngẫu nhiên mỗi ảnh (bao rời, mẫu âm). Augment áp lên
ảnh đầy đủ trước khi crop. Val: crop lề cố định 15% quanh pallet nhãn, đúng như pipeline.py.
Một lớp duy nhất: 0 = bag. Bao bị crop cắt dưới 30% thì tô xám (clip_objects).
"""
import argparse
import shutil

import cv2
import numpy as np

from augment import augment, bag_donors, clip_objects, clutter_donors
from build_dataset import ROOT, load_items, mask_unlabeled, write_sample

CROP_PAD = 0.15     # phải trùng pipeline.py
MAX_CROP_SIDE = 1024


def crop(img, objects, x0, y0, x1, y1):
    x0, y0 = int(max(0, x0)), int(max(0, y0))
    x1, y1 = int(min(img.shape[1], x1)), int(min(img.shape[0], y1))
    if x1 - x0 < 64 or y1 - y0 < 64:
        return None, None
    sub = img[y0:y1, x0:x1].copy()
    bags = [{**o, "class_id": 0, "corners": (np.array(o["corners"]) - [x0, y0]).tolist()}
            for o in objects if o["class_id"] == 1]
    sub, bags = clip_objects(sub, bags)
    s = min(1.0, MAX_CROP_SIDE / max(sub.shape[:2]))
    if s < 1:
        sub = cv2.resize(sub, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        bags = [{**o, "corners": (np.array(o["corners"]) * s).tolist()} for o in bags]
    return sub, bags


def pallet_box(o, pad):
    """pad: số hoặc mảng 4 lề (trái, trên, phải, dưới) theo tỉ lệ kích thước pallet."""
    pts = np.array(o["corners"])
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    pl, pt, pr, pb = np.broadcast_to(pad, 4)
    w, h = x1 - x0, y1 - y0
    return x0 - pl * w, y0 - pt * h, x1 + pr * w, y1 + pb * h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--val-groups", nargs="*", default=[])
    ap.add_argument("--aug", type=int, default=15)
    ap.add_argument("--heavy", action="store_true", help="augment v3 (mặc định v1)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = ROOT / "datasets" / args.name
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    samples = []
    for it in load_items():
        img = cv2.imread(str(ROOT / it["image_path"]))
        img, objects = mask_unlabeled(img, it)
        samples.append((it, img, objects, it["sequence_group"] in args.val_groups))
    train = [(img, objs) for it, img, objs, v in samples if not v]
    donors = [d for img, objs in train for d in bag_donors(img, objs)]
    clutter = [c for img, objs in train for c in clutter_donors(img, objs, rng)] if args.heavy else None

    counts = {"train": 0, "val": 0}
    for it, img, objects, is_val in samples:
        stem = it["id"].replace("/", "__")
        if is_val:
            for i, p in enumerate(o for o in objects if o["class_id"] == 0):
                sub, bags = crop(img, objects, *pallet_box(p, CROP_PAD))
                if sub is not None:
                    write_sample(sub, bags, f"{stem}__p{i}", out / "val")
                    counts["val"] += 1
            continue
        for k in range(args.aug + 1):
            if k == 0:
                im, objs = img, objects
            else:
                im, objs = augment(img, objects, rng, donors=donors, scene=True, heavy=args.heavy,
                                   clutter=clutter)
            pallets = [o for o in objs if o["class_id"] == 0]
            wins = [pallet_box(p, rng.uniform(0.05, 0.3, 4)) for p in pallets]
            h, w = im.shape[:2]
            cw, ch = rng.uniform(0.3, 0.6) * w, rng.uniform(0.4, 0.8) * h
            x, y = rng.uniform(0, w - cw), rng.uniform(0, h - ch)
            wins.append((x, y, x + cw, y + ch))
            for i, win in enumerate(wins):
                sub, bags = crop(im, objs, *win)
                if sub is not None:
                    write_sample(sub, bags, f"{stem}__a{k:02d}w{i}", out / "train")
                    counts["train"] += 1

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: train/images\nval: {'val' if counts['val'] else 'train'}/images\n"
        "names:\n  0: bag\n", encoding="utf-8")
    print(f"{args.name}: train {counts['train']} crop, val {counts['val']} crop -> {out}")


if __name__ == "__main__":
    main()
