"""Dựng dataset YOLO-OBB từ annotations.json: chia theo nhóm chuỗi + augment offline.

    .venv/Scripts/python code/train/build_dataset.py --name fold_red --val-groups august_red september_pallet_4 --aug 15

Val chỉ gồm ảnh gốc của các nhóm được giữ lại (không augment, không trùng cảnh với train).
--val-groups rỗng: toàn bộ ảnh vào train (dùng cho model cuối), val = train để Ultralytics chạy được.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from augment import augment, bag_donors, clutter_donors, empty_pallet_donors

ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = ROOT / "code/obb_labeler/results/annotations.json"
NAMES = {0: "pallet_region", 1: "bag"}
MAX_SIDE = 1280  # đủ cho imgsz ≤ 1280, đọc ảnh nhanh hơn PNG 1920×1080
# Pallet phụ có bao thấy rõ nhưng chưa khoanh (README). Tô xám vùng đó và bỏ nhãn của nó,
# để model không học "bao chưa có nhãn = nền".
UNLABELED_SIDE_PALLETS = {"bao-tai/session_031", "bao-tai/session_043",
                          "bao-tai/session_058"}


def mask_unlabeled(img, item):
    parents = {o["parent_id"] for o in item["objects"] if o["class_id"] == 1}
    keep = []
    for obj in item["objects"]:
        if (item["id"] in UNLABELED_SIDE_PALLETS and obj["class_id"] == 0
                and not obj["target"] and obj["id"] not in parents):
            cv2.fillPoly(img, [np.array(obj["corners"], np.int32)], (114, 114, 114))
        else:
            keep.append(obj)
    return img, keep


def load_items():
    doc = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    items = []
    for image in doc["images"]:
        record = doc["records"].get(image["id"])
        if not record or not record.get("reviewed"):
            continue
        items.append({**image, "objects": record["objects"]})
    return items


def write_sample(img, objects, stem, split_dir):
    h, w = img.shape[:2]
    cv2.imwrite(str(split_dir / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = []
    for obj in objects:
        pts = np.array(obj["corners"]) / [w, h]
        pts = np.clip(pts, 0, 1).reshape(-1)
        lines.append(f"{obj['class_id']} " + " ".join(f"{v:.6f}" for v in pts))
    (split_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--val-groups", nargs="*", default=[])
    ap.add_argument("--aug", type=int, default=15, help="số bản augment cho mỗi ảnh train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copies-only", action="store_true",
                    help="nhân bản ảnh gốc thay vì augment (baseline cùng số bước train)")
    ap.add_argument("--scene", action="store_true",
                    help="augment v2: thêm dán bao rời, che khuất, màng co")
    ap.add_argument("--heavy", action="store_true",
                    help="augment v3 (kèm v2): cắt/zoom/xoay/nghiêng có cắt nhãn, mẫu âm, sọc, mất nét")
    ap.add_argument("--v4", action="store_true",
                    help="augment v4 (kèm v3): bao đè bao, dán pallet rỗng, mẫu âm dày hơn")
    ap.add_argument("--v5", action="store_true",
                    help="augment v5 (kèm v4): ánh sáng theo lớp xếp/từng bao, bóng loáng, cháy sáng, bao biến dạng")
    args = ap.parse_args()

    out = ROOT / "datasets" / args.name
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (out / split / sub).mkdir(parents=True)

    rng = np.random.default_rng(args.seed)
    items = load_items()
    unknown = set(args.val_groups) - {it["sequence_group"] for it in items}
    if unknown:
        raise SystemExit(f"Không có nhóm: {unknown}")
    # Đọc + thu nhỏ trước; toạ độ nhãn đổi sang pixel của ảnh đã thu nhỏ.
    samples = []
    for it in items:
        src = cv2.imread(str(ROOT / it["image_path"]), cv2.IMREAD_COLOR)
        if src is None:
            raise SystemExit(f"Không đọc được {it['image_path']} (chạy download_data.py trước)")
        src, objects = mask_unlabeled(src, it)
        scale = min(1.0, MAX_SIDE / max(src.shape[:2]))
        img = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else src
        objects = [{**o, "corners": (np.array(o["corners"]) * scale).tolist()} for o in objects]
        samples.append((it, img, objects, it["sequence_group"] in args.val_groups))
    # Bao để dán chỉ lấy từ ảnh train, tránh rò rỉ val.
    donors = [d for it, img, objects, is_val in samples if not is_val for d in bag_donors(img, objects)]
    args.v4 |= args.v5
    args.heavy |= args.v4
    clutter = [c for it, img, objects, is_val in samples if not is_val
               for c in clutter_donors(img, objects, rng, n=10 if args.v4 else 6,
                                       max_frac=0.35 if args.v4 else 0.25)] if args.heavy else None
    pallets = [p for it, img, objects, is_val in samples if not is_val
               for p in empty_pallet_donors(img, objects)] if args.v4 else None

    counts = {"train": 0, "val": 0}
    for it, img, objects, is_val in samples:
        stem = it["id"].replace("/", "__")
        split = "val" if is_val else "train"
        write_sample(img, objects, stem, out / split)
        counts[split] += 1
        if is_val:
            continue
        for k in range(args.aug):
            if args.copies_only:
                sample, objs = img, objects
            else:
                sample, objs = augment(img, objects, rng, donors=donors, scene=args.scene or args.heavy,
                                       heavy=args.heavy, clutter=clutter, v4=args.v4, pallets=pallets,
                                       v5=args.v5)
            write_sample(sample, objs, f"{stem}__aug{k:02d}", out / split)
            counts[split] += 1

    val_dir = "val" if counts["val"] else "train"
    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: train/images\nval: {val_dir}/images\n"
        "names:\n" + "".join(f"  {k}: {v}\n" for k, v in NAMES.items()), encoding="utf-8")
    print(f"{args.name}: train {counts['train']} ảnh, val {counts['val']} ảnh -> {out}")


if __name__ == "__main__":
    main()
