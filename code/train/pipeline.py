"""So sánh phương pháp suy luận trên ảnh val, chấm cùng một cách (mAP như Ultralytics val).

    .venv/Scripts/python code/train/pipeline.py --fold red --val-groups august_red september_pallet_4 \
        --model runs/red_v3/weights/best.pt --crop-model runs/red_crop/weights/best.pt

Phương pháp (chạy trên ảnh gốc 1920x1080, đã tô xám vùng chưa nhãn như lúc train):
- plain: model thường trên cả ảnh. Đối chứng, phải gần với số Ultralytics val báo.
- tiled: thêm bao dự đoán trên các ô chồng nhau ở độ phân giải gốc (bao to ra ~2 lần),
  gộp với bao trên cả ảnh bằng NMS xoay. Pallet vẫn lấy từ cả ảnh.
- two_stage: pallet từ model thường; mỗi pallet được crop ở độ phân giải gốc (nới 15%) rồi
  model crop (build_crops.py) tìm bao trong đó. Bao ngoài mọi crop lấy từ cả ảnh.
- <method>+inpallet: chỉ giữ bao có tâm nằm trong pallet đã phát hiện (bỏ báo nhầm ngoài
  pallet; robot chỉ cần bao trên pallet). Bao rời ngoài pallet có nhãn sẽ thành bỏ sót.

Kết quả nối vào runs/pipeline.jsonl (mỗi dòng một fold x phương pháp).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import load_items, mask_unlabeled  # noqa: E402

IOUV = np.linspace(0.5, 0.95, 10)
CROP_PAD = 0.15
CONF = 0.001  # như Ultralytics val: lấy hết để tính đường PR
NMS_IOU = 0.7  # mặc định Ultralytics; đổi bằng --nms-iou


def to_xywhr(polys):
    polys = np.asarray(polys, np.float32).reshape(-1, 8)
    return xyxyxyxy2xywhr(polys) if len(polys) else np.zeros((0, 5), np.float32)


def predict(model, img, imgsz, cls_map=None):
    """-> dict poly (N,4,2), conf (N,), cls (N,) theo toạ độ của img."""
    r = model.predict(img, imgsz=imgsz, conf=CONF, iou=NMS_IOU, max_det=300, verbose=False)[0]
    cls = r.obb.cls.cpu().numpy().astype(int)
    if cls_map is not None:
        cls = np.array([cls_map.get(c, -1) for c in cls], int)
    return {"poly": r.obb.xyxyxyxy.cpu().numpy(), "conf": r.obb.conf.cpu().numpy(), "cls": cls}


def concat(*ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ("poly", "conf", "cls")}


def take(d, keep):
    return {k: v[keep] for k, v in d.items()}


def shift(d, dx, dy):
    return {**d, "poly": d["poly"] + np.array([dx, dy], np.float32)}


def nms(d, iou_thr=0.5):
    """NMS xoay theo lớp, dùng probiou như Ultralytics."""
    if not len(d["conf"]):
        return d
    keep = []
    for c in np.unique(d["cls"]):
        idx = np.where(d["cls"] == c)[0]
        idx = idx[np.argsort(-d["conf"][idx])]
        iou = batch_probiou(torch.tensor(to_xywhr(d["poly"][idx])), torch.tensor(to_xywhr(d["poly"][idx]))).numpy()
        alive = np.ones(len(idx), bool)
        for i in range(len(idx)):
            if alive[i]:
                keep.append(idx[i])
                alive[i + 1:] &= iou[i, i + 1:] < iou_thr
    return take(d, np.array(sorted(keep)))


def touches_cut(poly, x0, y0, x1, y1, w, h, margin=4):
    """Box chạm cạnh ô/crop mà cạnh đó không phải mép ảnh: bao bị ô cắt, bỏ."""
    xs, ys = poly[:, 0], poly[:, 1]
    return ((x0 > 0 and xs.min() < x0 + margin) or (y0 > 0 and ys.min() < y0 + margin)
            or (x1 < w and xs.max() > x1 - margin) or (y1 < h and ys.max() > y1 - margin))


def from_windows(model, img, windows, imgsz, cls_map=None, only_cls=1):
    """Chạy model trên từng cửa sổ (x0, y0, x1, y1), trả về box lớp only_cls theo toạ độ ảnh."""
    h, w = img.shape[:2]
    out = []
    for x0, y0, x1, y1 in windows:
        d = shift(predict(model, img[y0:y1, x0:x1], imgsz, cls_map), x0, y0)
        keep = [(c == only_cls and not touches_cut(p, x0, y0, x1, y1, w, h))
                for p, c in zip(d["poly"], d["cls"])]
        out.append(take(d, np.array(keep, bool)))
    return concat(*out) if out else {"poly": np.zeros((0, 4, 2)), "conf": np.zeros(0), "cls": np.zeros(0, int)}


def tile_windows(w, h, tile, overlap=0.25):
    step = int(tile * (1 - overlap))
    xs = list(range(0, max(w - tile, 0) + 1, step)) + [max(w - tile, 0)]
    ys = list(range(0, max(h - tile, 0) + 1, step)) + [max(h - tile, 0)]
    return sorted({(x, y, min(x + tile, w), min(y + tile, h)) for x in xs for y in ys})


def pallet_windows(pallets, w, h):
    wins = []
    for p in pallets:
        (x0, y0), (x1, y1) = p.min(0), p.max(0)
        px, py = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
        wins.append((int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(w, x1 + px)), int(min(h, y1 + py))))
    return wins


def inside_any(poly, regions):
    c = tuple(map(float, poly.mean(0)))
    return any(cv2.pointPolygonTest(np.float32(r), c, False) >= 0 for r in regions)


def run_methods(img, model, crop_model, imgsz, tile, pallet_conf=0.25, tiled=True):
    h, w = img.shape[:2]
    full = predict(model, img, imgsz)
    pallets = take(full, full["cls"] == 0)
    bags = take(full, full["cls"] == 1)
    out = {"plain": full}
    if tiled:
        tiles = from_windows(model, img, tile_windows(w, h, tile), imgsz)
        out["tiled"] = concat(pallets, nms(concat(bags, tiles)))
    if crop_model is not None:
        sure = pallets["poly"][pallets["conf"] >= pallet_conf]
        wins = pallet_windows(sure, w, h)
        crop_bags = from_windows(crop_model, img, wins, imgsz, cls_map={0: 1})
        outside = np.array([not any(x0 <= p[:, 0].mean() <= x1 and y0 <= p[:, 1].mean() <= y1
                                    for x0, y0, x1, y1 in wins) for p in bags["poly"]], bool)
        out["two_stage"] = concat(pallets, nms(concat(crop_bags, take(bags, outside))))
    for name in list(out):
        d = out[name]
        regions = d["poly"][(d["cls"] == 0) & (d["conf"] >= pallet_conf)]
        keep = np.array([c == 0 or inside_any(p, regions) for p, c in zip(d["poly"], d["cls"])], bool)
        out[name + "+inpallet"] = take(d, keep)
    return out


def score(records):
    """records: list (pred dict, gt_poly, gt_cls). Ghép như BaseValidator.match_predictions."""
    tps, confs, pcls, tcls = [], [], [], []
    for pred, gpoly, gcls in records:
        n = len(pred["conf"])
        correct = np.zeros((n, len(IOUV)), bool)
        if n and len(gcls):
            iou = batch_probiou(torch.tensor(to_xywhr(gpoly)), torch.tensor(to_xywhr(pred["poly"]))).numpy()
            iou = iou * (gcls[:, None] == pred["cls"][None])
            for i, t in enumerate(IOUV):
                m = np.array(np.nonzero(iou >= t)).T
                if len(m):
                    if len(m) > 1:
                        m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                        m = m[np.unique(m[:, 1], return_index=True)[1]]
                        m = m[np.unique(m[:, 0], return_index=True)[1]]
                    correct[m[:, 1], i] = True
        tps.append(correct)
        confs.append(pred["conf"])
        pcls.append(pred["cls"])
        tcls.append(gcls)
    res = ap_per_class(np.concatenate(tps), np.concatenate(confs), np.concatenate(pcls), np.concatenate(tcls))
    ap, classes = res[5], res[6]
    per = {["pallet_region", "bag"][int(c)]: float(ap[i].mean()) for i, c in enumerate(classes)}
    return {"mAP50": float(ap[:, 0].mean()), "mAP50-95": float(ap.mean()), "per_class_mAP50-95": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--val-groups", nargs="+", required=True, help="sequence_group làm val (như build_dataset)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--crop-model")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tile", type=int, default=960, help="cạnh ô tiling, pixel ảnh gốc")
    ap.add_argument("--tag", default="", help="ghi kèm vào kết quả, vd tên cấu hình train")
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--methods", nargs="*", help="chỉ ghi các phương pháp này (mặc định tất cả)")
    args = ap.parse_args()
    global NMS_IOU
    NMS_IOU = args.nms_iou

    items = [it for it in load_items() if it["sequence_group"] in args.val_groups]
    model, crop_model = YOLO(args.model), YOLO(args.crop_model) if args.crop_model else None
    records = {}
    for it in items:
        img = cv2.imread(str(ROOT / it["image_path"]))
        img, objects = mask_unlabeled(img, it)
        gpoly = np.array([o["corners"] for o in objects], np.float32).reshape(-1, 4, 2)
        gcls = np.array([o["class_id"] for o in objects], int)
        want_tiled = not args.methods or any(m.startswith("tiled") for m in args.methods)
        for name, pred in run_methods(img, model, crop_model, args.imgsz, args.tile, tiled=want_tiled).items():
            records.setdefault(name, []).append((pred, gpoly, gcls))
    with open(ROOT / "runs" / "pipeline.jsonl", "a", encoding="utf-8") as f:
        for name, recs in records.items():
            if args.methods and name not in args.methods:
                continue
            r = {"fold": args.fold, "tag": args.tag, "method": name, "n_val": len(items), **score(recs)}
            print(json.dumps(r))
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
