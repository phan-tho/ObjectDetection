"""Phân tích lỗi model OBB trên tập val: bỏ sót, báo nhầm, box lệch, theo từng đặc điểm nhãn.

    .venv/Scripts/python code/train/analyze_errors.py \
        --pair runs/kaggle/runs/red_y26n_aug/weights/best.pt fold_red_aug \
        --pair runs/kaggle/runs/sep3_y26n_aug/weights/best.pt fold_sep3_aug

Mỗi `--pair` là weights + tên dataset trong datasets/ (lấy ảnh val của dataset đó).
Kết quả trong runs/errors/: objects.csv (mỗi box nhãn/dự đoán một dòng), summary.txt,
vis/<ảnh>.jpg (xanh = đúng, đỏ = bỏ sót, cam = báo nhầm, vàng = box lệch IoU < 0.75).
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = ROOT / "code/obb_labeler/results/annotations.json"
NAMES = {0: "pallet", 1: "bag"}
MATCH_IOU = 0.5   # ngưỡng tính là bắt được
TIGHT_IOU = 0.75  # dưới mức này coi là box lệch (kéo mAP50-95 xuống)


def poly_iou(a, b):
    a, b = np.float32(a), np.float32(b)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0


def match(gts, preds):
    """Ghép tham lam theo IoU giảm dần, cùng lớp. Trả về list (gt_idx, pred_idx, iou)."""
    pairs = sorted(((poly_iou(g["poly"], p["poly"]), i, j)
                    for i, g in enumerate(gts) for j, p in enumerate(preds) if g["cls"] == p["cls"]),
                   reverse=True)
    used_g, used_p, out = set(), set(), []
    for iou, i, j in pairs:
        if iou < MATCH_IOU or i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        out.append((i, j, iou))
    return out


def gt_meta(stem, annotations):
    """Cờ nhãn gốc (occluded/truncated) theo thứ tự box trong file label."""
    image_id = stem.replace("__", "/", 1)
    record = annotations["records"].get(image_id)
    return record["objects"] if record else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("WEIGHTS", "DATASET"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "errors")
    args = ap.parse_args()

    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    (args.out / "vis").mkdir(parents=True, exist_ok=True)
    rows = []
    for weights, dataset in args.pair:
        model = YOLO(weights)
        val = ROOT / "datasets" / dataset / "val"
        for img_path in sorted((val / "images").glob("*.jpg")):
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            gts = []
            for line in (val / "labels" / (img_path.stem + ".txt")).read_text().split("\n"):
                if line.strip():
                    v = line.split()
                    gts.append({"cls": int(v[0]), "poly": np.array(v[1:], float).reshape(4, 2) * [w, h]})
            # Ảnh val có pallet phụ bị tô xám nên số box nhãn có thể ít hơn JSON; chỉ lấy cờ khi khớp số lượng.
            meta = gt_meta(img_path.stem, annotations)
            meta = meta if meta and len(meta) == len(gts) else None
            res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            preds = [{"cls": int(c), "poly": p, "conf": float(s)} for p, c, s in
                     zip(res.obb.xyxyxyxy.cpu().numpy(), res.obb.cls.tolist(), res.obb.conf.tolist())]
            matched = match(gts, preds)
            gt_hit = {i: (j, iou) for i, j, iou in matched}
            pred_hit = {j for _, j, _ in matched}
            vis = img.copy()
            for i, g in enumerate(gts):
                area = cv2.contourArea(np.float32(g["poly"])) / (w * h)
                cx, cy = g["poly"].mean(0) / [w, h]
                edge = min(cx, cy, 1 - cx, 1 - cy) < 0.08 or (g["poly"].min() < 3) \
                    or (g["poly"][:, 0].max() > w - 3) or (g["poly"][:, 1].max() > h - 3)
                j, iou = gt_hit.get(i, (None, 0.0))
                kind = "miss" if j is None else ("loose" if iou < TIGHT_IOU else "ok")
                m = meta[i] if meta else {}
                rows.append({"dataset": dataset, "image": img_path.stem, "type": "gt", "cls": NAMES[g["cls"]],
                             "result": kind, "iou": round(iou, 3), "conf": round(preds[j]["conf"], 3) if j is not None else "",
                             "area_pct": round(100 * area, 2), "at_edge": int(edge),
                             "occluded": int(m.get("occluded", False)), "truncated": int(m.get("truncated", False))})
                color = {"miss": (0, 0, 255), "loose": (0, 255, 255), "ok": (0, 200, 0)}[kind]
                cv2.polylines(vis, [g["poly"].astype(np.int32)], True, color, 3)
            for j, p in enumerate(preds):
                if j in pred_hit:
                    continue
                best = max((poly_iou(p["poly"], g["poly"]) for g in gts if g["cls"] == p["cls"]), default=0.0)
                inside = any(cv2.pointPolygonTest(np.float32(g["poly"]), tuple(map(float, p["poly"].mean(0))), False) >= 0
                             for g in gts if g["cls"] == 0)
                rows.append({"dataset": dataset, "image": img_path.stem, "type": "pred", "cls": NAMES[p["cls"]],
                             "result": "duplicate" if best >= 0.3 else "false_pos", "iou": round(best, 3),
                             "conf": round(p["conf"], 3), "area_pct": "", "at_edge": "",
                             "occluded": "", "truncated": "", "inside_pallet": int(inside)})
                cv2.polylines(vis, [p["poly"].astype(np.int32)], True, (0, 140, 255), 2)
                cv2.putText(vis, f"{p['conf']:.2f}", tuple(p["poly"][0].astype(int)), 0, 0.6, (0, 140, 255), 2)
            cv2.imwrite(str(args.out / "vis" / f"{dataset}__{img_path.stem}.jpg"), vis)

    fields = ["dataset", "image", "type", "cls", "result", "iou", "conf", "area_pct", "at_edge",
              "occluded", "truncated", "inside_pallet"]
    with open(args.out / "objects.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = []
    gt = [r for r in rows if r["type"] == "gt"]
    for cls in NAMES.values():
        sub = [r for r in gt if r["cls"] == cls]
        c = Counter(r["result"] for r in sub)
        ious = [r["iou"] for r in sub if r["result"] != "miss"]
        lines.append(f"{cls}: {len(sub)} nhãn | đúng {c['ok']} | lệch {c['loose']} | bỏ sót {c['miss']} | "
                     f"IoU trung bình khi bắt được {np.mean(ious):.3f}")
    lines.append("\nTỷ lệ lỗi (bỏ sót + lệch) của bao theo đặc điểm:")
    bags = [r for r in gt if r["cls"] == "bag"]
    for key in ("at_edge", "occluded", "truncated"):
        for val in (1, 0):
            sub = [r for r in bags if r[key] == val]
            if sub:
                bad = sum(r["result"] != "ok" for r in sub)
                lines.append(f"  {key}={val}: {bad}/{len(sub)} ({100 * bad / len(sub):.0f}%)")
    for lo, hi in [(0, 1), (1, 2), (2, 100)]:
        sub = [r for r in bags if lo <= r["area_pct"] < hi]
        if sub:
            bad = sum(r["result"] != "ok" for r in sub)
            lines.append(f"  diện tích {lo}-{hi}% ảnh: {bad}/{len(sub)} ({100 * bad / len(sub):.0f}%)")
    fp = [r for r in rows if r["type"] == "pred"]
    lines.append(f"\nBáo nhầm: {sum(r['result'] == 'false_pos' for r in fp)} "
                 f"(trong vùng pallet {sum(r['result'] == 'false_pos' and r['inside_pallet'] for r in fp)}), "
                 f"box trùng {sum(r['result'] == 'duplicate' for r in fp)}")
    per_img = defaultdict(Counter)
    for r in rows:
        per_img[r["image"]][r["result"]] += 1
    lines.append("\nTheo ảnh (ok/loose/miss/false_pos/duplicate):")
    for k, c in per_img.items():
        lines.append(f"  {k}: {c['ok']}/{c['loose']}/{c['miss']}/{c['false_pos']}/{c['duplicate']}")
    text = "\n".join(lines)
    (args.out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
