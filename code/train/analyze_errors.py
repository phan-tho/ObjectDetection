"""Phân tích lỗi model OBB trên ảnh val: bỏ sót, báo nhầm, box lệch, theo từng đặc điểm nhãn.

    .venv/Scripts/python code/train/analyze_errors.py --imgsz 960 \
        --fold runs/red_v3_960/weights/last.pt august_red september_pallet_4 \
        --fold runs/sep3_v3_960/weights/last.pt september_pallet_3 september_pallet_2

Mỗi `--fold` là weights + các sequence_group làm val của fold đó. Ảnh gốc 1920x1080, tô xám vùng
chưa nhãn như lúc train. Nên dùng last.pt: best.pt được chọn bằng chính val nên lạc quan.
Kết quả trong --out (mặc định runs/errors/): objects.csv (mỗi box nhãn/dự đoán một dòng),
summary.txt, vis/<ảnh>.jpg (xanh = đúng, đỏ = bỏ sót, vàng = box lệch IoU < 0.75, cam = báo nhầm).
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset import ROOT, load_items, mask_unlabeled  # noqa: E402

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


def rate(rows, cond):
    sub = [r for r in rows if cond(r)]
    bad = sum(r["result"] != "ok" for r in sub)
    return f"{bad}/{len(sub)} ({100 * bad / len(sub):.0f}%)" if sub else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", nargs="+", action="append", required=True, metavar="WEIGHTS GROUP",
                    help="weights rồi các sequence_group val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "errors")
    args = ap.parse_args()

    (args.out / "vis").mkdir(parents=True, exist_ok=True)
    items = load_items()
    rows = []
    for weights, *groups in args.fold:
        model = YOLO(weights)
        for it in (it for it in items if it["sequence_group"] in groups):
            img = cv2.imread(str(ROOT / it["image_path"]))
            img, objects = mask_unlabeled(img, it)
            h, w = img.shape[:2]
            gts = [{"cls": o["class_id"], "poly": np.array(o["corners"]), "meta": o} for o in objects]
            target = next((o["id"] for o in objects if o.get("target")), None)
            res = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            preds = [{"cls": int(c), "poly": p, "conf": float(s)} for p, c, s in
                     zip(res.obb.xyxyxyxy.cpu().numpy(), res.obb.cls.tolist(), res.obb.conf.tolist())]
            matched = match(gts, preds)
            gt_hit = {i: (j, iou) for i, j, iou in matched}
            pred_hit = {j for _, j, _ in matched}
            stem = it["id"].replace("/", "__")
            vis = img.copy()
            for i, g in enumerate(gts):
                m = g["meta"]
                area = cv2.contourArea(np.float32(g["poly"])) / (w * h)
                edge = g["poly"].min() < 3 or g["poly"][:, 0].max() > w - 3 or g["poly"][:, 1].max() > h - 3
                j, iou = gt_hit.get(i, (None, 0.0))
                kind = "miss" if j is None else ("loose" if iou < TIGHT_IOU else "ok")
                cx, cy = g["poly"].mean(0)
                rows.append({"image": stem, "group": it["sequence_group"], "type": "gt", "cls": NAMES[g["cls"]],
                             "label": ("P" if g["cls"] == 0 else "B") + str(m.get("id", "")),
                             "cx": round(float(cx)), "cy": round(float(cy)), "result": kind, "iou": round(iou, 3),
                             "conf": round(preds[j]["conf"], 3) if j is not None else "",
                             "area_pct": round(100 * area, 2), "at_edge": int(edge),
                             "occluded": int(m.get("occluded", False)), "truncated": int(m.get("truncated", False)),
                             "on_target": int(m.get("parent_id") == target) if g["cls"] == 1 else "",
                             "loose_bag": int(m.get("parent_id") is None) if g["cls"] == 1 else "",
                             "inside_pallet": ""})
                color = {"miss": (0, 0, 255), "loose": (0, 255, 255), "ok": (0, 200, 0)}[kind]
                cv2.polylines(vis, [g["poly"].astype(np.int32)], True, color, 3)
                if j is not None and kind == "loose":
                    cv2.polylines(vis, [preds[j]["poly"].astype(np.int32)], True, (255, 255, 0), 1)
            for j, p in enumerate(preds):
                if j in pred_hit:
                    continue
                best = max((poly_iou(p["poly"], g["poly"]) for g in gts if g["cls"] == p["cls"]), default=0.0)
                inside = any(cv2.pointPolygonTest(np.float32(g["poly"]), tuple(map(float, p["poly"].mean(0))), False) >= 0
                             for g in gts if g["cls"] == 0)
                cx, cy = p["poly"].mean(0)
                rows.append({"image": stem, "group": it["sequence_group"], "type": "pred", "cls": NAMES[p["cls"]],
                             "label": "", "cx": round(float(cx)), "cy": round(float(cy)), "result": "duplicate" if best >= 0.3 else "false_pos", "iou": round(best, 3),
                             "conf": round(p["conf"], 3), "area_pct": "", "at_edge": "", "occluded": "",
                             "truncated": "", "on_target": "", "loose_bag": "", "inside_pallet": int(inside)})
                cv2.polylines(vis, [p["poly"].astype(np.int32)], True, (0, 140, 255), 2)
                cv2.putText(vis, f"{NAMES[p['cls']][0]}{p['conf']:.2f}", tuple(p["poly"][0].astype(int)), 0, 0.8,
                            (0, 140, 255), 2)
            cv2.imwrite(str(args.out / "vis" / f"{stem}.jpg"), vis)

    with open(args.out / "objects.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gt = [r for r in rows if r["type"] == "gt"]
    bags = [r for r in gt if r["cls"] == "bag"]
    lines = []
    for cls in NAMES.values():
        sub = [r for r in gt if r["cls"] == cls]
        c = Counter(r["result"] for r in sub)
        ious = [r["iou"] for r in sub if r["result"] != "miss"]
        lines.append(f"{cls}: {len(sub)} nhãn | đúng {c['ok']} | lệch {c['loose']} | bỏ sót {c['miss']} | "
                     f"IoU trung bình khi bắt được {np.mean(ious):.3f}")
    lines.append("\nTỷ lệ lỗi (bỏ sót + lệch) của bao theo đặc điểm:")
    for label, cond in [("trên pallet mục tiêu", lambda r: r["on_target"] == 1),
                        ("pallet khác", lambda r: r["on_target"] == 0 and r["loose_bag"] == 0),
                        ("bao rời (không thuộc pallet)", lambda r: r["loose_bag"] == 1),
                        ("bị che", lambda r: r["occluded"] == 1), ("không bị che", lambda r: r["occluded"] == 0),
                        ("cắt biên", lambda r: r["truncated"] == 1), ("chạm mép ảnh", lambda r: r["at_edge"] == 1),
                        ("diện tích <1% ảnh", lambda r: r["area_pct"] < 1),
                        ("diện tích 1-2%", lambda r: 1 <= r["area_pct"] < 2),
                        ("diện tích >=2%", lambda r: r["area_pct"] >= 2)]:
        lines.append(f"  {label}: {rate(bags, cond)}")
    fp = [r for r in rows if r["type"] == "pred"]
    for cls in NAMES.values():
        f_ = [r for r in fp if r["cls"] == cls]
        lines.append(f"\nBáo nhầm {cls} (conf ≥ {args.conf}): {sum(r['result'] == 'false_pos' for r in f_)} "
                     f"(trong vùng pallet {sum(r['result'] == 'false_pos' and r['inside_pallet'] == 1 for r in f_)}), "
                     f"box trùng {sum(r['result'] == 'duplicate' for r in f_)}")
    per_img = defaultdict(Counter)
    for r in rows:
        per_img[(r["group"], r["image"])][r["result"]] += 1
    lines.append("\nTheo ảnh, xếp theo số lỗi (ok/loose/miss/false_pos/duplicate):")
    for (g, k), c in sorted(per_img.items(), key=lambda x: -(x[1]["loose"] + x[1]["miss"] + x[1]["false_pos"])):
        lines.append(f"  {g:24s} {k}: {c['ok']}/{c['loose']}/{c['miss']}/{c['false_pos']}/{c['duplicate']}")
    text = "\n".join(lines)
    (args.out / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
