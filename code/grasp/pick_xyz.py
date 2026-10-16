"""Chọn bao cần gắp và tính toạ độ XYZ từ box OBB + ảnh depth (rule-based, không cần train).

    .venv/Scripts/python code/grasp/pick_xyz.py --predictions runs/predict_<run>/predictions.json
    .venv/Scripts/python code/grasp/pick_xyz.py --labels            # dùng nhãn tay để kiểm tra luật

Các bước cho mỗi ảnh:
1. Depth (mm, đã khớp ảnh màu): dữ liệu 2026-09-11 đã là median 5 frame; dữ liệu bao-tai lấy median
   các frame khác 0. Pixel 0 = không đo được, luôn bỏ.
2. Pallet mục tiêu: box pallet có cờ target (nhãn tay), nếu không thì box có tâm gần tâm ảnh nhất.
   Bao của nó: tâm nằm trong box pallet.
3. Mặt trên mỗi bao: lấy vùng lõi (box co còn SHRINK), đổi pixel sang điểm 3D, khớp mặt phẳng có
   loại điểm lệch (MAD). Toạ độ gắp = giao của tia qua tâm box với mặt phẳng đó. Không dùng giá trị
   của một pixel lẻ: nhiễu từng pixel 5–20 mm, còn median của hàng nghìn pixel chỉ lệch vài mm.
4. Độ cao đo theo hướng "lên": trục Z robot nếu có cam_to_robot, nếu không thì hướng pháp tuyến trung
   vị của các mặt bao (cả chồng bao/camera có thể nghiêng 10–20°, đo theo trục Z camera sẽ lệch tới
   ~200 mm trên một bao). Lớp trên cùng: bao thấp hơn bao cao nhất không quá LAYER_TOL (nửa lớp 130 mm).
5. Bị đè: tại vùng hai box (nới 15%) giao nhau, bao kia ở phía cao hơn và depth đo thật cao hơn mặt bao
   này quá CONTACT_DZ. So tại chỗ tiếp xúc vì bao đè thường nằm nghiêng/chéo: median độ cao của nó có
   thể bằng hoặc thấp hơn bao bên dưới.
6. Loại: thiếu depth, mặt quá nghiêng hoặc quá gồ ghề, phần thấy được quá nhỏ so với kích thước bao.
   Mặt đo hỏng (nghiêng > BAD_TILT hoặc to hơn 1.5 lần bao) không được dùng để kết luận đè bao khác.
7. Thứ tự gắp: cao nhất trước; chênh dưới 30 mm thì bao thấy được nhiều hơn, conf cao hơn trước.

Toạ độ camera (OpenCV): X sang phải, Y xuống dưới, Z hướng ra trước (mm). Có cam_to_robot trong
camera.json thì xuất thêm toạ độ robot. Kết quả: runs/grasp/<tên>/picks.json và vis/<ảnh>.jpg.
"""
import argparse
import json
import math
import re
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

SHRINK = 0.6          # vùng lõi của box: tránh mép bao, bao bên cạnh và lệch depth-màu ở biên
MIN_VALID = 0.5       # tỷ lệ pixel depth hợp lệ tối thiểu trong vùng lõi
LAYER_HEIGHT = 130.0  # mm, độ dày một lớp bao (bag_spec trong meta)
LAYER_TOL = 0.5 * LAYER_HEIGHT
CONTACT_DZ = 25.0     # mm, tại chỗ hai box tiếp xúc, depth thật cao hơn mặt bao này mức này = có bao đè
CONTACT_Q = 75        # phân vị của (độ cao đo được - mặt bao này) ở vùng tiếp xúc
TIE_DZ = 30.0         # mm, chênh dưới mức này coi như cùng độ cao
MAX_TILT = 35.0       # độ, mặt bao nghiêng quá thì không gắp (bao nằm chéo trên cùng nghiêng ~30°)
BAD_TILT = 60.0       # độ, mặt đo nghiêng hơn mức này hoặc to quá 1.5 lần bao thật là đo hỏng
MAX_RESID = 35.0      # mm, độ gồ ghề (std phần dư mặt phẳng) tối đa
MIN_LONG_FRAC = 0.6   # cạnh dài thấy được >= 60% chiều dài bao (700 mm), ngắn hơn coi như bị che
BAG_MM = (700.0, 450.0)
MIN_CONF = 0.5


def load_meta(image_path):
    folder = (ROOT / image_path).parent
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    k = meta.get("intrinsics") or meta["camera_params"]["color"]["intrinsics"]
    dist = (meta.get("camera_params", {}).get("color", {}).get("distortion"))
    spec = meta.get("bag_spec") or {}
    return k, dist, spec


def load_depth(depth_path):
    """Median các frame khác 0 nếu là dữ liệu nhiều frame rời (bao-tai); bản 2026-09-11 đã gộp sẵn."""
    path = ROOT / depth_path
    frames = sorted(path.parent.glob(re.sub(r"frame_\d+_depth", "frame_*_depth", path.name)))
    if len(frames) <= 1:
        return cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32)
    stack = np.stack([cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32) for f in frames])
    stack[stack == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # pixel 0 ở mọi frame -> NaN -> 0
        fused = np.nanmedian(stack, axis=0)
    return np.nan_to_num(fused, nan=0.0)


class Camera:
    def __init__(self, k, dist):
        self.A = np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1]], np.float64)
        self.D = (np.array([dist[n] for n in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")])
                  if dist else None)

    def rays(self, uv):
        """Pixel (N,2) -> hướng tia chuẩn hoá (x/z, y/z, 1), đã sửa méo ống kính."""
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        if self.D is not None:
            xy = cv2.undistortPoints(uv, self.A, self.D).reshape(-1, 2)
        else:
            xy = (uv.reshape(-1, 2) - self.A[:2, 2]) / [self.A[0, 0], self.A[1, 1]]
        return np.c_[xy, np.ones(len(xy))]


def shrink(corners, k):
    c = np.float32(corners)
    m = c.mean(0)
    return m + (c - m) * k


def fit_plane(pts, iters=3):
    """Mặt phẳng n·p = d, loại điểm lệch > 2.5 MAD; trả về (n hướng về camera, d, inlier, std phần dư)."""
    keep = np.ones(len(pts), bool)
    for _ in range(iters):
        c = pts[keep].mean(0)
        n = np.linalg.svd(pts[keep] - c, full_matrices=False)[2][-1]
        r = (pts - c) @ n
        mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r) <= max(2.5 * mad, 5.0)
    if n[2] > 0:
        n = -n
    return n, float(n @ c), keep, float(r[keep].std())


def bag_surface(depth, cam, obj):
    """Mặt trên của một bao từ vùng lõi box; None nếu thiếu depth."""
    poly = shrink(obj["corners"], SHRINK)
    mask = np.zeros(depth.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    valid = z > 0
    if len(z) < 100 or valid.mean() < MIN_VALID:
        return None
    idx = np.flatnonzero(valid)
    if len(idx) > 6000:
        idx = np.random.default_rng(0).choice(idx, 6000, replace=False)
    pts = cam.rays(np.c_[xs[idx], ys[idx]]) * z[idx, None]
    n, d, inl, resid = fit_plane(pts)

    def on_plane(uv):
        ray = cam.rays(uv)
        return ray * (d / (ray @ n))[:, None]

    cx, cy = obj["cx"], obj["cy"]
    a = math.radians(obj["angle"])
    ux, uy = math.cos(a), math.sin(a)  # hướng cạnh dài trên ảnh (le90)
    center, e1, e2, f1, f2 = on_plane([[cx, cy],
                                       [cx - ux * obj["w"] / 2, cy - uy * obj["w"] / 2],
                                       [cx + ux * obj["w"] / 2, cy + uy * obj["w"] / 2],
                                       [cx + uy * obj["h"] / 2, cy - ux * obj["h"] / 2],
                                       [cx - uy * obj["h"] / 2, cy + ux * obj["h"] / 2]])
    axis = e2 - e1
    return {"xyz_cam": center, "normal": n, "plane_d": d, "axis": axis / np.linalg.norm(axis),
            "pts": pts[inl], "z_top": float(np.median(pts[inl, 2])), "resid": resid, "valid": float(valid.mean()),
            "tilt": math.degrees(math.acos(min(1.0, abs(n[2])))),
            "size_mm": (float(np.linalg.norm(axis)), float(np.linalg.norm(f2 - f1)))}


def contact_dz(a, sa, b, sb, cam, up, depth, grow=1.15):
    """Tại vùng giao của hai box nới rộng: (median độ cao mặt b - mặt a, độ cao depth đo được so với mặt a
    ở phân vị CONTACT_Q) theo hướng up. Bao b đè lên a thì ở chỗ tiếp xúc depth thật cao hơn mặt a, kể
    cả khi mép bao b phồng/rủ xuống. Không kéo dài mặt phẳng của b ra xa (bao nằm nghiêng dễ sai).
    None nếu hai box không chạm nhau hoặc thiếu depth ở đó."""
    pa, pb = shrink(a["corners"], grow).astype(np.float32), shrink(b["corners"], grow).astype(np.float32)
    area, poly = cv2.intersectConvexConvex(pa, pb)
    if poly is None or area < 0.005 * cv2.contourArea(pa):
        return None
    x0, y0 = np.floor(poly.reshape(-1, 2).min(0)).astype(int)
    x1, y1 = np.ceil(poly.reshape(-1, 2).max(0)).astype(int)
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1, 2), np.arange(y0, y1 + 1, 2))
    uv = np.c_[xs.ravel(), ys.ravel()]
    h, w = depth.shape
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
    inside = np.array([cv2.pointPolygonTest(poly, (float(u), float(v)), False) >= 0 for u, v in uv], bool)
    uv = uv[inside] if len(uv) else uv
    z = depth[uv[:, 1], uv[:, 0]] if len(uv) else np.zeros(0)
    uv, z = uv[z > 0], z[z > 0]
    if len(z) < 5:
        return None
    ray = cam.rays(uv)
    ha = (ray * (sa["plane_d"] / (ray @ sa["normal"]))[:, None]) @ up
    hb = (ray * (sb["plane_d"] / (ray @ sb["normal"]))[:, None]) @ up
    hobs = (ray * z[:, None]) @ up
    return float(np.median(hb - ha)), float(np.percentile(hobs - ha, CONTACT_Q))


def plan_image(image_path, depth_path, objects, cam_to_robot, default_dist, ignore_conf=False):
    k, dist, spec = load_meta(image_path)
    return plan_frame(load_depth(depth_path), Camera(k, dist or default_dist), objects, cam_to_robot,
                      float(spec.get("length_mm", BAG_MM[0])), ignore_conf)


def plan_frame(depth, cam, objects, cam_to_robot, bag_len=BAG_MM[0], ignore_conf=False):
    """Chọn bao từ depth (mm, đã khớp ảnh màu, 0 = không đo được) và box OBB của cùng khung hình.
    Dùng chung cho ảnh đã lưu (plan_image) và khung hình lấy thẳng từ camera."""
    h, w = depth.shape
    pallets = [o for o in objects if o["class_id"] == 0]
    if not pallets:
        return {"status": "không thấy pallet", "bags": []}
    marked = [p for p in pallets if p.get("target")]  # nhãn tay có cờ pallet mục tiêu
    target = marked[0] if marked else min(pallets, key=lambda p: math.hypot(p["cx"] - w / 2, p["cy"] - h / 2))
    tpoly = np.float32(target["corners"])
    bags = [o for o in objects if o["class_id"] == 1
            and cv2.pointPolygonTest(tpoly, (o["cx"], o["cy"]), False) >= 0
            and (ignore_conf or o.get("conf", 1.0) >= MIN_CONF)]

    if not bags:
        return {"status": "pallet rỗng", "target_pallet": {k2: target[k2] for k2 in ("cx", "cy", "w", "h", "angle")},
                "bags": []}
    rows = []
    for o in bags:
        s = bag_surface(depth, cam, o)
        rows.append({"obj": o, "s": s, "reasons": [] if s else ["thiếu depth"]})
    measured = [r for r in rows if r["s"]]
    if not measured:
        return {"status": "không đo được bao nào", "target_pallet": target, "bags": []}
    if cam_to_robot is not None:
        up = np.asarray(cam_to_robot, np.float64)[:3, :3].T @ [0.0, 0.0, 1.0]  # trục Z robot hướng lên
    else:
        up = -np.median([r["s"]["normal"] for r in measured], axis=0)  # normal hướng về camera = lên
    up = up / np.linalg.norm(up)
    if up[2] > 0:
        up = -up
    for r in measured:
        r["s"]["height"] = float(np.median(r["s"]["pts"] @ up))
    for r in measured:
        s = r["s"]
        r["bad"] = s["tilt"] > BAD_TILT or s["size_mm"][0] > 1.5 * bag_len or s["size_mm"][1] > 1.5 * bag_len
        if r["bad"]:
            r["reasons"].append("mặt đo không tin cậy")
    h_max = max(r["s"]["height"] for r in measured if not r["bad"]) if any(not r["bad"] for r in measured)         else max(r["s"]["height"] for r in measured)
    z_min = min(r["s"]["z_top"] for r in measured)
    for r in measured:
        s, o = r["s"], r["obj"]
        if s["height"] < h_max - LAYER_TOL:
            r["reasons"].append(f"lớp dưới ({h_max - s['height']:.0f} mm)")
        for other in measured:
            if other is r or other["bad"]:
                continue
            c = contact_dz(o, s, other["obj"], other["s"], cam, up, depth)
            if c and c[0] > 0 and c[1] > CONTACT_DZ:
                r["reasons"].append(f"bị bao #{other['obj'].get('id', '?')} đè ({c[1]:.0f} mm)")
                break
        if s["tilt"] > MAX_TILT:
            r["reasons"].append(f"nghiêng {s['tilt']:.0f}°")
        if s["resid"] > MAX_RESID:
            r["reasons"].append(f"mặt gồ ghề {s['resid']:.0f} mm")
        if s["size_mm"][0] < MIN_LONG_FRAC * bag_len:
            r["reasons"].append(f"chỉ thấy {s['size_mm'][0]:.0f} mm chiều dài")

    ok = [r for r in measured if not r["reasons"]]
    # Cao nhất trước; cùng độ cao (TIE_DZ) thì bao thấy được nhiều hơn rồi conf cao hơn.
    ok.sort(key=lambda r: (round((h_max - r["s"]["height"]) / TIE_DZ), -r["s"]["size_mm"][0] * r["s"]["size_mm"][1],
                           -r["obj"].get("conf", 1.0)))
    out = []
    for r in rows:
        o, s = r["obj"], r["s"]
        item = {"id": o.get("id"), "conf": o.get("conf"), "box_px": [round(o["cx"], 1), round(o["cy"], 1)],
                "rank": ok.index(r) + 1 if r in ok else None, "reasons": r["reasons"]}
        if s:
            item.update(xyz_cam_mm=np.round(s["xyz_cam"], 1).tolist(), z_top_mm=round(s["z_top"], 1),
                        below_top_mm=round(h_max - s["height"], 1),
                        normal=np.round(s["normal"], 3).tolist(), long_axis=np.round(s["axis"], 3).tolist(),
                        yaw_cam_deg=round(math.degrees(math.atan2(s["axis"][1], s["axis"][0])), 1),
                        tilt_deg=round(s["tilt"], 1), plane_resid_mm=round(s["resid"], 1),
                        size_mm=[round(v) for v in s["size_mm"]], depth_valid=round(s["valid"], 3))
            if cam_to_robot is not None:
                T = np.asarray(cam_to_robot, np.float64)
                item["xyz_robot_mm"] = np.round(T[:3, :3] @ s["xyz_cam"] + T[:3, 3], 1).tolist()
                item["long_axis_robot"] = np.round(T[:3, :3] @ s["axis"], 3).tolist()
        out.append(item)
    out.sort(key=lambda x: (x["rank"] is None, x["rank"] or 0, x.get("z_top_mm", 1e9)))
    status = "ok" if ok else "không có bao gắp được"
    return {"status": status, "target_pallet": {k2: target[k2] for k2 in ("cx", "cy", "w", "h", "angle")},
            "z_top_min_mm": round(z_min, 1), "up_cam": np.round(up, 3).tolist(), "bags": out}


def render(img, objects, plan, scale=0.5):
    """Vẽ kết quả lên ảnh màu (BGR): xanh lá = gắp trước, vàng = gắp được, đỏ = chưa gắp được,
    xanh dương nhạt = pallet mục tiêu. Chữ chỉ dùng ASCII (cv2.putText không có dấu tiếng Việt)."""
    img = img.copy()
    by_id = {o.get("id"): o for o in objects}
    if "target_pallet" in plan:
        tp = [o for o in objects if o["class_id"] == 0 and abs(o["cx"] - plan["target_pallet"]["cx"]) < 1e-6]
        for o in tp:
            cv2.polylines(img, [np.int32(o["corners"])], True, (255, 255, 0), 3)
    for b in plan["bags"]:
        o = by_id.get(b["id"])
        if o is None:
            continue
        color = (0, 220, 0) if b["rank"] == 1 else (0, 220, 220) if b["rank"] else (0, 0, 230)
        cv2.polylines(img, [np.int32(o["corners"])], True, color, 6 if b["rank"] == 1 else 3)
        x, y = int(o["cx"]), int(o["cy"])
        label = f"#{b['rank']}" if b["rank"] else "x"
        if b["rank"] == 1 and "xyz_cam_mm" in b:
            X, Y, Z = b["xyz_cam_mm"]
            label += f" ({X:.0f},{Y:.0f},{Z:.0f})mm yaw {b['yaw_cam_deg']:.0f}"
        cv2.putText(img, label, (x - 60, y), 0, 1.1, color, 3)
    return cv2.resize(img, None, fx=scale, fy=scale)


def height_image(image_path, depth_path, plan, default_dist=None, span=300.0, scale=0.5):
    """Bản đồ độ cao theo hướng "lên" của plan, tô màu: đỏ sẫm = cao nhất, xanh = thấp hơn span mm,
    đen = không có depth. Mốc 0 lấy theo phân vị 98% độ cao trong pallet mục tiêu."""
    k, dist, _ = load_meta(image_path)
    return height_map(load_depth(depth_path), Camera(k, dist or default_dist), plan, span, scale)


def height_map(depth, cam, plan, span=300.0, scale=0.5):
    """Như height_image nhưng nhận thẳng depth (mm) và Camera."""
    step = int(round(1 / scale))
    ys, xs = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    z = depth[ys, xs]
    pts = cam.rays(np.c_[xs.ravel(), ys.ravel()]) * z.reshape(-1, 1)
    up = np.asarray(plan.get("up_cam", [0.0, 0.0, -1.0]))
    h = (pts @ up).reshape(z.shape)
    valid = z > 0
    region = valid.copy()
    if "target_pallet" in plan:
        t = plan["target_pallet"]
        box = cv2.boxPoints(((t["cx"] / step, t["cy"] / step), (t["w"] / step, t["h"] / step), t["angle"]))
        mask = np.zeros(z.shape, np.uint8)
        cv2.fillPoly(mask, [np.int32(box)], 1)
        region &= mask > 0
    top = np.percentile(h[region], 98) if region.any() else (np.percentile(h[valid], 98) if valid.any() else 0.0)
    rel = np.clip((top - h) / span, 0, 1)
    vis = cv2.applyColorMap(((1 - rel) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis


def draw(image_path, objects, plan, out_path):
    img = cv2.imread(str(ROOT / image_path))
    cv2.imwrite(str(out_path), render(img, objects, plan))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--predictions", type=Path, help="predictions.json của predict.py")
    src.add_argument("--labels", action="store_true", help="dùng nhãn đã kiểm tra trong annotations.json")
    ap.add_argument("--camera", type=Path, default=HERE / "camera.json")
    ap.add_argument("--image", help="chỉ chạy một ảnh (id)")
    args = ap.parse_args()

    camera = json.loads(args.camera.read_text(encoding="utf-8"))
    if args.labels:
        doc = json.loads((ROOT / "code/obb_labeler/results/annotations.json").read_text(encoding="utf-8"))
        images = {im["id"]: im for im in doc["images"]}
        items = {i: {"image_path": images[i]["image_path"], "depth_path": images[i]["depth_path"],
                     "objects": r["objects"]} for i, r in doc["records"].items() if r.get("reviewed")}
        name = "labels"
    else:
        preds = json.loads(args.predictions.read_text(encoding="utf-8"))
        items = {}
        for i, p in preds.items():
            for n, o in enumerate(p["objects"], 1):
                o.setdefault("id", n)
            items[i] = {"image_path": p["image_path"], "depth_path": p.get("depth_path")
                        or re.sub(r"color(\.png)$", r"depth\1", p["image_path"]), "objects": p["objects"]}
        name = args.predictions.parent.name.removeprefix("predict_")
    if args.image:
        items = {args.image: items[args.image]}

    out = ROOT / "runs" / "grasp" / name
    (out / "vis").mkdir(parents=True, exist_ok=True)
    results = {}
    for image_id, it in items.items():
        plan = plan_image(it["image_path"], it["depth_path"], it["objects"], camera.get("cam_to_robot"),
                          camera.get("distortion"), ignore_conf=args.labels)
        results[image_id] = plan
        draw(it["image_path"], it["objects"], plan, out / "vis" / (image_id.replace("/", "__") + ".jpg"))
        first = next((b for b in plan["bags"] if b["rank"] == 1), None)
        msg = (f"gắp #{first['id']} tại {first['xyz_cam_mm']} mm, yaw {first['yaw_cam_deg']}°"
               if first else plan["status"])
        print(f"{image_id:40s} {msg}")
    (out / "picks.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
