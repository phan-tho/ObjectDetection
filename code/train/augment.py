"""Augment offline cho ảnh pallet/bao nhìn từ trên xuống.

v1 chỉ đổi giá trị điểm ảnh (nhãn giữ nguyên); v2 (scene) thêm dán bao, che khuất,
màng co; v3 (heavy) thêm biến đổi hình học có cắt nhãn theo khung ảnh, mẫu âm, sọc,
mất nét. Ultralytics vẫn xoay/lật/scale/mosaic online lúc train.

Mỗi biến đổi mô phỏng một tình huống thật ở kho:
- recolor_bags: bao loại khác màu. Cả pallet thường cùng một loại bao, nên mỗi ảnh
  chọn một màu đích cho mọi bao; chỉ đổi điểm ảnh gần màu thân bao (bỏ nền lọt vào box).
- color_temperature / brightness_contrast / gamma: đèn kho vàng/trắng, sáng/tối, phơi sáng.
- glare: đốm chói trên bao nilon/bạt + vùng cháy sáng.
- shadow: bóng cánh tay robot, người, xà nhà đổ lên pallet.
- noise / blur / motion_blur / jpeg / downscale: camera rẻ, rung, nén, độ phân giải thấp.
- vignette: tối góc ống kính.
"""
import cv2
import numpy as np

# (hue OpenCV 0-179, saturation 0-255, hệ số độ sáng). None = giữ hue gốc.
# Bao trong ảnh gốc chủ yếu trắng (V~255) nên màu đậm cần hệ số < 1.
BAG_PALETTE = {
    "red": (0, 190, 0.75), "orange": (12, 200, 0.85), "yellow": (26, 170, 0.95),
    "green": (60, 150, 0.7), "teal": (85, 140, 0.7), "blue": (110, 170, 0.65),
    "purple": (140, 120, 0.65), "white": (None, 15, 1.1), "gray": (None, 20, 0.75),
    "brown": (14, 120, 0.55), "black": (None, 25, 0.35),
}


def bag_mask(img, polys, shrink=0.92, color_tol=38.0):
    """Mask các điểm ảnh thuộc thân bao: trong OBB (co nhẹ) và gần màu trung vị của bao."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = np.zeros((h, w), np.float32)
    for poly in polys:
        # Chỉ tính trong khung bao quanh box (nhanh hơn nhiều so với tính cả ảnh mỗi bao).
        x0, y0 = np.maximum(np.floor(poly.min(0)).astype(int) - 12, 0)
        x1, y1 = np.minimum(np.ceil(poly.max(0)).astype(int) + 12, [w, h])
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        local = poly - [x0, y0]
        c = local.mean(0)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [((local - c) * shrink + c).astype(np.int32)], 1)
        mc = np.zeros_like(m)
        cv2.fillPoly(mc, [((local - c) * 0.5 + c).astype(np.int32)], 1)
        if mc.sum() < 20:
            continue
        roi = lab[y0:y1, x0:x1]
        ref = np.median(roi[mc.astype(bool)], axis=0)
        # Làm mượt khoảng cách màu trước khi so ngưỡng để mask không lốm đốm theo vân vải.
        dist = cv2.GaussianBlur(np.linalg.norm(roi - ref, axis=2), (0, 0), 4)
        mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1],
                                        m * np.clip((color_tol * 1.5 - dist) / (color_tol * 0.5), 0, 1))
    # Làm mềm biên để không lộ đường cắt.
    return cv2.GaussianBlur(mask, (0, 0), 3)


def recolor_bags(img, polys, rng, target=None):
    if not polys:
        return img
    name = target or rng.choice(list(BAG_PALETTE))
    hue, sat, vscale = BAG_PALETTE[name]
    mask = bag_mask(img, polys)[..., None]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    sel = mask[..., 0] > 0.5
    if not sel.any():
        return img
    # "Colorize": hue đích đồng nhất (bao trắng có hue gốc là nhiễu nên không dựa vào nó),
    # saturation theo màu đích, giữ độ sáng gốc để còn nếp nhăn/bóng của bao.
    out = hsv.copy()
    if hue is not None:
        out[..., 0] = (hue + rng.uniform(-5, 5)) % 180
    out[..., 1] = sat * rng.uniform(0.75, 1.15)
    out[..., 2] = np.clip(hsv[..., 2] * vscale * rng.uniform(0.85, 1.1), 0, 255)
    recolored = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return (img * (1 - mask) + recolored * mask).astype(np.uint8)


def color_temperature(img, rng):
    k = rng.uniform(-0.18, 0.18)  # âm: lạnh/xanh, dương: ấm/vàng (đèn sodium)
    gains = np.array([1 - k, 1 + 0.15 * rng.uniform(-1, 1) * abs(k), 1 + k], np.float32)
    return np.clip(img * gains, 0, 255).astype(np.uint8)


def brightness_contrast(img, rng):
    a = rng.uniform(0.65, 1.3)
    b = rng.uniform(-40, 30)
    return np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


def gamma(img, rng):
    g = rng.uniform(0.55, 1.8)
    lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def saturation(img, rng):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.4, 1.4)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def glare(img, rng):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for _ in range(rng.integers(1, 4)):
        cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        ax, ay = rng.uniform(0.04, 0.22) * w, rng.uniform(0.04, 0.22) * h
        t = rng.uniform(0, np.pi)
        dx, dy = xx - cx, yy - cy
        u = (dx * np.cos(t) + dy * np.sin(t)) / ax
        v = (-dx * np.sin(t) + dy * np.cos(t)) / ay
        blob = np.exp(-(u * u + v * v) * rng.uniform(1.0, 3.0))[..., None]
        out = out + blob * rng.uniform(60, 180)
    return np.clip(out, 0, 255).astype(np.uint8)


def shadow(img, rng):
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.float32)
    for _ in range(rng.integers(1, 3)):
        # Dải dài (cánh tay/xà) hoặc đa giác lớn.
        if rng.random() < 0.5:
            p0 = np.array([rng.uniform(0, w), rng.uniform(0, h)])
            ang = rng.uniform(0, np.pi)
            d = np.array([np.cos(ang), np.sin(ang)]) * max(w, h)
            n = np.array([-d[1], d[0]]) / max(w, h) * rng.uniform(0.04, 0.15) * w
            pts = np.array([p0 - d + n, p0 + d + n, p0 + d - n, p0 - d - n])
        else:
            c = np.array([rng.uniform(0, w), rng.uniform(0, h)])
            k = rng.integers(3, 7)
            ang = np.sort(rng.uniform(0, 2 * np.pi, k))
            r = rng.uniform(0.1, 0.4, k)[:, None] * np.array([w, h])
            pts = c + np.stack([np.cos(ang), np.sin(ang)], 1) * r
        cv2.fillPoly(m, [pts.astype(np.int32)], 1.0)
    m = cv2.GaussianBlur(m, (0, 0), rng.uniform(8, 40))[..., None]
    dark = rng.uniform(0.35, 0.75)
    return (img * (1 - m * (1 - dark))).astype(np.uint8)


def noise(img, rng):
    sigma = rng.uniform(4, 18)
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def blur(img, rng):
    return cv2.GaussianBlur(img, (0, 0), rng.uniform(0.8, 2.5))


def motion_blur(img, rng):
    k = int(rng.integers(7, 21))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1 / k
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    return cv2.filter2D(img, -1, kernel / max(kernel.sum(), 1e-6))


def jpeg(img, rng):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 75))])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def downscale(img, rng):
    h, w = img.shape[:2]
    f = rng.uniform(0.3, 0.6)
    small = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def vignette(img, rng):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    m = 1 - rng.uniform(0.2, 0.55) * np.clip(r - 0.4, 0, None) ** 2
    return np.clip(img * m[..., None], 0, 255).astype(np.uint8)


# ---- Augment có thay đổi nội dung cảnh (v2) ----

def bag_donors(img, objects):
    """Cắt thẳng các thân bao (không bị che/cắt biên) thành ảnh chữ nhật để dán sang chỗ khác."""
    donors = []
    for o in objects:
        if o["class_id"] != 1 or o["occluded"] or o["truncated"]:
            continue
        src = np.array(o["corners"], np.float32)
        w, h = np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[1])
        if min(w, h) < 20:
            continue
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        patch = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (int(w), int(h)))
        donors.append(patch)
    return donors


def paste_bags(img, objects, donors, rng, n_max=3, v5=False):
    """Dán 1–n bao rời lên nền trống (bao rơi, bao để cạnh pallet): thêm nhãn bag mới,
    đa dạng kích thước/góc/vị trí. Không đè lên pallet/bao đã có."""
    if not donors:
        return img, objects
    h, w = img.shape[:2]
    occupied = np.zeros((h, w), np.uint8)
    for o in objects:
        cv2.fillPoly(occupied, [np.array(o["corners"], np.int32)], 1)
    occupied = cv2.dilate(occupied, np.ones((15, 15), np.uint8))
    objects = list(objects)
    for _ in range(rng.integers(1, n_max + 1)):
        patch = donors[rng.integers(len(donors))]
        if v5:
            patch = deform_patch(patch, rng)
        ph, pw = patch.shape[:2]
        s = rng.uniform(0.6, 1.4) if v5 else rng.uniform(0.7, 1.25)  # v5: cả bao 25 kg lẫn 50 kg
        bw, bh = pw * s, ph * s
        for _try in range(30):
            cx, cy, a = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(-np.pi / 2, np.pi / 2)
            c, si = np.cos(a), np.sin(a)
            local = np.array([[-bw / 2, -bh / 2], [bw / 2, -bh / 2], [bw / 2, bh / 2], [-bw / 2, bh / 2]])
            dst = local @ np.array([[c, si], [-si, c]]) + [cx, cy]
            if dst.min() < 2 or dst[:, 0].max() > w - 2 or dst[:, 1].max() > h - 2:
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [dst.astype(np.int32)], 1)
            if (m & occupied).any():
                continue
            # Bóng đổ nhẹ lệch xuống dưới-phải cho bao nhìn như đặt trên nền.
            sh = cv2.GaussianBlur(np.roll(np.roll(m, 8, 0), 8, 1).astype(np.float32), (0, 0), 6)
            img = (img * (1 - 0.35 * sh[..., None])).astype(np.uint8)
            src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
            M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
            warped = cv2.warpPerspective(patch, M, (w, h))
            alpha = cv2.warpPerspective(np.ones((ph, pw), np.float32), M, (w, h))
            alpha = cv2.GaussianBlur(cv2.erode(alpha, np.ones((3, 3), np.uint8)), (0, 0), 1.5)[..., None]
            img = (img * (1 - alpha) + warped * alpha).astype(np.uint8)
            occupied |= cv2.dilate(m, np.ones((15, 15), np.uint8))
            objects.append({"class_id": 1, "corners": dst.tolist(), "pasted": True})
            break
    return img, objects


def arm_occluder(img, objects, rng):
    """Vật dài che một phần (tay người, cánh tay robot, ống/dây hơi) đi ngang pallet."""
    h, w = img.shape[:2]
    targets = [o for o in objects if o["class_id"] == 1] or objects
    if not targets:
        return img
    o = targets[rng.integers(len(targets))]
    c = np.array(o["corners"]).mean(0)
    a = rng.uniform(0, np.pi)
    length, width = rng.uniform(0.15, 0.4) * w, rng.uniform(0.02, 0.05) * w
    d = np.array([np.cos(a), np.sin(a)])
    start = c - d * length * rng.uniform(0.2, 0.6)
    m = np.zeros((h, w), np.uint8)
    cv2.line(m, tuple(start.astype(int)), tuple((start + d * length).astype(int)), 1, int(width))
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2)[..., None]
    color = rng.choice([[40, 40, 45], [70, 90, 160], [60, 60, 60], [200, 200, 200], [30, 160, 200]])
    tex = np.clip(np.array(color, np.float32) + rng.normal(0, 12, img.shape), 0, 255)
    shade = cv2.GaussianBlur(np.roll(np.roll(m[..., 0], 15, 0), 15, 1), (0, 0), 10)[..., None]
    img = img * (1 - 0.4 * shade)
    return (img * (1 - m) + tex * m).astype(np.uint8)


def stretch_wrap(img, objects, rng):
    """Màng co nilon quấn pallet: các dải trong mờ phản sáng + dây đai."""
    pallets = [o for o in objects if o["class_id"] == 0]
    if not pallets:
        return img
    h, w = img.shape[:2]
    p = pallets[rng.integers(len(pallets))]
    region = np.zeros((h, w), np.uint8)
    cv2.fillPoly(region, [np.array(p["corners"], np.int32)], 1)
    film = np.zeros((h, w), np.float32)
    pts = np.array(p["corners"])
    a = rng.uniform(0, np.pi)
    d, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c, span = pts.mean(0), np.linalg.norm(pts[2] - pts[0])
    for off in np.linspace(-span / 2, span / 2, rng.integers(4, 12)):
        q = c + n * (off + rng.normal(0, 10))
        cv2.line(film, tuple((q - d * span).astype(int)), tuple((q + d * span).astype(int)),
                 rng.uniform(0.3, 1.0), int(rng.uniform(4, 30)))
    film = cv2.GaussianBlur(film, (0, 0), rng.uniform(3, 8)) * region
    out = img.astype(np.float32)
    out = out * (1 - 0.35 * film[..., None]) + 235 * 0.35 * film[..., None]
    if rng.random() < 0.5:  # dây đai
        q = c + n * rng.uniform(-span / 3, span / 3)
        strap = np.zeros((h, w), np.float32)
        cv2.line(strap, tuple((q - d * span).astype(int)), tuple((q + d * span).astype(int)), 1, 6)
        strap = (strap * region)[..., None]
        out = out * (1 - strap) + np.array(rng.choice([[30, 30, 200], [200, 200, 200], [40, 40, 40]])) * strap
    return np.clip(out, 0, 255).astype(np.uint8)


# ---- Augment v3: nhắm vào lỗi thấy trên val (analyze_errors.py) ----
# Bao cắt ở mép ảnh lỗi 100%, bao nhỏ <1% ảnh lỗi 90%, báo nhầm cửa/kệ/áo ngoài pallet.

GRAY = 114  # màu tô vùng không còn nhãn (giống letterbox của YOLO)
MIN_VISIBLE = {0: 0.15, 1: 0.3}  # phần còn thấy tối thiểu để giữ nhãn (pallet / bao)


def clip_objects(img, objects):
    """Cắt mỗi OBB theo khung ảnh; nhãn mới = hình chữ nhật nhỏ nhất bao phần còn thấy
    (đúng quy ước gán nhãn: chỉ khoanh phần nhìn thấy). Bao còn quá ít thì tô xám và bỏ
    nhãn, để model không học "bao không nhãn = nền"."""
    h, w = img.shape[:2]
    frame = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    kept = []
    for o in objects:
        poly = np.float32(o["corners"])
        full = cv2.contourArea(poly)
        inter, pts = cv2.intersectConvexConvex(poly, frame)
        if full <= 0 or inter <= 1 or pts is None:
            continue
        if inter / full < MIN_VISIBLE[o["class_id"]]:
            cv2.fillPoly(img, [pts.reshape(-1, 2).astype(np.int32)], (GRAY, GRAY, GRAY))
            continue
        if inter / full > 0.98:
            kept.append(o)
            continue
        box = cv2.boxPoints(cv2.minAreaRect(pts.reshape(-1, 2)))
        kept.append({**o, "corners": box.tolist(), "truncated": True})
    return img, kept


def floor_texture(img, shape, rng):
    """Nền giả (sàn kho) để lấp chỗ trống sau khi xoay/thu nhỏ; không dùng ảnh phản chiếu
    vì sẽ sinh bao không có nhãn."""
    border = np.concatenate([img[:20].reshape(-1, 3), img[-20:].reshape(-1, 3)])
    base = np.median(border, 0) * rng.uniform(0.7, 1.1)
    tex = base + rng.normal(0, 18, (shape[0] // 8 + 1, shape[1] // 8 + 1, 3))
    tex = cv2.resize(tex.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return np.clip(tex + rng.normal(0, 6, (shape[0], shape[1], 3)), 0, 255).astype(np.uint8)


def warp(img, objects, M, rng):
    """Áp ma trận 3x3 cho ảnh và 4 góc mỗi box, lấp nền, rồi cắt nhãn theo khung ảnh."""
    h, w = img.shape[:2]
    out = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    valid = cv2.warpPerspective(np.ones((h, w), np.float32), M, (w, h))[..., None]
    out = (out * valid + floor_texture(img, (h, w), rng) * (1 - valid)).astype(np.uint8)
    moved = []
    for o in objects:
        pts = cv2.perspectiveTransform(np.float32(o["corners"])[None], M)[0]
        # Phối cảnh làm hình chữ nhật thành tứ giác: lấy hình chữ nhật nhỏ nhất bao nó.
        moved.append({**o, "corners": cv2.boxPoints(cv2.minAreaRect(pts)).tolist()})
    return clip_objects(out, moved)


def crop_zoom(img, objects, rng):
    """Phóng to một vùng: pallet bị cắt ở mép ảnh, bao to hơn (camera thấp/lệch)."""
    h, w = img.shape[:2]
    s = rng.uniform(0.5, 0.85)
    x0, y0 = rng.uniform(0, w * (1 - s)), rng.uniform(0, h * (1 - s))
    M = np.array([[1 / s, 0, -x0 / s], [0, 1 / s, -y0 / s], [0, 0, 1]])
    return warp(img, objects, M, rng)


def zoom_out(img, objects, rng):
    """Thu nhỏ cả cảnh: bao nhỏ (camera cao, pallet xa)."""
    h, w = img.shape[:2]
    s = rng.uniform(0.45, 0.8)
    tx, ty = rng.uniform(0, w * (1 - s)), rng.uniform(0, h * (1 - s))
    M = np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]])
    return warp(img, objects, M, rng)


def rotate_scene(img, objects, rng):
    """Xoay cả ảnh góc bất kỳ quanh một điểm gần tâm: pallet nằm chéo, góc bị cắt khỏi khung."""
    h, w = img.shape[:2]
    c = (w / 2 + rng.uniform(-0.15, 0.15) * w, h / 2 + rng.uniform(-0.15, 0.15) * h)
    M = np.vstack([cv2.getRotationMatrix2D(c, rng.uniform(-180, 180), rng.uniform(0.8, 1.15)), [0, 0, 1]])
    return warp(img, objects, M, rng)


def tilt(img, objects, rng):
    """Camera không vuông góc sàn: phối cảnh nhẹ (lệch mỗi góc tới 8% kích thước)."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + rng.uniform(-0.08, 0.08, (4, 2)) * [w, h]).astype(np.float32)
    return warp(img, objects, cv2.getPerspectiveTransform(src, dst), rng)


GEOMETRIC = [(crop_zoom, 0.3), (zoom_out, 0.2), (rotate_scene, 0.25), (tilt, 0.25)]


def clutter_donors(img, objects, rng, n=6, max_frac=0.25):
    """Mẫu âm: cắt vùng không chạm nhãn nào (cửa, kệ, người, thùng carton...)."""
    h, w = img.shape[:2]
    busy = np.zeros((h, w), np.uint8)
    for o in objects:
        cv2.fillPoly(busy, [np.array(o["corners"], np.int32)], 1)
    busy = cv2.dilate(busy, np.ones((25, 25), np.uint8))
    busy |= (np.abs(img.astype(int) - GRAY).sum(2) < 6).astype(np.uint8)  # vùng đã tô xám
    patches = []
    for _ in range(n * 10):
        pw, ph = int(rng.uniform(0.06, max_frac) * w), int(rng.uniform(0.06, max_frac) * h)
        x, y = int(rng.uniform(0, w - pw)), int(rng.uniform(0, h - ph))
        if busy[y:y + ph, x:x + pw].any():
            continue
        patch = img[y:y + ph, x:x + pw]
        if patch.std() > 18:  # bỏ mảng sàn trơn, cần vật có cấu trúc
            patches.append(patch.copy())
        if len(patches) == n:
            break
    return patches


def paste_clutter(img, objects, clutter, rng, n_max=3):
    """Dán vật nền vào chỗ trống, KHÔNG thêm nhãn: dạy model đây không phải bao/pallet."""
    if not clutter:
        return img
    h, w = img.shape[:2]
    occupied = np.zeros((h, w), np.uint8)
    for o in objects:
        cv2.fillPoly(occupied, [np.array(o["corners"], np.int32)], 1)
    occupied = cv2.dilate(occupied, np.ones((15, 15), np.uint8))
    for _ in range(rng.integers(1, n_max + 1)):
        patch = clutter[rng.integers(len(clutter))]
        if rng.random() < 0.5:
            patch = cv2.rotate(patch, rng.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180,
                                                  cv2.ROTATE_90_COUNTERCLOCKWISE]))
        ph, pw = patch.shape[:2]
        if ph >= h or pw >= w:
            continue
        for _try in range(20):
            x, y = int(rng.uniform(0, w - pw)), int(rng.uniform(0, h - ph))
            if not occupied[y:y + ph, x:x + pw].any():
                alpha = np.ones((ph, pw), np.float32)
                alpha = cv2.GaussianBlur(cv2.copyMakeBorder(alpha[4:-4, 4:-4], 4, 4, 4, 4, cv2.BORDER_CONSTANT),
                                         (0, 0), 2)[..., None]
                roi = img[y:y + ph, x:x + pw]
                img[y:y + ph, x:x + pw] = (roi * (1 - alpha) + patch * alpha).astype(np.uint8)
                occupied[y:y + ph, x:x + pw] = 1
                break
    return img


def stripes(img, rng):
    """Sọc: đèn LED nhấp nháy + màn trập cuộn (dải sáng tối song song), hoặc vạch lỗi cảm biến."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    if rng.random() < 0.6:
        period, amp = rng.uniform(15, 140), rng.uniform(0.06, 0.25)
        phase, a = rng.uniform(0, 2 * np.pi), rng.choice([0, np.pi / 2, rng.uniform(0, np.pi)])
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        t = xx * np.sin(a) + yy * np.cos(a)
        out *= (1 + amp * np.sin(2 * np.pi * t / period + phase))[..., None]
    else:
        vertical = rng.random() < 0.5
        for _ in range(rng.integers(2, 12)):
            pos, thick = int(rng.uniform(0, w if vertical else h)), int(rng.integers(1, 5))
            val = rng.uniform(-90, 90)
            if vertical:
                out[:, pos:pos + thick] += val
            else:
                out[pos:pos + thick] += val
    return np.clip(out, 0, 255).astype(np.uint8)


def defocus(img, rng):
    """Mất nét (sai tiêu cự): nhân đĩa tròn, khác blur Gauss."""
    r = int(rng.integers(3, 9))
    kernel = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(kernel, (r, r), r, 1, -1)
    return cv2.filter2D(img, -1, kernel / kernel.sum())


def grayscale(img, rng):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def channel_shuffle(img, rng):
    return np.ascontiguousarray(img[..., rng.permutation(3)])


def cutout(img, rng):
    """Vài mảng nhỏ che ngẫu nhiên (vật lạ, giấy, băng keo): không đổi nhãn."""
    h, w = img.shape[:2]
    for _ in range(rng.integers(1, 5)):
        cw, ch = int(rng.uniform(0.02, 0.07) * w), int(rng.uniform(0.02, 0.07) * h)
        x, y = int(rng.uniform(0, w - cw)), int(rng.uniform(0, h - ch))
        img[y:y + ch, x:x + cw] = rng.integers(0, 255, 3) if rng.random() < 0.5 \
            else rng.integers(0, 255, (ch, cw, 3))
    return img


# ---- Augment v4: nhắm vào case kém khi chấm cả 25 ảnh (analyze_errors.py, last.pt) ----
# Bao bị bao khác đè lỗi 45% (không bị che 16%); người/áo/vạch sơn/cửa bị nhận nhầm;
# pallet rỗng bị sót (cả bộ chỉ vài pallet rỗng).

def blend_patch(img, patch, dst, shadow=0.35):
    """Dán patch chữ nhật vào tứ giác dst (pixel) kèm bóng đổ nhẹ xuống dưới-phải."""
    h, w = img.shape[:2]
    ph, pw = patch.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [dst.astype(np.int32)], 1)
    sh = cv2.GaussianBlur(np.roll(np.roll(m, 8, 0), 8, 1).astype(np.float32), (0, 0), 6)
    img = (img * (1 - shadow * sh[..., None])).astype(np.uint8)
    M = cv2.getPerspectiveTransform(np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]), dst.astype(np.float32))
    warped = cv2.warpPerspective(patch, M, (w, h))
    alpha = cv2.warpPerspective(np.ones((ph, pw), np.float32), M, (w, h))
    alpha = cv2.GaussianBlur(cv2.erode(alpha, np.ones((3, 3), np.uint8)), (0, 0), 1.5)[..., None]
    return (img * (1 - alpha) + warped * alpha).astype(np.uint8)


def rect_at(center, size, angle):
    (cx, cy), (bw, bh), c, s = center, size, np.cos(angle), np.sin(angle)
    local = np.array([[-bw / 2, -bh / 2], [bw / 2, -bh / 2], [bw / 2, bh / 2], [-bw / 2, bh / 2]])
    return local @ np.array([[c, s], [-s, c]]) + [cx, cy]


def occlude_bags(img, objects, donors, rng, n_max=2, v5=False):
    """Bao đè lên bao (xếp lớp, bao lệch trên đỉnh pallet): dán bao thật đè một phần lên bao có
    sẵn. Bao bên dưới đổi nhãn thành hình chữ nhật bao phần còn thấy, đúng quy ước gán nhãn;
    còn dưới 25% thì bỏ nhãn vì gần như bị che hết. Pallet chứa nó nới ra theo bao mới
    (vùng pallet gồm cả bao tràn biên)."""
    if not donors or not any(o["class_id"] == 1 for o in objects):
        return img, objects
    h, w = img.shape[:2]
    for _ in range(rng.integers(1, n_max + 1)):
        bags = [o for o in objects if o["class_id"] == 1]
        if not bags:
            break
        base = np.array(bags[rng.integers(len(bags))]["corners"])
        c = base.mean(0)
        bw, bh = np.linalg.norm(base[1] - base[0]), np.linalg.norm(base[2] - base[1])
        patch = donors[rng.integers(len(donors))]
        if v5:
            patch = deform_patch(patch, rng)
        ph, pw = patch.shape[:2]
        s = rng.uniform(0.85, 1.1) * np.sqrt(bw * bh / (pw * ph))  # cỡ như bao bên dưới
        d = rng.uniform(0, 2 * np.pi)
        center = c + np.array([np.cos(d), np.sin(d)]) * rng.uniform(0.3, 0.6) * min(bw, bh)
        dst = rect_at(center, (pw * s, ph * s), rng.uniform(-np.pi / 2, np.pi / 2))
        if dst.min() < 0 or dst[:, 0].max() > w or dst[:, 1].max() > h:
            continue
        top = np.zeros((h, w), np.uint8)
        cv2.fillPoly(top, [dst.astype(np.int32)], 1)
        img = blend_patch(img, patch, dst, shadow=0.0 if v5 else 0.35)
        new, under = [], []
        for o in objects:
            poly = np.array(o["corners"])
            if o["class_id"] == 0:
                if cv2.pointPolygonTest(np.float32(poly), tuple(map(float, c)), False) >= 0:
                    o = {**o, "corners": cv2.boxPoints(cv2.minAreaRect(np.float32(np.vstack([poly, dst])))).tolist()}
                new.append(o)
                continue
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [poly.astype(np.int32)], 1)
            area = int(m.sum())
            vis = m & (1 - top)
            ratio = vis.sum() / area if area else 0
            if ratio <= 0.98:
                under.append(poly)
            if ratio > 0.98:
                new.append(o)
            elif ratio >= 0.25:
                cnts, _ = cv2.findContours(vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                big = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
                new.append({**o, "corners": cv2.boxPoints(cv2.minAreaRect(big)).tolist(), "occluded": True})
        new.append({"class_id": 1, "corners": dst.tolist(), "occluded": False, "truncated": False, "pasted": True})
        objects = new
        if v5 and under:
            img = layer_shading(img, under, dst, rng)
    return img, objects


def empty_pallet_donors(img, objects):
    """Pallet không có bao nào (pallet rỗng), không bị cắt biên: cắt thẳng thành ảnh chữ nhật."""
    parents = {o.get("parent_id") for o in objects if o["class_id"] == 1}
    out = []
    for o in objects:
        if o["class_id"] != 0 or o.get("id") in parents or o.get("truncated"):
            continue
        src = np.array(o["corners"], np.float32)
        w, h = np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[1])
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        out.append(cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (int(w), int(h))))
    return out


def paste_empty_pallet(img, objects, pallets, rng):
    """Thêm một pallet rỗng (cỡ 0.5–1.0×, góc tùy ý) vào chỗ trống, có nhãn pallet."""
    if not pallets:
        return img, objects
    h, w = img.shape[:2]
    occupied = np.zeros((h, w), np.uint8)
    for o in objects:
        cv2.fillPoly(occupied, [np.array(o["corners"], np.int32)], 1)
    patch = pallets[rng.integers(len(pallets))]
    ph, pw = patch.shape[:2]
    for _ in range(30):
        s = rng.uniform(0.5, 1.0)
        dst = rect_at((rng.uniform(0, w), rng.uniform(0, h)), (pw * s, ph * s), rng.uniform(-np.pi / 2, np.pi / 2))
        if dst.min() < 0 or dst[:, 0].max() > w or dst[:, 1].max() > h:
            continue
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [dst.astype(np.int32)], 1)
        if (m & occupied).any():
            continue
        return blend_patch(img, patch, dst, shadow=0.2), objects + [
            {"class_id": 0, "corners": dst.tolist(), "occluded": False, "truncated": False, "pasted": True}]
    return img, objects


# ---- Augment v5: ánh sáng theo từng bao, theo lớp xếp (đo trên ảnh thật) ----
# Bao lớp dưới (bị đè) tối hơn bao lớp trên 5–15% (bao xanh 111–126 vs 126–144; session_058 197 vs
# 232), có dải bóng dọc mép tiếp giáp. Bao đỏ tháng 8 gần như cháy sáng (205–246). Mỗi bao sáng/màu
# hơi khác nhau; bao nilon có vệt bóng loáng; bao phồng/nhăn chứ không phẳng.

def poly_mask(shape, poly, feather=0.0):
    m = np.zeros(shape[:2], np.float32)
    cv2.fillPoly(m, [np.asarray(poly).astype(np.int32)], 1.0)
    return cv2.GaussianBlur(m, (0, 0), feather) if feather > 0 else m


def deform_patch(patch, rng, amount=0.04):
    """Bao phồng/nhăn: dịch điểm ảnh theo sóng sin ngẫu nhiên + co phình giữa bao, biên độ ~4%."""
    h, w = patch.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / max(w - 1, 1), yy / max(h - 1, 1)
    dx = amount * w * (rng.uniform(-1, 1) * np.sin(np.pi * v) * (u - 0.5)
                       + 0.3 * rng.uniform(-1, 1) * np.sin(2 * np.pi * (v * rng.uniform(1, 3) + rng.uniform(0, 1))))
    dy = amount * h * (rng.uniform(-1, 1) * np.sin(np.pi * u) * (v - 0.5)
                       + 0.3 * rng.uniform(-1, 1) * np.sin(2 * np.pi * (u * rng.uniform(1, 3) + rng.uniform(0, 1))))
    out = cv2.remap(patch, (xx + dx).astype(np.float32), (yy + dy).astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT)
    # Phồng giữa bao sáng hơn, rìa tối hơn (bao căng tròn dưới đèn trên cao).
    bulge = 1 + rng.uniform(0, 0.12) * (1 - ((u - 0.5) ** 2 + (v - 0.5) ** 2) * 4)
    return np.clip(out * bulge[..., None], 0, 255).astype(np.uint8)


def layer_shading(img, under_polys, top_poly, rng):
    """Bao dưới tối đi (đều + dải bóng tiếp xúc dọc mép bao trên), bao trên sáng hơn.
    30% không đổi sáng, để model phải học mép bao chứ không chỉ chênh sáng."""
    if rng.random() < 0.3:
        return img
    h, w = img.shape[:2]
    top = poly_mask(img.shape, top_poly)
    under = np.zeros(img.shape[:2], np.float32)
    for p in under_polys:
        under = np.maximum(under, poly_mask(img.shape, p))
    under *= 1 - top
    tw = np.linalg.norm(np.asarray(top_poly)[1] - np.asarray(top_poly)[0])
    # Bóng: mask bao trên dịch theo hướng đèn, độ dài theo độ dày bao (~6–15% cạnh bao).
    a = rng.uniform(0, 2 * np.pi)
    d = rng.uniform(0.06, 0.15) * tw
    M = np.float32([[1, 0, d * np.cos(a)], [0, 1, d * np.sin(a)]])
    cast = cv2.GaussianBlur(cv2.warpAffine(top, M, (w, h)), (0, 0), max(d / 3, 2)) * (1 - top)
    gain = 1 - under * rng.uniform(0.05, 0.25) - cast * rng.uniform(0.15, 0.4)
    gain = gain * (1 + top * rng.uniform(0, 0.15))
    return np.clip(img * gain[..., None], 0, 255).astype(np.uint8)


def per_bag_jitter(img, objects, rng):
    """Mỗi bao sáng/màu khác nhau một chút (bao cũ, bụi, phai, lô khác)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    for o in objects:
        if o["class_id"] != 1:
            continue
        m = poly_mask(img.shape, o["corners"], feather=2)
        hsv[..., 0] = (hsv[..., 0] + m * rng.uniform(-4, 4)) % 180
        hsv[..., 1] *= 1 + m * rng.uniform(-0.25, 0.25)
        hsv[..., 2] *= 1 + m * rng.uniform(-0.15, 0.15)
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)


def mixed_recolor(img, objects, rng):
    """Pallet trộn hai loại bao: đổi màu khoảng nửa số bao sang một màu khác."""
    picked = [np.array(o["corners"]) for o in objects if o["class_id"] == 1 and rng.random() < 0.5]
    return recolor_bags(img, picked, rng) if picked else img


def bag_specular(img, objects, rng):
    """Vệt bóng loáng của bao nilon/PP dưới đèn: dải sáng dài chạy dọc thân 1–3 bao."""
    bags = [o for o in objects if o["class_id"] == 1]
    if not bags:
        return img
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for k in rng.choice(len(bags), min(len(bags), int(rng.integers(1, 4))), replace=False):
        p = np.array(bags[k]["corners"])
        c = p.mean(0) + (p[2] - p[0]) * rng.uniform(-0.25, 0.25, 2) * 0.5
        ax = p[1] - p[0] if rng.random() < 0.5 else p[2] - p[1]
        length = np.linalg.norm(ax)
        t = ax / max(length, 1)
        along = (xx - c[0]) * t[0] + (yy - c[1]) * t[1]
        across = -(xx - c[0]) * t[1] + (yy - c[1]) * t[0]
        streak = np.exp(-(along / (0.4 * length)) ** 2 - (across / (rng.uniform(0.03, 0.1) * length)) ** 2)
        out += (streak * poly_mask(img.shape, p))[..., None] * rng.uniform(50, 140)
    return np.clip(out, 0, 255).astype(np.uint8)


def overexpose(img, rng):
    """Cháy sáng (như nhóm bao đỏ): tăng phơi sáng, cắt trắng, quầng sáng quanh vùng cháy."""
    out = img.astype(np.float32) * rng.uniform(1.3, 1.7)
    bloom = cv2.GaussianBlur(np.clip(out - 255, 0, None), (0, 0), 15)
    return np.clip(out + 0.5 * bloom, 0, 255).astype(np.uint8)


# (hàm, xác suất). recolor đứng đầu vì cần màu gốc để tách thân bao.
LIGHTING = [(color_temperature, 0.5), (brightness_contrast, 0.6), (gamma, 0.35),
            (saturation, 0.35), (glare, 0.35), (shadow, 0.35), (vignette, 0.2)]
SENSOR = [(noise, 0.3), (blur, 0.2), (motion_blur, 0.15), (jpeg, 0.3), (downscale, 0.15)]
SENSOR_V3 = SENSOR + [(stripes, 0.25), (defocus, 0.15)]
COLOR_V3 = [(grayscale, 0.08), (channel_shuffle, 0.08), (cutout, 0.25)]


def augment(img, objects, rng, donors=None, p_recolor=0.7, scene=False, heavy=False, clutter=None,
            v4=False, pallets=None, v5=False):
    """objects: list dict có class_id và corners (pixel). Trả về (ảnh, objects).
    scene=True bật augment v2 (dán bao, che khuất, màng co).
    heavy=True bật thêm v3: biến đổi hình học có cắt nhãn theo khung, mẫu âm, sọc/mất nét.
    v4=True (kèm v3) thêm bao đè bao, pallet rỗng, mẫu âm dày hơn; pallets = empty_pallet_donors.
    v5=True (kèm v4) thêm ánh sáng theo lớp xếp, màu/sáng từng bao, vệt bóng loáng, cháy sáng,
    bao phồng/nhăn, cỡ bao 0.6–1.4×."""
    v4 |= v5
    img = img.copy()
    if scene:
        if rng.random() < 0.25:
            img = stretch_wrap(img, objects, rng)
        if donors and rng.random() < 0.5:
            img, objects = paste_bags(img, objects, donors, rng, n_max=4 if heavy else 3, v5=v5)
    if v4:
        if donors and rng.random() < 0.5:
            img, objects = occlude_bags(img, objects, donors, rng, v5=v5)
        if rng.random() < 0.3:
            img, objects = paste_empty_pallet(img, objects, pallets, rng)
    if heavy and clutter and rng.random() < (0.7 if v4 else 0.5):
        img = paste_clutter(img, objects, clutter, rng, n_max=5 if v4 else 3)
    bag_polys = [np.array(o["corners"]) for o in objects if o["class_id"] == 1]
    if rng.random() < p_recolor:
        img = recolor_bags(img, bag_polys, rng)
    if v5:
        if rng.random() < 0.2:
            img = mixed_recolor(img, objects, rng)
        if rng.random() < 0.5:
            img = per_bag_jitter(img, objects, rng)
        if rng.random() < 0.3:
            img = bag_specular(img, objects, rng)
    if scene and rng.random() < 0.25:
        img = arm_occluder(img, objects, rng)
    if heavy:
        # Tối đa một phép hình học offline; Ultralytics còn xoay/lật/scale online.
        geo = [fn for fn, p in GEOMETRIC if rng.random() < p]
        if geo:
            img, objects = geo[rng.integers(len(geo))](img, objects, rng)
        for fn, p in COLOR_V3:
            if rng.random() < p:
                img = fn(img, rng)
    # Tối đa 3 phép ánh sáng: chồng nhiều phép làm ảnh cháy trắng, không thực tế.
    lighting = [fn for fn, p in LIGHTING if rng.random() < p]
    if len(lighting) > 3:
        lighting = [lighting[i] for i in sorted(rng.choice(len(lighting), 3, replace=False))]
    for fn in lighting:
        img = fn(img, rng)
    if v5 and rng.random() < 0.12:
        img = overexpose(img, rng)
    # Nhiễu cảm biến tối đa 2 loại để ảnh không nát quá.
    sensor = [fn for fn, p in (SENSOR_V3 if heavy else SENSOR) if rng.random() < p]
    for fn in rng.permutation(sensor)[:2]:
        img = fn(img, rng)
    return img, objects
