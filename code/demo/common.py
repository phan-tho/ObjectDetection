"""Phần dùng chung của hai trang: dữ liệu lượt chụp, model, suy luận (có cache)."""
import contextlib
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
for sub in ("code/obb_labeler", "code/train", "code/grasp"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))
from pick_xyz import height_image, plan_image  # noqa: E402

CONFIG = json.loads((ROOT / "code/demo/models.json").read_text(encoding="utf-8"))
CAMERA = json.loads((ROOT / "code/grasp/camera.json").read_text(encoding="utf-8"))

# Thứ tự = thứ tự nên xem khi demo. "look" là điều nên để ý ở chuỗi đó.
GROUPS = {
    "september_pallet_1": {
        "name": "Tháng 9 – dỡ hàng dần đến khi pallet rỗng", "icon": ":material/conveyor_belt:",
        "about": "Cùng một pallet, chụp lại sau mỗi lần dỡ bớt bao, từ lúc đầy đến lúc rỗng.",
        "look": "Bấm Sau qua từng lượt: mỗi lượt hệ thống chỉ ra bao gắp tiếp theo, đến khi báo pallet rỗng."},
    "august_green_and_empty": {
        "name": "Tháng 8 – bao xanh, pallet vơi, bao đè chéo", "icon": ":material/layers:",
        "about": "Bao xanh, pallet từ đầy đến vơi rồi rỗng. Có bao nằm chéo đè lên bao khác.",
        "look": "Cột “Lý do chưa gắp” và bản đồ độ cao: bao bị đè không được chọn dù nằm gần đỉnh."},
    "august_red": {
        "name": "Tháng 8 – bao đỏ, ánh sáng mạnh", "icon": ":material/wb_sunny:",
        "about": "Bao đỏ, pallet đầy, nắng chiếu mạnh, có chồng hàng khác bên cạnh.",
        "look": "Model vẫn tìm đủ bao dù ảnh chói và có hàng không liên quan sát bên."},
    "september_pallet_2": {
        "name": "Tháng 9 – pallet 2, nhiều pallet cạnh nhau", "icon": ":material/view_column:",
        "about": "Nhiều pallet đứng cạnh nhau trong cùng khung hình.",
        "look": "Chỉ pallet mục tiêu ở giữa ảnh (khung xanh dương nhạt) được xét, bao ở pallet khác bị bỏ qua."},
    "september_pallet_3": {
        "name": "Tháng 9 – pallet 3, bao trắng/xanh, màng co", "icon": ":material/inventory_2:",
        "about": "Bao trắng và xanh, có màng co bọc, cạnh thùng carton.",
        "look": "Bao lớp dưới lộ ra ở mép được nhận ra nhưng bị loại vì thấp hơn lớp trên cùng."},
    "september_pallet_4": {
        "name": "Tháng 9 – pallet 4, bao nâu", "icon": ":material/inventory:",
        "about": "Bao nâu, màu gần với nền và pallet gỗ.",
        "look": "Bao khó phân biệt với nền bằng mắt, xem box có khít mép bao không."},
}
STATE_NAMES = {"full": "đầy", "partial": "vơi", "few": "còn ít bao", "empty": "rỗng"}


@st.cache_data
def load_images():
    """Lượt chụp theo chuỗi, xếp theo thời gian; kèm nhãn tay nếu ảnh đã được kiểm tra."""
    sel = json.loads((ROOT / "code/obb_labeler/selection.json").read_text(encoding="utf-8"))["images"]
    doc = json.loads((ROOT / "code/obb_labeler/results/annotations.json").read_text(encoding="utf-8"))
    groups = {}
    for im in sorted(sel, key=lambda i: i["captured_at"]):
        rec = doc["records"].get(im["id"], {})
        groups.setdefault(im["sequence_group"], []).append(
            {**im, "labels": rec.get("objects") if rec.get("reviewed") else None})
    return groups


def fold_of(group):
    return next(f for f, v in CONFIG["folds"].items() if group in v["groups"])


@st.cache_resource
def load_weights(path):
    from ultralytics import YOLO
    return YOLO(ROOT / path)


def load_model(fold):
    return load_weights(CONFIG["folds"][fold]["weights"])


@st.cache_data(max_entries=64)
def analyze(image_id, image_path, depth_path, fold, tta):
    from predict import detect
    img = cv2.imread(str(ROOT / image_path))
    t0 = time.perf_counter()
    objects = detect([load_model(fold)], img, CONFIG["imgsz"], tta=tta)
    t1 = time.perf_counter()
    for n, o in enumerate(objects, 1):
        o["id"] = n
    plan = plan_image(image_path, depth_path, objects, CAMERA.get("cam_to_robot"), CAMERA.get("distortion"))
    t2 = time.perf_counter()
    return {"objects": objects, "plan": plan, "detect_ms": (t1 - t0) * 1000, "plan_ms": (t2 - t1) * 1000,
            "height": height_image(image_path, depth_path, plan, CAMERA.get("distortion"))}


def missing_files():
    weights = [f["weights"] for f in CONFIG["folds"].values() if not (ROOT / f["weights"]).is_file()]
    images = [g for g, ims in load_images().items() if not (ROOT / ims[0]["image_path"]).is_file()]
    return weights, images


# Widget mất state khi sang trang khác, nên giữ bản sao ở khoá "_<key>" và gán lại trước khi vẽ widget.
def keep(key):
    st.session_state["_" + key] = st.session_state[key]


def persist(key, default):
    """Gọi ngay trước widget có key này, widget nhận on_change=keep, args=(key,)."""
    st.session_state[key] = st.session_state.get("_" + key, default)


def open_group(group):
    """Chọn chuỗi chụp cho trang demo (giá trị giữ qua lúc chuyển trang), về lượt đầu."""
    st.session_state["_group"] = group
    st.session_state[f"_shot_{group}"] = 1


@st.cache_resource
def _source_slot():
    return {"key": None, "src": None, "lock": threading.Lock()}


def open_source(kind, **opts):
    """Nguồn ảnh dùng chung cho cả app (một camera chỉ mở được một lần). Đổi nguồn thì đóng nguồn cũ.
    kind: "orbbec" | "webcam" (opts: index) | "replay" (opts: group, period)."""
    from camera import OrbbecSource, ReplaySource, WebcamSource
    slot = _source_slot()
    key = (kind, tuple(sorted(opts.items())))
    with slot["lock"]:
        if slot["key"] != key:
            close_source(locked=True)
            dist = CAMERA.get("distortion")
            if kind == "orbbec":
                src = OrbbecSource(dist)
            elif kind == "webcam":
                src = WebcamSource(opts["index"])
            else:
                src = ReplaySource(load_images()[opts["group"]], opts["period"], dist)
            slot["src"], slot["key"] = src, key
        return slot["src"]


def close_source(locked=False):
    slot = _source_slot()
    with contextlib.nullcontext() if locked else slot["lock"]:
        if slot["src"] is not None:
            slot["src"].close()
        slot["src"], slot["key"] = None, None


def legend(show_labels=False):
    with st.container(horizontal=True, gap="small"):
        st.badge("Gắp trước", icon=":material/looks_one:", color="green")
        st.badge("Gắp được", icon=":material/check:", color="yellow")
        st.badge("Chưa gắp được", icon=":material/close:", color="red")
        st.badge("Pallet mục tiêu", icon=":material/crop_free:", color="blue")
        if show_labels:
            st.badge("Nhãn tay", icon=":material/edit:", color="gray")


def bag_table(bags):
    """Bảng các bao của plan: thứ tự gắp, toạ độ, góc kẹp, lý do chưa gắp."""
    rows = [{"Thứ tự": str(b["rank"]) if b["rank"] else "–",
             "Trạng thái": ("Gắp trước" if b["rank"] == 1 else "Gắp được") if b["rank"] else "Chưa gắp",
             "Bao": f"#{b['id']}", "Độ tin cậy": b["conf"],
             "X (mm)": b.get("xyz_cam_mm", [None] * 3)[0], "Y (mm)": b.get("xyz_cam_mm", [None] * 3)[1],
             "Z (mm)": b.get("xyz_cam_mm", [None] * 3)[2], "Góc kẹp (°)": b.get("yaw_cam_deg"),
             "Nghiêng (°)": b.get("tilt_deg"), "Thấp hơn đỉnh (mm)": b.get("below_top_mm"),
             "Lý do chưa gắp": ", ".join(b["reasons"])} for b in bags]
    num = st.column_config.NumberColumn
    st.dataframe(rows, hide_index=True, alt="Bảng các bao: thứ tự gắp, toạ độ, góc kẹp và lý do",
                 column_config={
                     "Thứ tự": st.column_config.TextColumn(width="small"),
                     "Độ tin cậy": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1,
                                                                   help="Độ tin cậy của model khi nhận ra bao."),
                     "X (mm)": num(format="%.0f", help="Sang phải trong hệ camera."),
                     "Y (mm)": num(format="%.0f", help="Xuống dưới trong hệ camera."),
                     "Z (mm)": num(format="%.0f", help="Khoảng cách ra trước camera."),
                     "Góc kẹp (°)": num(format="%.1f", help="Hướng cạnh dài của bao, 0° = dọc trục X camera."),
                     "Nghiêng (°)": num(format="%.1f", help="Quá 35° thì không gắp."),
                     "Thấp hơn đỉnh (mm)": num(format="%.0f", help="Quá 65 mm (nửa lớp bao) coi là lớp dưới."),
                     "Lý do chưa gắp": st.column_config.TextColumn(width="large"),
                 })
