"""App demo: chọn lượt chụp -> nhận diện pallet/bao -> chọn bao cần gắp -> XYZ + góc kẹp.

    .venv/Scripts/streamlit run code/demo/streamlit_app.py

Mỗi ảnh được chạy bằng model của fold không chứa chuỗi chụp đó (code/demo/models.json), nên đây là
kết quả trên cảnh model chưa từng thấy. App chỉ suy luận, không train. Cần ảnh gốc (code/train/download_data.py) và weights
(code/demo/weights.py download).
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
for sub in ("code/obb_labeler", "code/train", "code/grasp"):
    sys.path.insert(0, str(ROOT / sub))
from pick_xyz import height_image, plan_image, render  # noqa: E402

CONFIG = json.loads((ROOT / "code/demo/models.json").read_text(encoding="utf-8"))
CAMERA = json.loads((ROOT / "code/grasp/camera.json").read_text(encoding="utf-8"))
GROUP_NAMES = {
    "september_pallet_1": "Tháng 9 – dỡ hàng dần đến khi pallet rỗng",
    "august_green_and_empty": "Tháng 8 – bao xanh, pallet vơi, bao đè chéo",
    "august_red": "Tháng 8 – bao đỏ, ánh sáng mạnh",
    "september_pallet_2": "Tháng 9 – pallet 2, nhiều pallet cạnh nhau",
    "september_pallet_3": "Tháng 9 – pallet 3, bao trắng/xanh, màng co",
    "september_pallet_4": "Tháng 9 – pallet 4, bao nâu",
}
STATE_NAMES = {"full": "đầy", "partial": "vơi", "few": "còn ít bao", "empty": "rỗng"}

st.set_page_config(page_title="Gắp bao tự động", page_icon=":material/precision_manufacturing:", layout="wide")


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
def load_model(fold):
    from ultralytics import YOLO
    return YOLO(ROOT / CONFIG["folds"][fold]["weights"])


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


# --- Thanh bên: chọn lượt chụp ---
groups = load_images()


def move(key, delta, n):
    st.session_state[key] = min(max(1, st.session_state[key] + delta), n)


with st.sidebar:
    st.header("Chọn lượt chụp")
    group = st.selectbox("Chuỗi chụp", list(GROUP_NAMES), format_func=lambda g: f"{GROUP_NAMES[g]} ({len(groups[g])})",
                         key="group")
    shots = groups[group]
    key = f"shot_{group}"  # mỗi chuỗi nhớ vị trí riêng
    st.session_state.setdefault(key, 1)
    with st.container(horizontal=True):
        st.button("Trước", icon=":material/chevron_left:", on_click=move, args=(key, -1, len(shots)),
                  disabled=st.session_state[key] <= 1)
        st.button("Sau", icon=":material/chevron_right:", on_click=move, args=(key, 1, len(shots)),
                  disabled=st.session_state[key] >= len(shots))
    st.slider("Lượt chụp trong chuỗi (theo thời gian)", 1, len(shots), key=key)
    shot = shots[st.session_state[key] - 1]
    tta = st.toggle("Xoay/lật ảnh 8 hướng", help="Gộp 8 lần dự đoán: box khít hơn, chậm hơn khoảng 8 lần.")
    show_labels = st.toggle("Hiện nhãn tay (trắng)", disabled=shot["labels"] is None,
                            help="Chỉ có ở ảnh đã gán nhãn và kiểm tra.")
    st.divider()
    fold = fold_of(group)
    st.caption(f"Model: {CONFIG['name']}. Ảnh này chạy bằng model `{fold}`, model này chưa từng thấy chuỗi chụp này.")

# --- Nội dung chính ---
st.title("Chọn bao cần gắp từ ảnh màu + depth")
st.caption("Ảnh màu → tìm pallet và bao (box xoay) → đo mặt trên từng bao bằng depth → chọn bao trên cùng "
           "không bị đè → toạ độ XYZ và góc kẹp.")

weights_missing, images_missing = missing_files()
if weights_missing:
    st.error("Chạy `python code/demo/weights.py download` để tải weights (xem code/demo/README.md).",
             title="Chưa có weights", icon=":material/download:")
    st.stop()
if group in images_missing:
    st.error("Chạy `python code/train/download_data.py` để tải ảnh gốc (xem code/demo/README.md).",
             title="Chưa có ảnh gốc", icon=":material/image_not_supported:")
    st.stop()

with st.spinner("Đang xử lý..."):
    res = analyze(shot["id"], shot["image_path"], shot["depth_path"], fold, tta)
plan, objects = res["plan"], res["objects"]
bags = plan["bags"]
ranked = [b for b in bags if b["rank"]]
first = ranked[0] if ranked else None

st.subheader(f"{shot['id']} · {shot['captured_at'][:16].replace('T', ' ')} · pallet {STATE_NAMES.get(shot['state'], shot['state'])}")
with st.container(horizontal=True):
    st.metric("Bao trên pallet mục tiêu", len(bags), border=True, height="stretch")
    st.metric("Gắp được ngay", len(ranked), border=True, height="stretch")
    st.metric("Bao gắp trước", f"#{first['id']}" if first else "–", border=True, height="stretch")
    st.metric("Toạ độ camera (mm)", " · ".join(f"{v:.0f}" for v in first["xyz_cam_mm"]) if first else "–",
              help="X sang phải, Y xuống dưới, Z hướng ra trước camera.", border=True, height="stretch")
    st.metric("Thời gian xử lý", f"{res['detect_ms'] + res['plan_ms']:.0f} ms",
              help=f"Nhận diện {res['detect_ms']:.0f} ms, chọn bao + XYZ {res['plan_ms']:.0f} ms "
                   "(lần đầu chậm hơn do nạp model).", border=True, height="stretch")
if not first:
    st.info(f"Kết luận: {plan['status']}.", icon=":material/info:")

img = cv2.imread(str(ROOT / shot["image_path"]))
if show_labels and shot["labels"]:
    for o in shot["labels"]:
        cv2.polylines(img, [np.int32(o["corners"])], True, (255, 255, 255), 2)
left, right = st.columns(2)
left.image(render(img, objects, plan), channels="BGR", width="stretch",
           caption="Xanh lá: gắp trước · vàng: gắp được · đỏ: chưa gắp được · xanh dương nhạt: pallet mục tiêu",
           alt="Ảnh màu với box pallet và bao, đánh dấu bao cần gắp")
right.image(res["height"], channels="BGR", width="stretch",
            caption="Độ cao từ depth: đỏ = cao nhất, xanh = thấp hơn 30 cm, đen = không đo được",
            alt="Bản đồ độ cao của chồng bao từ ảnh depth")

st.subheader("Các bao trên pallet mục tiêu")
rows = [{"Thứ tự gắp": b["rank"], "Bao": f"#{b['id']}", "Độ tin cậy": b["conf"],
         "X (mm)": b.get("xyz_cam_mm", [None] * 3)[0], "Y (mm)": b.get("xyz_cam_mm", [None] * 3)[1],
         "Z (mm)": b.get("xyz_cam_mm", [None] * 3)[2], "Góc kẹp (°)": b.get("yaw_cam_deg"),
         "Nghiêng (°)": b.get("tilt_deg"), "Thấp hơn đỉnh (mm)": b.get("below_top_mm"),
         "Lý do chưa gắp": ", ".join(b["reasons"])} for b in bags]
st.dataframe(rows, hide_index=True, alt="Bảng các bao: thứ tự gắp, toạ độ, góc kẹp và lý do",
             column_config={"Độ tin cậy": st.column_config.NumberColumn(format="%.2f"),
                            "X (mm)": st.column_config.NumberColumn(format="%.0f"),
                            "Y (mm)": st.column_config.NumberColumn(format="%.0f"),
                            "Z (mm)": st.column_config.NumberColumn(format="%.0f")})
