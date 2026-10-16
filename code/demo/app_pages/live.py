"""Trang camera trực tiếp: xem camera của robot, nhận diện liên tục, bấm chụp để chọn bao và ra toạ độ gắp.

Xem trực tiếp chỉ chạy nhận diện (model nhẹ, ~90 ms/khung CPU) cho nhanh. Bước chọn bao + XYZ chạy khi
bấm "Chụp và chọn bao" (depth = median 5 khung, như lúc thu dữ liệu), hoặc trên mọi khung nếu bật
"Chọn bao liên tục". Đúng quy trình trên robot: gắp xong một bao thì chụp lại.
"""
import importlib.util
import json
import time

import cv2
import numpy as np
import streamlit as st

from common import (CAMERA, CONFIG, GROUPS, ROOT, bag_table, close_source, keep, legend, load_images, load_weights,
                    open_source, persist)
from pick_xyz import height_map, plan_frame, render
from predict import detect

LIVE = CONFIG.get("live")
HAS_SDK = importlib.util.find_spec("pyorbbecsdk") is not None
SOURCES = {"orbbec": "Camera robot (Orbbec)", "webcam": "Webcam / camera USB", "replay": "Phát lại dữ liệu đã chụp"}
SOURCE_HELP = {"orbbec": "Ảnh màu + depth: nhận diện, chọn bao và tính XYZ." + ("" if HAS_SDK else " Chưa cài SDK."),
               "webcam": "Chỉ có ảnh màu: chỉ nhận diện, không tính được XYZ.",
               "replay": "Giả lập camera bằng một chuỗi chụp có sẵn, để thử khi không có camera."}
groups = load_images()


def on_toggle():
    keep("live_on")
    if not st.session_state["live_on"]:
        close_source()  # trả camera cho chương trình khác


# --- Thanh bên: nguồn ảnh và chế độ chạy ---
with st.sidebar:
    st.header("Nguồn ảnh", icon=":material/videocam:")
    persist("live_kind", "orbbec" if HAS_SDK else "replay")
    kind = st.radio("Lấy ảnh từ", list(SOURCES), key="live_kind", on_change=keep, args=("live_kind",),
                    format_func=SOURCES.get, captions=[SOURCE_HELP[k] for k in SOURCES])
    opts = {}
    if kind == "webcam":
        persist("live_index", 0)
        opts["index"] = st.number_input("Số thứ tự webcam", 0, 9, key="live_index", on_change=keep,
                                        args=("live_index",), help="0 = webcam mặc định của máy.")
    elif kind == "replay":
        persist("live_group", next(iter(GROUPS)))
        opts["group"] = st.selectbox("Chuỗi chụp", [g for g in GROUPS if g in groups], key="live_group",
                                     on_change=keep, args=("live_group",),
                                     format_func=lambda g: f"{GROUPS[g]['name']} ({len(groups[g])})")
        persist("live_period", 2)
        opts["period"] = st.select_slider("Mỗi lượt chụp hiện trong (giây)", [1, 2, 3, 5], key="live_period",
                                          on_change=keep, args=("live_period",))
    has_depth = kind != "webcam"

    st.subheader("Chạy", icon=":material/play_circle:")
    persist("live_on", False)
    running = st.toggle("Bật camera", key="live_on", on_change=on_toggle)
    persist("live_fps", 5)
    fps = st.select_slider("Làm mới tối đa (khung/giây)", [1, 2, 5, 10], key="live_fps", on_change=keep,
                           args=("live_fps",), help="Thực tế còn tuỳ tốc độ máy, xem số khung/giây đo được.")
    persist("live_plan", False)
    continuous = st.toggle("Chọn bao liên tục", key="live_plan", on_change=keep, args=("live_plan",),
                           disabled=not has_depth,
                           help="Chạy cả bước chọn bao + XYZ trên mọi khung (thêm ~150 ms/khung). Tắt thì chỉ "
                                "nhận diện, bấm “Chụp và chọn bao” khi cần.")
    st.space("small")
    if LIVE:
        st.caption(f"Model: {LIVE['name']}.")
    st.page_link("app_pages/intro.py", label="Xem hướng dẫn", icon=":material/menu_book:")

# --- Nội dung chính ---
st.title("Camera trực tiếp", icon=":material/videocam:")
st.caption("Xem camera liên tục, model đánh dấu pallet và bao trên từng khung. Khi robot sẵn sàng gắp, bấm "
           "**Chụp và chọn bao** để lấy bao cần gắp và toạ độ. Gắp xong một bao thì chụp lại.")

if not LIVE or not (ROOT / LIVE["weights"]).is_file():
    st.error("Chạy `python code/demo/weights.py download` để tải model nhẹ cho camera trực tiếp "
             f"(`{LIVE['weights'] if LIVE else 'live'}`).", title="Chưa có model", icon=":material/download:")
    st.stop()
if kind == "orbbec" and not HAS_SDK:
    st.warning("Cài trên máy cắm camera: `.venv\\Scripts\\python -m pip install pyorbbecsdk2`, rồi khởi động "
               "lại app. Cắm camera vào cổng USB 3. Trong lúc chờ, chọn “Phát lại dữ liệu đã chụp” để thử.",
               title="Chưa cài SDK của camera Orbbec", icon=":material/extension_off:")
    st.stop()


def get_source():
    try:
        return open_source(kind, **opts)
    except Exception as e:  # SDK camera ném lỗi riêng (OBError), không chỉ RuntimeError
        st.error(f"{e}", title="Không mở được nguồn ảnh", icon=":material/videocam_off:")
        return None


def find(img):
    objects = detect([load_weights(LIVE["weights"])], img, LIVE["imgsz"])
    for n, o in enumerate(objects, 1):
        o["id"] = n
    return objects


def draw(img, objects, scale=0.5):
    """Khung xem trực tiếp khi chưa chọn bao: pallet xanh dương nhạt, bao vàng nhạt kèm độ tin cậy."""
    img = img.copy()
    for o in objects:
        color = (255, 255, 0) if o["class_id"] == 0 else (180, 230, 255)
        cv2.polylines(img, [np.int32(o["corners"])], True, color, 3)
        if o["class_id"] == 1:
            cv2.putText(img, f"{o['conf']:.2f}", (int(o["cx"]) - 30, int(o["cy"])), 0, 1.1, color, 3)
    return cv2.resize(img, None, fx=scale, fy=scale)


@st.fragment(run_every=1 / fps if running else None)
def live_view():
    if not running:
        with st.container(border=True, height=320, horizontal_alignment="center", vertical_alignment="center"):
            st.markdown(":material/videocam_off: Camera đang tắt", text_alignment="center")
            st.caption("Bật **Bật camera** ở thanh bên để xem trực tiếp.", text_alignment="center")
        return
    src = get_source()
    if src is None:
        return
    try:
        t0 = time.perf_counter()
        img, depth = src.read()
        t1 = time.perf_counter()
        objects = find(img)
        t2 = time.perf_counter()
        plan = plan_frame(depth, src.camera, objects, CAMERA.get("cam_to_robot")) \
            if continuous and depth is not None else None
        t3 = time.perf_counter()
    except Exception as e:
        st.error(f"{e}", title="Lỗi khi đọc camera", icon=":material/error:")
        return
    ticks = st.session_state.setdefault("_live_ticks", [])
    ticks.append(time.monotonic())
    ticks[:] = ticks[-15:]
    real_fps = (len(ticks) - 1) / (ticks[-1] - ticks[0]) if len(ticks) > 1 and ticks[-1] > ticks[0] else 0.0
    bags = [o for o in objects if o["class_id"] == 1]
    first = next((b for b in plan["bags"] if b["rank"] == 1), None) if plan else None
    with st.container(horizontal=True):
        st.metric("Khung/giây", f"{real_fps:.1f}", icon=":material/speed:", border=True, height="stretch")
        st.metric("Nhận diện", f"{(t2 - t1) * 1000:.0f} ms", icon=":material/timer:", border=True, height="stretch",
                  help=f"Đọc ảnh {(t1 - t0) * 1000:.0f} ms" + (f", chọn bao {(t3 - t2) * 1000:.0f} ms" if plan else ""))
        st.metric("Bao thấy được", len(bags), icon=":material/inventory_2:", border=True, height="stretch")
        st.metric("Bao gắp trước", f"#{first['id']}" if first else "–", icon=":material/my_location:", border=True,
                  height="stretch", help="Chỉ có khi bật “Chọn bao liên tục”.")
    with st.container(border=True):
        st.image(render(img, objects, plan) if plan else draw(img, objects), channels="BGR", width="stretch",
                 alt="Khung hình camera trực tiếp với box pallet và bao")
        with st.container(horizontal=True, vertical_alignment="center"):
            st.badge("Trực tiếp", icon=":material/fiber_manual_record:", color="red")
            st.caption(src.label)


live_view()

st.subheader("Chụp và chọn bao", icon=":material/my_location:")
st.caption("Lấy 5 khung liên tiếp, depth gộp bằng median để bớt nhiễu, rồi chọn bao trên cùng không bị đè và "
           "tính toạ độ gắp." + ("" if has_depth else " Webcam không có depth nên không dùng được bước này."))
if st.button("Chụp và chọn bao", type="primary", icon=":material/photo_camera:", disabled=not has_depth):
    src = get_source()
    if src is not None:
        with st.spinner("Đang chụp và chọn bao..."):
            t0 = time.perf_counter()
            img, depth = src.capture(5)
            objects = find(img)
            plan = plan_frame(depth, src.camera, objects, CAMERA.get("cam_to_robot"))
            st.session_state["live_capture"] = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": src.label, "plan": plan,
                "ms": (time.perf_counter() - t0) * 1000, "vis": render(img, objects, plan),
                "height": height_map(depth, src.camera, plan)}

cap = st.session_state.get("live_capture")
if cap:
    plan = cap["plan"]
    first = next((b for b in plan["bags"] if b["rank"] == 1), None)
    st.caption(f"Lần chụp lúc {cap['time']} · {cap['label']} · xử lý {cap['ms']:.0f} ms")
    if first:
        X, Y, Z = first["xyz_cam_mm"]
        st.success(f"Toạ độ camera X = {X:.0f}, Y = {Y:.0f}, Z = {Z:.0f} mm · góc kẹp {first['yaw_cam_deg']:.1f}° · "
                   f"mặt bao nghiêng {first['tilt_deg']:.1f}°.",
                   title=f"Gắp bao #{first['id']} trước", icon=":material/my_location:")
    else:
        st.info(f"Kết luận: {plan['status']}.", icon=":material/info:")
    left, right = st.columns(2)
    with left.container(border=True, height="stretch"):
        st.markdown("**:material/image: Ảnh lúc chụp**")
        st.image(cap["vis"], channels="BGR", width="stretch", alt="Ảnh lúc chụp, đánh dấu bao cần gắp")
        legend()
    with right.container(border=True, height="stretch"):
        st.markdown("**:material/terrain: Bản đồ độ cao từ depth**")
        st.image(cap["height"], channels="BGR", width="stretch", alt="Bản đồ độ cao lúc chụp")
        st.caption("Đỏ sẫm = cao nhất · xanh = thấp hơn 30 cm · đen = không đo được")
    if plan["bags"]:
        bag_table(plan["bags"])

    st.markdown("**:material/precision_manufacturing: Dữ liệu gửi cho robot**")
    keys = ("id", "xyz_cam_mm", "yaw_cam_deg", "normal", "long_axis", "tilt_deg", "xyz_robot_mm", "long_axis_robot")
    payload = {"time": cap["time"], "status": plan["status"],
               "pick": {k: first[k] for k in keys if k in first} if first else None}
    st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
    st.caption("App chưa nối với bộ điều khiển robot: đây là nội dung sẽ gửi khi tích hợp. Toạ độ đang ở hệ "
               "camera; điền `cam_to_robot` trong code/grasp/camera.json (sau hiệu chuẩn hand-eye) thì có thêm "
               "`xyz_robot_mm`.")
