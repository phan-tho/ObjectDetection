"""Trang camera trực tiếp: xem camera của robot, nhận diện liên tục, bấm chụp để chọn bao và ra toạ độ gắp.

Xem trực tiếp chỉ chạy nhận diện (model nhẹ, ~90 ms/khung CPU) cho nhanh. Bước chọn bao + XYZ chạy khi
bấm "Chụp và chọn bao" (depth = median 5 khung, như lúc thu dữ liệu), hoặc trên mọi khung nếu bật
"Chọn bao liên tục". Đúng quy trình trên robot: gắp xong một bao thì chụp lại.

Camera gắn trên tay robot: nhập tư thế mặt bích lúc chụp (khi tích hợp sẽ đọc thẳng từ bộ điều khiển) và
hiệu chuẩn hand-eye một lần (code/grasp/handeye.py) thì có toạ độ theo hệ robot. Khi đó số # của bao giữ
nguyên giữa các lần chụp theo vị trí trong hệ robot, dù tay robot chụp từ chỗ khác.
"""
import importlib.util
import json
import time

import cv2
import numpy as np
import streamlit as st

from common import (CONFIG, GROUPS, ROOT, bag_table, close_source, keep, legend, load_images, load_weights,
                    open_source, persist)
from handeye import board_pose, make_board
from pick_xyz import height_map, plan_frame, render
from predict import detect
from robot_pose import cam_to_base, parse_pose

LIVE = CONFIG.get("live")
HAS_SDK = importlib.util.find_spec("pyorbbecsdk") is not None
SOURCES = {"orbbec": "Camera robot (Orbbec)", "webcam": "Webcam / camera USB", "replay": "Phát lại dữ liệu đã chụp"}
SOURCE_HELP = {"orbbec": "Ảnh màu + depth: nhận diện, chọn bao và tính XYZ." + ("" if HAS_SDK else " Chưa cài SDK."),
               "webcam": "Chỉ có ảnh màu: chỉ nhận diện, không tính được XYZ.",
               "replay": "Giả lập camera bằng một chuỗi chụp có sẵn, để thử khi không có camera."}
POSE_HINT = {"rpy": "x, y, z, rx, ry, rz (mm, độ)", "kuka_abc": "x, y, z, A, B, C (mm, độ)",
             "rotvec": "x, y, z, rx, ry, rz (mm, rad)", "quat": "x, y, z, qw, qx, qy, qz (mm)",
             "matrix": "16 số của ma trận 4×4, theo hàng"}
TRACK_TOL = 70.0  # mm; một lớp bao dày 130 mm nên bao lộ ra dưới bao vừa gắp không bị nhận nhầm là bao cũ
CALIB_DIR = ROOT / "calib/handeye"
groups = load_images()


def camera_cfg():
    """Đọc lại mỗi lần chạy: handeye.py solve ghi cam_to_flange vào file mà không cần khởi động lại app."""
    return json.loads((ROOT / "code/grasp/camera.json").read_text(encoding="utf-8"))


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
    st.subheader("Tay robot", icon=":material/precision_manufacturing:")
    cfg = camera_cfg()
    fmt = cfg.get("robot_pose_format", "rpy")
    calibrated = cfg.get("cam_to_flange") is not None
    if calibrated:
        st.badge("Đã hiệu chuẩn hand-eye", icon=":material/verified:", color="green")
    else:
        st.badge("Chưa hiệu chuẩn hand-eye", icon=":material/warning:", color="orange",
                 help="Toạ độ đang ở hệ camera. Xem mục Hiệu chuẩn hand-eye cuối trang.")
    persist("live_pose", "")
    pose_text = st.text_input("Tư thế mặt bích lúc chụp", key="live_pose", on_change=keep, args=("live_pose",),
                              placeholder=POSE_HINT[fmt],
                              help=f"Kiểu `{fmt}` ({POSE_HINT[fmt]}), đổi ở robot_pose_format trong "
                                   "code/grasp/camera.json. Đọc trên bộ điều khiển khi tay đứng yên; khi tích hợp, "
                                   "app sẽ lấy tự động.")
    flange = None
    if pose_text.strip():
        try:
            flange = parse_pose(pose_text, fmt)
        except ValueError as e:
            st.caption(f":red[{e}]")
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


def renumber(objects, plan, tracks):
    """Đổi số # của bao theo vị trí trong hệ robot: bao cách một bao đã thấy ở lần chụp trước dưới TRACK_TOL
    giữ số cũ, bao mới lấy số tiếp theo. tracks {số: xyz_robot_mm} được cập nhật tại chỗ."""
    pos = {b["id"]: np.array(b["xyz_robot_mm"]) for b in plan["bags"] if "xyz_robot_mm" in b}
    pairs = sorted((float(np.linalg.norm(p - q)), i, t) for i, p in pos.items() for t, q in tracks.items())
    new, used = {}, set()
    for d, i, t in pairs:
        if d < TRACK_TOL and i not in new and t not in used:
            new[i] = t
            used.add(t)
    nxt = max(tracks, default=0) + 1
    for o in sorted(objects, key=lambda o: (o["id"] not in pos, o["id"])):
        if o["id"] not in new:
            new[o["id"]], nxt = nxt, nxt + 1
    for o in objects:
        o["id"] = new[o["id"]]
    tracks.update({new[i]: p.tolist() for i, p in pos.items()})


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
        plan = plan_frame(depth, src.camera, objects, cam_to_base(cfg, flange)) \
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
            T = cam_to_base(cfg, flange)
            plan = plan_frame(depth, src.camera, objects, T)
            if T is not None:  # có hệ robot: giữ số # theo vị trí rồi chọn lại với số mới
                renumber(objects, plan, st.session_state.setdefault("live_tracks", {}))
                plan = plan_frame(depth, src.camera, objects, T)
            st.session_state["live_capture"] = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": src.label, "plan": plan,
                "flange": None if flange is None or T is None else np.round(flange, 4).tolist(),
                "ms": (time.perf_counter() - t0) * 1000, "vis": render(img, objects, plan),
                "height": height_map(depth, src.camera, plan)}

cap = st.session_state.get("live_capture")
if cap:
    plan = cap["plan"]
    first = next((b for b in plan["bags"] if b["rank"] == 1), None)
    st.caption(f"Lần chụp lúc {cap['time']} · {cap['label']} · xử lý {cap['ms']:.0f} ms")
    if first:
        robot = "xyz_robot_mm" in first
        X, Y, Z = first["xyz_robot_mm"] if robot else first["xyz_cam_mm"]
        st.success(f"Toạ độ {'robot' if robot else 'camera'} X = {X:.0f}, Y = {Y:.0f}, Z = {Z:.0f} mm · góc kẹp "
                   f"{first['yaw_cam_deg']:.1f}° (hệ camera) · mặt bao nghiêng {first['tilt_deg']:.1f}°.",
                   title=f"Gắp bao #{first['id']} trước", icon=":material/my_location:")
    else:
        st.info(f"Kết luận: {plan['status']}.", icon=":material/info:")
    if calibrated and cap["flange"] is None:
        st.warning("Nhập tư thế mặt bích ở thanh bên rồi chụp lại để có toạ độ theo hệ robot.",
                   icon=":material/precision_manufacturing:")
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
    if st.session_state.get("live_tracks"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(f"Số # giữ theo vị trí trong hệ robot ({len(st.session_state['live_tracks'])} bao đã thấy). "
                       "Đổi pallet thì đánh số lại.")
            if st.button("Đánh số lại", icon=":material/restart_alt:", type="tertiary"):
                st.session_state["live_tracks"] = {}

    st.markdown("**:material/precision_manufacturing: Dữ liệu gửi cho robot**")
    keys = ("id", "xyz_cam_mm", "yaw_cam_deg", "normal", "long_axis", "tilt_deg", "xyz_robot_mm", "long_axis_robot")
    payload = {"time": cap["time"], "status": plan["status"], "frame": "robot" if cap["flange"] else "camera",
               "flange_pose": cap["flange"], "pick": {k: first[k] for k in keys if k in first} if first else None}
    st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")
    st.caption("App chưa nối với bộ điều khiển robot: đây là nội dung sẽ gửi khi tích hợp. Có hiệu chuẩn hand-eye "
               "và tư thế mặt bích lúc chụp thì có thêm `xyz_robot_mm` (toạ độ trong hệ gốc robot).")

st.space("medium")
with st.expander("Hiệu chuẩn hand-eye (làm một lần)", icon=":material/tune:"):
    st.markdown(f"""
Tìm vị trí của camera so với mặt bích tay robot, để đổi toạ độ camera sang hệ robot.
1. Tạo bảng: `.venv\\Scripts\\python code/grasp/handeye.py board`, in file `calib/charuco_board.png` khổ A3
   đúng 100%, dán phẳng lên tấm cứng, đặt **cố định** trên bàn hoặc sàn. Đo lại cạnh ô vuông (phải là
   {cfg['charuco']['square_mm']} mm), khác thì sửa `charuco` trong code/grasp/camera.json.
2. Chọn nguồn **Camera robot**, đưa tay tới một tư thế nhìn thấy bảng, dừng tay, nhập tư thế mặt bích ở thanh
   bên, bấm **Lưu ảnh hiệu chuẩn**. Làm 15–25 tư thế, đổi cả vị trí lẫn hướng (nghiêng 15–30° theo nhiều
   trục, xoay quanh trục camera).
3. Chạy `.venv\\Scripts\\python code/grasp/handeye.py solve calib/handeye`. Tốt khi vị trí bảng lệch nhau
   vài mm giữa các lần chụp. Kết quả tự ghi vào camera.json, app dùng ngay.
""")
    saved = sorted(p for p in CALIB_DIR.glob("*") if (p / "color.png").is_file()) if CALIB_DIR.is_dir() else []
    ready = kind == "orbbec" and flange is not None
    if st.button("Lưu ảnh hiệu chuẩn", icon=":material/add_a_photo:", disabled=not ready,
                 help=None if ready else "Cần nguồn Camera robot và tư thế mặt bích ở thanh bên."):
        src = get_source()
        if src is not None:
            img, _ = src.read()
            found = board_pose(img, make_board(cfg), src.camera.A, src.camera.D)
            if found is None:
                st.warning("Không thấy đủ bảng ChArUco trong ảnh, chưa lưu. Đưa camera lại gần hoặc chỉnh góc.",
                           icon=":material/grid_off:")
            else:
                out = CALIB_DIR / f"{len(saved) + 1:03d}"
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / "color.png"), img)
                (out / "pose.json").write_text(json.dumps({"pose": flange.ravel().tolist(), "format": "matrix",
                                                           "entered": pose_text, "entered_format": fmt}), "utf-8")
                (out / "intrinsics.json").write_text(json.dumps(
                    {"A": src.camera.A.tolist(), "D": None if src.camera.D is None else src.camera.D.tolist()}),
                    "utf-8")
                saved.append(out)
                st.success(f"Đã lưu {out.relative_to(ROOT)}: thấy {found[1]} góc, cách bảng "
                           f"{np.linalg.norm(found[0][:3, 3]):.0f} mm.", icon=":material/check:")
    st.caption(f"Đã lưu {len(saved)} lần chụp trong `calib/handeye/` (nên 15–25).")
