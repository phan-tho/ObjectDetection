"""Trang demo: chọn chuỗi + lượt chụp -> box pallet/bao, bao gắp trước, XYZ, bản đồ độ cao, bảng các bao."""
import cv2
import numpy as np
import streamlit as st

from common import (CONFIG, GROUPS, ROOT, STATE_NAMES, analyze, bag_table, fold_of, keep, legend,
                    load_images, missing_files)
from pick_xyz import render

groups = load_images()
order = [g for g in GROUPS if g in groups]


def move(key, delta, n):
    st.session_state[key] = st.session_state["_" + key] = min(max(1, st.session_state[key] + delta), n)


# --- Thanh bên: chọn lượt chụp ---
with st.sidebar:
    st.header("Chọn cảnh", icon=":material/photo_camera:")
    st.session_state["group"] = st.session_state.get("_group", order[0])
    group = st.selectbox("Chuỗi chụp", order, key="group", on_change=keep, args=("group",),
                         format_func=lambda g: f"{GROUPS[g]['name']} ({len(groups[g])})",
                         help="Các lượt chụp liên tiếp của cùng một pallet trong một buổi, xếp theo thời gian. "
                              "Mỗi chuỗi là một tình huống khác nhau, xem trang Giới thiệu.")
    st.caption(GROUPS[group]["about"])
    shots = groups[group]
    n = len(shots)
    key = f"shot_{group}"  # mỗi chuỗi nhớ vị trí riêng
    st.session_state[key] = min(st.session_state.get("_" + key, 1), n)
    if n > 1:
        st.slider("Lượt chụp trong chuỗi", 1, n, key=key, on_change=keep, args=(key,),
                  help="1 = lượt chụp sớm nhất. Kéo để nhảy tới lượt bất kỳ.")
    with st.container(horizontal=True):
        st.button("Trước", icon=":material/chevron_left:", on_click=move, args=(key, -1, n),
                  disabled=st.session_state[key] <= 1, width="stretch")
        st.button("Sau", icon=":material/chevron_right:", icon_position="right", on_click=move,
                  args=(key, 1, n), disabled=st.session_state[key] >= n, width="stretch")
    shot = shots[st.session_state[key] - 1]

    st.subheader("Tuỳ chọn", icon=":material/tune:")
    tta = st.toggle("Xoay/lật ảnh 8 hướng", help="Gộp 8 lần dự đoán: box khít hơn, chậm hơn khoảng 8 lần.")
    show_labels = st.toggle("Hiện nhãn tay (trắng)", disabled=shot["labels"] is None,
                            help="Box do người khoanh, để so với box của model. Chỉ có ở ảnh đã gán nhãn và kiểm tra.")
    st.space("small")
    fold = fold_of(group)
    st.caption(f"Model: {CONFIG['name']}. Chuỗi này chạy bằng model `{fold}`, model này chưa từng thấy chuỗi "
               "chụp này lúc train.")
    st.page_link("app_pages/intro.py", label="Xem hướng dẫn", icon=":material/menu_book:")

# --- Nội dung chính ---
st.title("Chọn bao cần gắp", icon=":material/precision_manufacturing:")
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

with st.container(horizontal=True, vertical_alignment="center"):
    st.subheader(f"Lượt {st.session_state[key]}/{n} · {shot['captured_at'][:16].replace('T', ' ')}",
                 width="content")
    state = shot["state"]
    st.badge(f"pallet {STATE_NAMES.get(state, state)}", icon=":material/pallet:",
             color={"full": "blue", "partial": "orange", "few": "orange", "empty": "gray"}.get(state, "gray"))
    if shot["labels"] is not None:
        st.badge("có nhãn tay", icon=":material/edit:", color="violet")
    if tta:
        st.badge("TTA 8 hướng", icon=":material/rotate_90_degrees_cw:", color="violet")
    st.caption(shot["id"])

with st.spinner("Đang nhận diện và chọn bao..."):
    res = analyze(shot["id"], shot["image_path"], shot["depth_path"], fold, tta)
plan, objects = res["plan"], res["objects"]
bags = plan["bags"]
ranked = [b for b in bags if b["rank"]]
first = ranked[0] if ranked else None

with st.container(horizontal=True):
    st.metric("Bao trên pallet mục tiêu", len(bags), icon=":material/inventory_2:", border=True, height="stretch")
    st.metric("Gắp được ngay", len(ranked), icon=":material/task_alt:", border=True, height="stretch")
    st.metric("Chưa gắp được", len(bags) - len(ranked), icon=":material/block:", border=True, height="stretch")
    st.metric("Thời gian xử lý", f"{res['detect_ms'] + res['plan_ms']:.0f} ms", icon=":material/timer:",
              help=f"Nhận diện {res['detect_ms']:.0f} ms, chọn bao + XYZ {res['plan_ms']:.0f} ms "
                   "(lần đầu chậm hơn do nạp model).", border=True, height="stretch")

if first:
    X, Y, Z = first["xyz_cam_mm"]
    st.success(f"Toạ độ camera X = {X:.0f}, Y = {Y:.0f}, Z = {Z:.0f} mm · góc kẹp {first['yaw_cam_deg']:.1f}° · "
               f"mặt bao nghiêng {first['tilt_deg']:.1f}°.",
               title=f"Gắp bao #{first['id']} trước", icon=":material/my_location:")
else:
    st.info(f"Kết luận: {plan['status']}.", icon=":material/info:")

img = cv2.imread(str(ROOT / shot["image_path"]))
if show_labels and shot["labels"]:
    for o in shot["labels"]:
        cv2.polylines(img, [np.int32(o["corners"])], True, (255, 255, 255), 2)
left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.markdown("**:material/image: Ảnh màu và kết quả nhận diện**")
    st.image(render(img, objects, plan), channels="BGR", width="stretch",
             alt="Ảnh màu với box pallet và bao, đánh dấu bao cần gắp")
    legend(show_labels and bool(shot["labels"]))
with right.container(border=True, height="stretch"):
    st.markdown("**:material/terrain: Bản đồ độ cao từ depth**")
    st.image(res["height"], channels="BGR", width="stretch", alt="Bản đồ độ cao của chồng bao từ ảnh depth")
    st.caption("Đỏ sẫm = cao nhất · xanh = thấp hơn 30 cm · đen = không đo được")

st.subheader("Các bao trên pallet mục tiêu", icon=":material/table_rows:")
bag_table(bags)
