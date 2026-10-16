"""Trang giới thiệu: hệ thống làm gì, chuỗi chụp là gì, các lựa chọn và cách đọc kết quả."""
import streamlit as st

from common import CONFIG, GROUPS, STATE_NAMES, load_images, open_group

DEMO = "app_pages/demo.py"
groups = load_images()
n_shots = sum(len(v) for v in groups.values())

st.title("Gắp bao tự động từ camera", icon=":material/precision_manufacturing:")
st.markdown(
    "Camera đặt phía trên pallet chụp một **ảnh màu** và một **ảnh depth** (khoảng cách từ camera tới từng "
    "điểm). Từ hai ảnh đó, hệ thống tìm các bao, chọn bao nên gắp trước và tính **toạ độ XYZ** cùng **góc "
    "kẹp** để gửi cho robot. Trang này giải thích các lựa chọn trước khi bạn vào demo.")
if st.button("Vào demo", type="primary", icon=":material/arrow_forward:"):
    st.switch_page(DEMO)

st.space("medium")
st.header("Hệ thống làm gì", icon=":material/account_tree:")
steps = [
    (":material/center_focus_strong:", "1. Nhận diện",
     "Model YOLO tìm pallet và từng bao trên ảnh màu, mỗi vật là một box xoay theo hướng bao."),
    (":material/straighten:", "2. Đo mặt trên",
     "Đổi depth trong box thành điểm 3D, khớp một mặt phẳng để biết bao cao bao nhiêu, nghiêng thế nào."),
    (":material/checklist:", "3. Chọn bao",
     "Chỉ giữ bao ở lớp trên cùng, không bị bao khác đè, không quá nghiêng. Bao cao nhất gắp trước."),
    (":material/my_location:", "4. Xuất toạ độ",
     "Tâm mặt trên bao (X, Y, Z, mm) và góc kẹp theo cạnh dài của bao."),
]
with st.container(horizontal=True):
    for icon, title, text in steps:
        with st.container(border=True, height="stretch"):
            st.markdown(f"#### {icon} {title}")
            st.caption(text)

st.space("medium")
st.header("Chuỗi chụp là gì?", icon=":material/burst_mode:")
st.markdown(
    f"""
- **Một lượt chụp** = một lần camera chụp: một ảnh màu + một ảnh depth của pallet tại một thời điểm.
- **Một chuỗi chụp** = các lượt chụp liên tiếp của *cùng một pallet trong cùng một buổi*, xếp theo thời
  gian. Ví dụ chuỗi “dỡ hàng dần” chụp lại sau mỗi lần lấy bớt bao, nên đi từ lượt đầu đến lượt cuối là
  thấy pallet vơi dần đến rỗng.
- Demo có **{len(groups)} chuỗi, {n_shots} lượt chụp**, mỗi chuỗi là một tình huống khác nhau (màu bao,
  ánh sáng, bao đè, nhiều pallet). Trên trang demo, bạn **chọn chuỗi** trước, rồi **chọn lượt** trong chuỗi.
- Mỗi chuỗi được chạy bằng một model **chưa từng thấy chuỗi đó** lúc train, nên kết quả là kết quả thật
  trên cảnh mới, không phải ảnh model đã học thuộc.
""")

cols = st.columns(3)
for i, (g, info) in enumerate(GROUPS.items()):
    shots = groups.get(g, [])
    states = sorted({s["state"] for s in shots}, key=list(STATE_NAMES).index)
    with cols[i % 3].container(border=True, height="stretch"):
        st.markdown(f"**{info['icon']} {info['name']}**")
        with st.container(horizontal=True, gap="small"):
            st.badge(f"{len(shots)} lượt", icon=":material/photo_library:", color="blue")
            if shots:
                st.badge(shots[0]["captured_at"][:10], icon=":material/calendar_today:", color="gray")
            st.badge("pallet " + ", ".join(STATE_NAMES[s] for s in states), color="violet")
        st.markdown(info["about"])
        st.caption(f":material/visibility: {info['look']}")
        if st.button("Xem chuỗi này", key=f"go_{g}", icon=":material/play_arrow:"):
            open_group(g)
            st.switch_page(DEMO)

st.space("medium")
st.header("Các lựa chọn trên trang demo", icon=":material/tune:")
st.markdown("""
| Lựa chọn (thanh bên trái) | Tác dụng | Khi nào dùng |
|---|---|---|
| :material/burst_mode: **Chuỗi chụp** | Chọn tình huống (pallet, buổi chụp) muốn xem. | Đầu tiên. Xem gợi ý ở các thẻ phía trên. |
| :material/chevron_left: **Trước** / **Sau** :material/chevron_right: | Lùi / tiến một lượt chụp trong chuỗi. | Đi theo thời gian, ví dụ xem pallet được dỡ dần. |
| :material/linear_scale: **Thanh trượt lượt chụp** | Nhảy thẳng tới lượt bất kỳ (1 = lượt chụp sớm nhất). | Khi chuỗi dài, muốn tới đoạn cuối nhanh. |
| :material/rotate_90_degrees_cw: **Xoay/lật ảnh 8 hướng** | Chạy model trên 8 bản xoay/lật của ảnh rồi gộp lại: box khít hơn. | Khi muốn kết quả tốt nhất. Chậm hơn khoảng 8 lần. |
| :material/edit: **Hiện nhãn tay** | Vẽ thêm box trắng do người khoanh để so với box của model. | Chỉ bật được ở ảnh đã gán nhãn và kiểm tra. |
""")

st.space("medium")
st.header("Đọc kết quả", icon=":material/insights:")
left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.markdown("**Ảnh màu**")
    with st.container(horizontal=True, gap="small"):
        st.badge("Gắp trước", icon=":material/looks_one:", color="green")
        st.badge("Gắp được", icon=":material/check:", color="yellow")
        st.badge("Chưa gắp được", icon=":material/close:", color="red")
        st.badge("Pallet mục tiêu", icon=":material/crop_free:", color="blue")
    st.caption("Bao gắp trước có ghi toạ độ (X, Y, Z) mm và góc kẹp ngay trên ảnh. Chỉ xét pallet gần giữa "
               "ảnh nhất, bao của pallet khác bị bỏ qua.")
    st.markdown("**Bản đồ độ cao** (từ depth)")
    st.caption("Đỏ sẫm = cao nhất, chuyển dần sang xanh = thấp hơn tới 30 cm, đen = camera không đo được. "
               "Giúp thấy bao nào nằm trên, bao nào bị đè.")
with right.container(border=True, height="stretch"):
    st.markdown("**Bảng các bao**")
    st.markdown("""
- **X, Y, Z (mm)**: toạ độ điểm gắp trong hệ camera. X sang phải, Y xuống dưới, Z là khoảng cách ra trước camera.
- **Góc kẹp (°)**: hướng cạnh dài của bao, 0° là dọc trục X camera (nằm ngang trên ảnh).
- **Nghiêng (°)**: góc giữa mặt trên bao và mặt vuông góc với hướng nhìn của camera. Quá 35° thì không gắp.
- **Thấp hơn đỉnh (mm)**: thấp hơn bao cao nhất bao nhiêu. Quá 65 mm (nửa lớp bao) coi là lớp dưới.
- **Lý do chưa gắp**: lớp dưới, bị bao khác đè, quá nghiêng, bị che nhiều, thiếu depth...
""")

st.space("medium")
st.header("Dùng với camera robot", icon=":material/videocam:")
st.markdown("""
Trang **Camera trực tiếp** chạy giống lúc dùng thật trên robot:
1. **Xem trực tiếp**: lấy khung hình liên tục từ camera, model nhẹ (YOLO26n, ~90 ms/khung trên CPU) đánh
   dấu pallet và bao trên từng khung.
2. **Chụp và chọn bao**: khi robot sẵn sàng, lấy 5 khung, gộp depth cho bớt nhiễu, chọn bao và ra toạ độ
   gắp kèm đoạn JSON sẽ gửi cho robot. Gắp xong một bao thì chụp lại.

Nguồn ảnh: **camera Orbbec** của robot (ảnh màu + depth, cần cài SDK), **webcam** (chỉ ảnh màu, chỉ nhận
diện), hoặc **phát lại** một chuỗi chụp có sẵn như camera đang chạy, để thử khi không có camera.
""")
if st.button("Mở camera trực tiếp", icon=":material/videocam:"):
    st.switch_page("app_pages/live.py")

st.space("medium")
st.info(
    f"Model: {CONFIG['name']}. Toạ độ hiện ở hệ camera; muốn ra hệ robot cần hiệu chuẩn hand-eye. "
    "Độ chính xác đo trên ảnh của chính hiện trường này, chưa có tập test độc lập.",
    title="Cần biết khi trình bày", icon=":material/info:")
if st.button("Vào demo", type="primary", icon=":material/arrow_forward:", key="go_bottom"):
    st.switch_page(DEMO)
