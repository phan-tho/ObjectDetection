"""App demo: chọn lượt chụp -> nhận diện pallet/bao -> chọn bao cần gắp -> XYZ + góc kẹp.

    .venv/Scripts/streamlit run code/demo/streamlit_app.py

Hai trang: giới thiệu (hướng dẫn các lựa chọn, mở đầu tiên) và demo. Mỗi ảnh được chạy bằng model của
fold không chứa chuỗi chụp đó (code/demo/models.json), nên đây là kết quả trên cảnh model chưa từng
thấy. App chỉ suy luận, không train. Cần ảnh gốc (code/train/download_data.py) và weights
(code/demo/weights.py download).
"""
import streamlit as st

st.set_page_config(page_title="Gắp bao tự động", page_icon=":material/precision_manufacturing:", layout="wide")

page = st.navigation([
    st.Page("app_pages/intro.py", title="Giới thiệu", icon=":material/menu_book:", default=True),
    st.Page("app_pages/demo.py", title="Chạy demo", icon=":material/precision_manufacturing:", url_path="demo"),
    st.Page("app_pages/live.py", title="Camera trực tiếp", icon=":material/videocam:", url_path="live"),
], position="top")
page.run()
