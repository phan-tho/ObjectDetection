# App demo: chọn bao cần gắp

**Demo chỉ chạy suy luận, không train.** Chỉ cần tải ảnh và weights đã train sẵn, rồi mở app.

Chọn một lượt chụp, app hiện: box pallet và bao, bao gắp trước cùng toạ độ XYZ và góc kẹp, bản đồ độ
cao từ depth, bảng các bao kèm lý do chưa gắp được.

Weights gồm 4 model, mỗi model đã được train trước (trên Kaggle) mà không dùng một nhóm chuỗi chụp,
và một model nhẹ `live.pt` (train trên mọi ảnh có nhãn) cho trang camera trực tiếp.
App chạy mỗi ảnh bằng model chưa từng thấy chuỗi chụp đó (`models.json`), nên kết quả trên demo là
kết quả thật trên cảnh mới, không phải ảnh model đã học thuộc.

## Cài đặt (làm một lần)

Cần Python 3.11 và khoảng 3 GB ổ đĩa (ảnh 2.4 GB, weights 86 MB). Chạy các lệnh từ thư mục gốc của repo.

**1. Môi trường Python**

Windows:
```sh
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -U pip
```
macOS / Linux:
```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -U pip
```

Có GPU NVIDIA thì cài torch bản CUDA trước, không có GPU thì bỏ qua (app vẫn chạy trên CPU, khoảng
1 giây mỗi ảnh):
```sh
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Sau đó:
```sh
.venv\Scripts\python -m pip install -r requirements.txt
```
(macOS/Linux: thay `.venv\Scripts\python` bằng `.venv/bin/python` ở mọi lệnh.)

**2. Token Kaggle**

Ảnh và weights đều để trên Kaggle. Vào kaggle.com > Settings > API > **Create New Token**, đặt file
`kaggle.json` vào `C:\Users\<tên>\.kaggle\` (Windows) hoặc `~/.kaggle/` (macOS/Linux).

Weights nằm ở dataset private `tuananhpham29/bao-tai-obb-weights`. Gửi username Kaggle của bạn cho chủ
dataset để được thêm vào (trang dataset > Settings > Sharing).

**3. Tải ảnh và weights** (script tải ảnh nằm trong `code/train/` nhưng chỉ tải, không train)
```sh
.venv\Scripts\python code/train/download_data.py
.venv\Scripts\python code/demo/weights.py download
```
Ảnh về `bao-tai/` và `2026-09-11/`, weights về `weights/demo/`. Cả hai không vào git.

## Chạy
```sh
.venv\Scripts\streamlit run code/demo/streamlit_app.py
```
Trình duyệt mở http://localhost:8501, vào trang **Giới thiệu** trước: giải thích hệ thống, chuỗi chụp
là gì, các lựa chọn và cách đọc kết quả; mỗi thẻ chuỗi chụp có nút mở thẳng chuỗi đó trong trang
**Chạy demo**. Lần đầu mỗi chuỗi chụp chậm hơn vài giây do nạp model, các lần sau được cache.
Theme (màu, font) ở `code/demo/.streamlit/config.toml`.

## Camera trực tiếp

Trang **Camera trực tiếp** chạy như trên robot: xem camera liên tục (chỉ nhận diện, model nhẹ
YOLO26n `weights/demo/live.pt`), bấm **Chụp và chọn bao** để lấy 5 khung, gộp depth bằng median, chọn
bao và ra toạ độ gắp + JSON sẽ gửi cho robot. Bật **Chọn bao liên tục** để chọn bao trên mọi khung
(chậm hơn ~150 ms/khung).

Nguồn ảnh (chọn ở thanh bên):
- **Camera robot (Orbbec Gemini 2)**: ảnh màu 1920×1080 + depth khớp ảnh màu, giống lúc thu dữ liệu.
  Cần SDK, chỉ cài trên máy cắm camera (không có trong `requirements.txt`):
  ```sh
  .venv\Scripts\python -m pip install pyorbbecsdk2
  ```
  Cắm cổng USB 3, khởi động lại app. Tắt **Bật camera** để trả camera cho chương trình khác.
- **Webcam / camera USB**: chỉ ảnh màu, chỉ nhận diện, không tính XYZ.
- **Phát lại dữ liệu đã chụp**: phát một chuỗi chụp như camera đang chạy, để thử khi không có camera.

Code camera ở `code/demo/camera.py`. Phần Orbbec chưa được chạy với camera thật (viết theo API
pyorbbecsdk v2); lần đầu cắm camera nên kiểm tra ảnh màu, depth và toạ độ có hợp lý không.

## Gợi ý khi demo

- **"Tháng 9 – dỡ hàng dần đến khi pallet rỗng"**: bấm **Sau** qua từng lượt. Mỗi lượt hệ thống chỉ ra
  bao gắp tiếp theo, đến khi báo pallet rỗng. Đây là phần dễ thấy nhất.
- **"Tháng 8 – bao xanh, pallet vơi, bao đè chéo"**: bao nằm chéo đè lên bao khác. Xem cột "Lý do chưa
  gắp" và bản đồ độ cao.
- **"Tháng 9 – pallet 2 / 3"**: nhiều pallet cạnh nhau, chỉ xét pallet mục tiêu ở giữa ảnh.
- Bật **Hiện nhãn tay** ở ảnh đã gán nhãn để so box của model với box người khoanh.
- Bật **Xoay/lật ảnh 8 hướng** để box khít hơn, nhưng chậm hơn khoảng 8 lần.

Khi trình bày cần nói rõ:
- Toạ độ đang ở hệ camera. Muốn ra hệ robot cần hiệu chuẩn hand-eye (`code/grasp/README.md`).
- Độ chính xác (mAP50-95 0.864) đo trên 25 ảnh của chính hiện trường này, chưa có tập test độc lập.

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| App báo "Chưa có weights" | Chạy `code/demo/weights.py download` |
| `weights.py download` báo 403 hoặc "Tải thất bại" | Tài khoản Kaggle chưa được share dataset weights, hoặc token chưa đặt đúng chỗ |
| App báo "Chưa có ảnh gốc" | Chạy `code/train/download_data.py` |
| `ModuleNotFoundError` | Chưa cài `requirements.txt` vào đúng `.venv`, hoặc đang chạy `streamlit` ngoài `.venv` |

Weights mới (sau khi train lại) do người train đẩy lên; cách làm ở `code/train/README.md`, mục
"Đưa weights lên cho demo". Khi `models.json` đổi, chạy lại `weights.py download`.
