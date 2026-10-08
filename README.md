# Nhận diện bao để robot gắp

**Mục tiêu:** RGB → tìm pallet mục tiêu gần giữa ảnh → tìm bao → dùng depth/rule chọn bao trên cùng có thể gắp → hiệu chuẩn tọa độ sang XYZ robot. Camera cần thấy toàn bộ pallet mục tiêu. OBB chưa phải nhãn điểm gắp/XYZ.

**Hiện tại:** đã phân loại 123 lượt chụp (105 dùng được, 18 loại), đã đánh nhãn và xác nhận 25 ảnh RGB đại diện: **49 box pallet, 254 box bao**. Chưa fine-tune, augment hoặc chia train/val/test.

## Dữ liệu đang có

Số dưới đây tính theo lượt chụp; một lượt có thể mang nhiều đặc điểm. Cột 25 ảnh là tập con của 105 lượt dùng được.

| Trạng thái / đặc điểm | Dùng được | Bộ 25 ảnh |
|---|---:|---:|
| Đầy — không hở đáy, không xét chiều cao | 88 | 18 |
| Vơi — hở đáy, còn trên 3 bao | 7 | 2 |
| Còn 1–3 bao | 4 | 3 |
| Pallet rỗng | 6 | 2 |
| Bao chồng lấn mạnh | 8 | 3 |
| Bao xoay nhiều góc | 8 | 4 |
| Bao nhăn / phồng / rách / móp | 7 | 2 |
| Ánh sáng khác thường | 21 | 5 |
| Màu / loại bao khác thường | 49 | 9 |
| Pallet lệch / xoay | 8 | 3 |
| Bao tràn biên pallet | 98 | 22 |
| Hở đáy pallet | 17 | 7 |

25 ảnh gồm bao đỏ, xanh, trắng, nâu; cảnh nhiều pallet; chuỗi tháo hàng còn 3 → 2 → 1 → 0 bao trong khi pallet bên cạnh vẫn đầy. Bộ hiện thiên về pallet đầy; cần bổ sung cảnh vơi, ít bao và biến dạng sau vòng thử đầu.

## File và bước tiếp theo

- **RGB/depth gốc:** `bao-tai.nosync/`, `2026-09-11.nosync/`; có nhiều frame mỗi lượt.
- **Phân loại:** `code/image_sorter/results.nosync/labels.json` và `labels.csv`.
- **25 ảnh + nhãn:** `code/obb_labeler/selection.json`; `code/obb_labeler/results.nosync/annotations.json` là bản gốc, `manifest.csv` giữ nhóm cảnh; `yolo/` là bản xuất OBB chuẩn hóa. Hai lớp: `0=pallet_region` (gồm đế và bao tràn biên), `1=bag` (thân bao). JSON còn giữ pallet mục tiêu, liên kết bao→pallet và cờ che khuất/cắt biên.
- **Vòng đầu:** rà nhãn → fine-tune model OBB nhẹ có pretrained trên bộ seed 25 ảnh → dự đoán 80 lượt dùng được còn lại → người sửa → train lại. Chia train/val theo cảnh/chuỗi, không random các frame gần giống nhau; cần test độc lập.

## Điểm cần nhớ

- **Rà `session_043`:** B4–B14 chưa liên kết P1; B3 là bao rời bên phải nhưng đang liên kết P1. `session_031/043/058` còn bao thấy được ở pallet phụ chưa khoanh: bổ sung nếu train toàn ảnh, hoặc chuẩn bị crop pallet mục tiêu cho model bao; tránh coi bao chưa có nhãn là nền.
- Nhãn cũ trong `annotations.nosync/` ở ảnh ortho 600×600; **không dùng trực tiếp tọa độ đó trên RGB 1920×1080**.
- Lượt `141341_948804` được ghi chú depth hỏng trong bộ cũ; không nằm trong seed 25 ảnh. Kiểm tra trước khi dùng cho XYZ.
- **Chuyển máy:** giữ cấu trúc thư mục và `annotations.json`. Xóa riêng bản xuất `code/obb_labeler/results.nosync/yolo/`, rồi chạy `python3 code/obb_labeler/app.py` để dựng lại; ảnh xuất đang là symlink, danh sách ảnh chứa đường dẫn tuyệt đối. App chỉ cần Python ≥3.9; chi tiết thao tác trong `code/obb_labeler/README.md`.
