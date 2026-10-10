# Nhận diện bao để robot gắp

**Mục tiêu:** RGB → tìm pallet mục tiêu gần giữa ảnh → tìm bao → dùng depth/rule chọn bao trên cùng có thể gắp → hiệu chuẩn tọa độ sang XYZ robot. Camera cần thấy toàn bộ pallet mục tiêu. OBB chưa phải nhãn điểm gắp/XYZ.

**Hiện tại:** đã phân loại 123 lượt chụp (105 dùng được, 18 loại), đã đánh nhãn và xác nhận 25 ảnh RGB đại diện: **49 box pallet, 254 box bao**. Đã fine-tune YOLO26n-OBB có augment trên Kaggle (xem [Kết quả thử nghiệm](#kết-quả-thử-nghiệm)); model dự đoán được 80 lượt chưa nhãn, cần người rà để làm vòng nhãn tiếp theo.

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

- **RGB/depth gốc:** `bao-tai/`, `2026-09-11/`; có nhiều frame mỗi lượt.
- **Phân loại:** `code/image_sorter/results/labels.json` và `labels.csv`.
- **25 ảnh + nhãn:** `code/obb_labeler/selection.json`; `code/obb_labeler/results/annotations.json` là bản gốc, `manifest.csv` giữ nhóm cảnh; `yolo/` là bản xuất OBB chuẩn hóa. Hai lớp: `0=pallet_region` (gồm đế và bao tràn biên), `1=bag` (thân bao). JSON còn giữ pallet mục tiêu, liên kết bao→pallet và cờ che khuất/cắt biên.
- **Vòng đầu:** rà nhãn → fine-tune model OBB nhẹ có pretrained trên bộ seed 25 ảnh → dự đoán 80 lượt dùng được còn lại → người sửa → train lại. Chia train/val theo cảnh/chuỗi, không random các frame gần giống nhau; cần test độc lập.

## Kết quả thử nghiệm

Model: **YOLO26n-OBB** (nano, pretrained DOTA), 2 lớp `pallet_region` và `bag`, ảnh 640, 60 epoch,
train trên GPU T4 của Kaggle. Code và cách chạy lại: `code/train/README.md`. Chỉ số chính là
**mAP50-95** trên tập val; mAP50 gần như bão hòa (0.84–0.99), còn mAP50-95 phản ánh cả độ khít
của box, vốn quan trọng khi robot gắp.

Val chia **theo nhóm cảnh** (`sequence_group`), không chia ngẫu nhiên: các frame cùng chuỗi gần như
giống nhau, chia ngẫu nhiên sẽ làm điểm ảo cao. Mọi ảnh val là cảnh model chưa thấy khi train.

### Vòng 1: so sánh cấu hình (2 fold, 7 ảnh val)

| Cấu hình | Fold bao đỏ (4 ảnh) | Fold cảnh khó (3 ảnh) | Trung bình |
|---|---:|---:|---:|
| Không augment (nhân bản ảnh gốc) | 0.880 | 0.696 | 0.788 |
| Augment v1: màu bao, ánh sáng, nhiễu | 0.888 | 0.726 | 0.807 |
| Augment v2: v1 + dán bao rời, che khuất, màng co | 0.861 | 0.711 | 0.786 |
| Augment v2, ảnh 960 | 0.893 | 0.727 | **0.810** |
| YOLO11n, augment v2 | 0.781 | 0.759 | 0.770 |

- Augment giúp, nhưng mức lợi chỉ khoảng 2–4 điểm. Cùng một cấu hình chạy hai lần (máy local và
  Kaggle) lệch nhau tới ~2 điểm, nên **chưa phân định được v1, v2 và 960**: chênh lệch nằm trong
  nhiễu vì val chỉ có 7 ảnh.
- YOLO11n không tốt hơn YOLO26n (kém hẳn ở fold bao đỏ) → giữ **YOLO26n nano**.
- Lớp bao yếu hơn lớp pallet (bao 0.66–0.87, pallet 0.74–0.95).

### Phân tích lỗi (augment v1, 7 ảnh val, `code/train/analyze_errors.py`)

So khớp box dự đoán với nhãn bằng IoU xoay. Bắt được khi IoU ≥ 0.5; box lệch khi IoU < 0.75.

| Nhóm bao | Lỗi (bỏ sót + box lệch) |
|---|---:|
| **Bị cắt bởi biên ảnh** | **9/9 (100%)** |
| Nằm sát mép ảnh | 10/18 (56%) |
| **Nhỏ, dưới 1% diện tích ảnh** | **9/10 (90%)** |
| Nhỏ, 1–2% diện tích | 7/17 (41%) |
| Bị che khuất | 5/18 (28%) |
| Không bị che | 21/75 (28%) |
| Còn lại (to, ở giữa) | 15–20% |

- Tổng: 93 bao → 67 đúng, 12 lệch, 14 bỏ sót. Pallet: 18 → 14 đúng, 4 lệch, không bỏ sót.
- **24 box báo nhầm**: 13 trong vùng pallet (tách sai ranh giới bao xếp sát nhau, nhất là bao bọc
  màng co), 11 ngoài pallet (tấm cửa/trần độ tin cậy 0.92, kệ, áo phản quang của công nhân).
- Che khuất **không** làm tăng lỗi. Lỗi tập trung ở **mép ảnh**, **bao nhỏ** và **vật nền giống bao**.
- Ảnh khó nhất: `142952` (nhiều pallet, bao nhỏ bọc màng co ở mép: 7 bỏ sót, 10 báo nhầm).

### Vòng 2: augment v3 nhắm vào các lỗi trên (đang chạy)

| Lỗi | Augment v3 thêm vào (`code/train/augment.py`) |
|---|---|
| Bao cắt biên | Cắt/zoom ngẫu nhiên để pallet bị cắt ở mép; xoay cả ảnh góc bất kỳ; nghiêng phối cảnh. Nhãn được cắt theo khung ảnh (giữ phần còn thấy, bao còn quá ít thì tô xám) |
| Bao nhỏ | Thu nhỏ cả cảnh (bao 0.45–0.8×) trên nền giả; thử thêm ảnh 960 |
| Báo nhầm ngoài pallet | Dán vật nền (cửa, kệ, người, carton) cắt từ vùng không có nhãn, làm mẫu âm |
| Ảnh xấu | Sọc nhấp nháy đèn LED, vạch lỗi cảm biến, mất nét, ảnh xám, đảo kênh màu, che ngẫu nhiên |

Số bản augment mỗi ảnh tăng từ 15 lên 30. Đánh giá đổi sang **4 fold theo nhóm cảnh phủ cả 25
ảnh** (mỗi ảnh làm val đúng một lần) để giảm nhiễu so sánh.

### Model và dự đoán hiện có

- Model cuối vòng 1 (cả 25 ảnh, augment v2): weights ở `runs/kaggle/runs/all_y26n_v2/weights/last.pt`,
  kèm bản 960. Thư mục `runs/` không đưa lên git; tải lại bằng
  `code/train/kaggle/push.py -a <tài khoản> --pull`.
- Dự đoán 80 lượt chưa nhãn: `runs/kaggle/runs/predict_all_y26n_v2/` (`predictions.json` cùng
  schema box với `annotations.json`, `vis/*.jpg` để xem). Pallet đầy/vơi/ít/rỗng đều khoanh đúng
  về cơ bản. Lỗi thấy được: áo phản quang và một vật nhỏ trên sàn bị nhận thành bao; 13/986 box có
  độ tin cậy dưới 0.5.

### Hướng tiếp theo

1. **Thêm nhãn là đòn bẩy lớn nhất**: nạp 80 dự đoán vào `obb_labeler` dạng nháp, người sửa và xác
   nhận, rồi train lại với ~100 ảnh. Ưu tiên cảnh vơi, ít bao, nhiều pallet, bao bọc màng co.
2. Có ~100 ảnh rồi thì tách hẳn một **tập test độc lập** (chuỗi chụp chưa dùng), không dùng để chọn
   cấu hình.
3. Nếu bao nhỏ vẫn lỗi: train ảnh 960, hoặc crop vùng pallet mục tiêu rồi mới tìm bao (hai giai đoạn).

## Điểm cần nhớ

- **Rà `session_043`:** B4–B14 chưa liên kết P1; B3 là bao rời bên phải nhưng đang liên kết P1. `session_031/043/058` còn bao thấy được ở pallet phụ chưa khoanh: bổ sung nếu train toàn ảnh, hoặc chuẩn bị crop pallet mục tiêu cho model bao; tránh coi bao chưa có nhãn là nền.
- Nhãn cũ trong `annotations/` ở ảnh ortho 600×600; **không dùng trực tiếp tọa độ đó trên RGB 1920×1080**.
- Lượt `141341_948804` được ghi chú depth hỏng trong bộ cũ; không nằm trong seed 25 ảnh. Kiểm tra trước khi dùng cho XYZ.
- **Chuyển máy:** giữ cấu trúc thư mục và `annotations.json`. Xóa riêng bản xuất `code/obb_labeler/results/yolo/`, rồi chạy `python3 code/obb_labeler/app.py` để dựng lại; ảnh xuất đang là symlink, danh sách ảnh chứa đường dẫn tuyệt đối. App chỉ cần Python ≥3.9; chi tiết thao tác trong `code/obb_labeler/README.md`.
