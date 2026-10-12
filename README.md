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

Model: **YOLO26n-OBB** (Ultralytics 8.4, nano, ~2.4M tham số, pretrained DOTA), 2 lớp
`pallet_region` và `bag`. Train trên GPU T4 của Kaggle: AdamW (Ultralytics tự chọn lr ≈ 0.0017),
batch 8, 60 epoch, seed 0; augment online xoay 0–180°, lật, scale ±50%, mosaic. Code và cách chạy
lại: `code/train/README.md`.

### Cách đánh giá

- **Chia theo nhóm cảnh, 4 fold phủ cả 25 ảnh** (`red` 4 ảnh, `sep3` 3, `green` 8, `sep1` 10): mỗi fold
  giữ lại một/vài `sequence_group` làm val, train trên phần còn lại. Mỗi ảnh làm val đúng một lần,
  không ảnh nào vừa train vừa val. Không chia ngẫu nhiên, vì các frame cùng chuỗi gần như trùng nhau.
- Chỉ số chính **mAP50-95** trung bình 2 lớp (IoU xoay kiểu probiou của Ultralytics). mAP50 gần như
  bão hòa; mAP50-95 phản ánh cả độ khít của box, cần cho robot gắp.
- **Chấm bằng model ở epoch cuối (`last.pt`)**, không dùng `best.pt`. `best.pt` là epoch có mAP val
  cao nhất, tức đã dùng chính tập val để chọn, nên điểm bị thổi lên. Ví dụ fold `red` không augment:
  0.880 ở epoch 8, còn epoch cuối chỉ 0.722.
- Vẫn còn hạn chế: không có tập test độc lập (cấu hình được chọn trên chính 4 fold này), fold nhỏ,
  và trước vòng 4 mỗi cấu hình chỉ chạy 1 seed. Chạy lại cùng cấu hình từng lệch ~2 điểm, nên
  chênh lệch dưới ~2 điểm coi như hòa.
- Số của vòng 1 và vòng 3 bên dưới vẫn chấm bằng `best.pt` (đã ghi chú). Vòng 2 có cả hai.

### Vòng 1: so sánh cấu hình (2 fold, 7 ảnh val, `best.pt`)

| Cấu hình | Fold bao đỏ (4 ảnh) | Fold cảnh khó (3 ảnh) | Trung bình |
|---|---:|---:|---:|
| Không augment (nhân bản ảnh gốc) | 0.880 | 0.696 | 0.788 |
| Augment v1: màu bao, ánh sáng, nhiễu | 0.888 | 0.726 | 0.807 |
| Augment v2: v1 + dán bao rời, che khuất, màng co | 0.861 | 0.711 | 0.786 |
| Augment v2, ảnh 960 | 0.893 | 0.727 | 0.810 |
| YOLO11n, augment v2 | 0.781 | 0.759 | 0.770 |

Chỉ dùng được một kết luận: **YOLO11n không tốt hơn YOLO26n** → giữ YOLO26n nano. Val 7 ảnh quá ít để
so các augment.

### Vòng 2: augment v3, 4 fold

Augment v3 (`--heavy`) thêm biến đổi hình học có cắt nhãn theo khung ảnh (cắt/zoom, thu nhỏ, xoay cả
ảnh, nghiêng phối cảnh), mẫu âm (dán vật nền cắt từ vùng không nhãn), sọc nhấp nháy đèn, vạch lỗi
cảm biến, mất nét, ảnh xám, che ngẫu nhiên. v3 có 30 bản augment mỗi ảnh, v1 có 15.

| mAP50-95, trung bình 10 epoch cuối | red | sep3 | green | sep1 | **TB 4 fold** | (best.pt, lạc quan) |
|---|---:|---:|---:|---:|---:|---:|
| Không augment | 0.713 | 0.594 | 0.729 | 0.858 | **0.724** | 0.807 |
| Augment v1 | 0.855 | 0.651 | 0.786 | 0.898 | **0.798** | 0.830 |
| Augment v3 | 0.798 | 0.680 | 0.758 | 0.938 | **0.794** | 0.843 |
| Augment v3, ảnh 960 | 0.852 | 0.746 | 0.804 | 0.919 | **0.830** | 0.854 |

- **Augment là thay đổi có tác dụng lớn nhất: +7.4 điểm.** Không augment thì model học vẹt rất nhanh:
  đạt đỉnh ở epoch ~8 rồi tụt dần.
- **v3 không hơn v1** ở ảnh 640, dù v3 được train gấp đôi số bước (30 so với 15 bản/ảnh).
- **Ảnh 960 giúp thật: +3.6 điểm** so với 640, rõ nhất ở `sep3` (nhiều bao nhỏ).

### Vòng 3: phương pháp suy luận (4 fold, `best.pt`)

Chấm bằng `code/train/pipeline.py` trên ảnh gốc 1920×1080. Bộ chấm tính mAP như Ultralytics val, đã
kiểm tra khớp tới 4 chữ số khi cùng tiền xử lý.

| Phương pháp | v1 640 | v3 640 | v3 960 |
|---|---:|---:|---:|
| plain: một lần trên cả ảnh | 0.819 | 0.812 | **0.838** |
| tiled: thêm bao từ các ô 960×960 của ảnh gốc | 0.776 | 0.797 | 0.829 |
| two_stage: tìm pallet → crop ảnh gốc → model crop tìm bao | 0.827 | 0.811 | – |
| plain + chỉ giữ bao trong pallet | 0.816 | 0.805 | 0.834 |
| v3 + đóng băng backbone (`--freeze 11`) | – | 0.682 | – |

- **Đóng băng backbone tệ hẳn** (−13 điểm): backbone pretrained trên ảnh vệ tinh DOTA không hợp với
  ảnh kho, cần được fine-tune.
- **Tiling tệ hơn**, vì model chưa học bao ở kích thước lớn đó. **Hai giai đoạn** hơn plain dưới 1 điểm
  (nằm trong nhiễu) mà phức tạp hơn nhiều. **Ảnh 960 vẫn là cách đơn giản và tốt nhất.**
- Lọc bao trong pallet không tăng mAP, vì bao rời có nhãn ngoài pallet cũng bị bỏ. Nó chỉ có ích ở
  đầu ra cho robot, để bỏ báo nhầm ngoài pallet.
- NMS chặt hơn khi suy luận (IoU 0.5 hoặc 0.4 thay vì 0.7) cũng không giúp (kiểm tra với `last.pt`).

### Vòng 4: sửa các case kém (đang chạy)

Phân tích lỗi cấu hình tốt nhất (v3 960, `last.pt`, cả 25 ảnh, `runs/errors_v3_960/`). Lỗi = bỏ sót
hoặc box lệch (IoU < 0.75):

| Nhóm bao (254 bao) | Lỗi |
|---|---:|
| Tổng | 56 (39 lệch, 17 bỏ sót) |
| **Bị bao khác đè lên** (56 bao) | **45%** |
| Không bị che (198 bao) | 16% |
| Nhỏ, dưới 1% ảnh (12) / 1–2% ảnh (61) | 83% / 43% |
| To, ≥ 2% ảnh (181) | 11% |
| Trên pallet mục tiêu / pallet khác / bao rời | 22% / 22% / 25% |

Báo nhầm ở conf ≥ 0.25: 28 box bao, 19 box pallet. Ảnh kém nhất và nguyên nhân thấy trên ảnh:

| Ảnh | Nguyên nhân | Augment v4 (`--v4`) sửa |
|---|---|---|
| `session_080/084/087/077` (bao xanh) | Bao nằm chéo đè lên bao khác. Nhãn chỉ khoanh **phần thân còn thấy** nhưng train ít mẫu loại này | `occlude_bags`: dán bao thật đè một phần lên bao có sẵn, tính lại nhãn của bao bên dưới theo phần còn thấy (đúng quy ước gán nhãn), nới vùng pallet |
| `session_080/084/087`, `142952` | Người, áo phản quang, vạch sơn, tấm cửa bị nhận là bao/pallet | Mẫu âm dày hơn: xác suất 0.5 → 0.7, tối đa 5 vật/ảnh, mảng to hơn |
| `session_062` | Pallet sắt rỗng bị sót, chỉ khoanh được từng mảnh. Cả bộ chỉ có 1 pallet sắt rỗng | `paste_empty_pallet`: dán pallet rỗng (cắt từ ảnh train) vào chỗ trống |
| `142952`, `142402` | Nhiều pallet, bao bọc màng co, lát bao nhỏ sát mép | Giữ ảnh 960. Phần còn lại là giới hạn của dữ liệu |

So sánh công bằng ở vòng 4: v1, v3, v4 **cùng ảnh 960, cùng 30 bản augment/ảnh, mỗi cấu hình 2 seed**,
chấm `last.pt` bằng `pipeline.py`. Model cuối: v4 960 trên cả 25 ảnh.

### Model và dự đoán hiện có

- Model cuối vòng 2 (cả 25 ảnh, augment v3, 100 epoch, không có val): `runs/kaggle/runs/all_v3/weights/last.pt`
  và bản 960 `all_v3_960`. Dự đoán 80 lượt chưa nhãn: `runs/kaggle/runs/predict_all_v3*/`
  (`predictions.json` cùng schema box với `annotations.json`, `vis/*.jpg` để xem).
- Bản vòng 1 (augment v2) ở `runs/kaggle_v1/`. Thư mục `runs/` không đưa lên git; tải lại bằng
  `code/train/kaggle/push.py -a <tài khoản> [--stage methods] --pull`.

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
