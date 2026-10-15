# Nhận diện bao để robot gắp

Ảnh màu + depth từ camera nhìn xuống pallet → tìm pallet mục tiêu (gần giữa ảnh) → tìm từng bao (box
xoay, OBB) → chọn bao trên cùng không bị đè → toạ độ XYZ và góc kẹp cho robot. Camera cần thấy toàn
bộ pallet mục tiêu.

## Kết quả

**Nhận diện** (mAP50-95, trung bình 4 fold chia theo nhóm cảnh, chấm model epoch cuối `last.pt`;
xem [Cách đánh giá](#cách-đánh-giá)):

| Cấu hình | mAP50-95 | Ghi chú |
|---|---:|---|
| YOLO26n, ảnh 640, không augment | 0.724 | Đối chứng |
| YOLO26n, ảnh 640, augment v1 | 0.798 | Augment là thay đổi có tác dụng lớn nhất |
| YOLO26n, ảnh 960, augment v3 | 0.815 | Ảnh 960 giúp bao nhỏ; thêm augment v4/v5 không tăng nữa |
| **YOLO26s, ảnh 960, augment v3** | **0.864** | **Model đang dùng.** Bao 0.835, pallet 0.892 |
| YOLO26s, 2 seed + xoay/lật 8 hướng | 0.877 | Chậm hơn 16 lần |

- Model tìm thấy gần hết bao (mAP50 trung bình 0.97, từng fold 0.93–0.99). Điểm còn mất chủ yếu do box chưa khít, ở bao bị bao
  khác đè và bao nhỏ sát mép ảnh.
- Tốc độ nhận diện trên laptop (RTX 3050, ảnh 960): 22 ms/ảnh với GPU, 176 ms/ảnh với CPU.
- Hạn chế: chỉ có 25 ảnh có nhãn, chưa có tập test độc lập. Chênh lệch dưới khoảng 2 điểm coi như bằng
  nhau.

**Chọn bao cần gắp** (`code/grasp/`, rule-based trên box + depth, không cần train):
- 25 ảnh có nhãn tay: 23 ảnh có bao đều chọn được bao để gắp, bao gắp trước không lần nào là bao bị che;
  2 ảnh pallet rỗng báo đúng.
- 80 ảnh chưa nhãn, dùng box của model: 76 ảnh chọn được bao, 4 ảnh pallet rỗng.
- Kích thước bao đo từ depth trung bình 685×443 mm (thật 700×450), tức đổi pixel sang mm đúng.
- Toạ độ đang ở hệ camera; cần hiệu chuẩn hand-eye để ra hệ robot.

## Cách chạy

**Demo** (chỉ suy luận, không train). Hướng dẫn cài đầy đủ cho Windows/macOS/Linux:
`code/demo/README.md`.

```sh
pip install -r requirements.txt                    # trong .venv; có GPU thì cài torch CUDA trước
python code/train/download_data.py                 # ảnh gốc, ~2.4 GB, cần token Kaggle
python code/demo/weights.py download               # weights, cần được share dataset trên Kaggle
streamlit run code/demo/streamlit_app.py
```

**Các phần khác:**

| Việc | Ở đâu |
|---|---|
| Train, đánh giá, phân tích lỗi (chạy trên Kaggle) | `code/train/README.md` |
| Gán nhãn, sửa nhãn nháp | `code/obb_labeler/README.md` |
| Chọn bao + XYZ từ file dự đoán | `code/grasp/README.md` |
| Phân loại lượt chụp | `code/image_sorter/README.md` |

## Dữ liệu

123 lượt chụp (camera Orbbec Gemini 2, ảnh màu 1920×1080 + depth đã khớp ảnh màu), trong đó 105 dùng
được, 18 loại. **25 ảnh đã gán nhãn và kiểm tra: 49 box pallet, 254 box bao.** 80 lượt còn lại đã được
nạp vào labeler dạng nháp (dự đoán của model), chờ người sửa.

Số dưới đây tính theo lượt chụp; một lượt có thể mang nhiều đặc điểm. Cột 25 ảnh là tập con của 105
lượt dùng được.

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

- 25 ảnh gồm bao đỏ, xanh, trắng, nâu; cảnh nhiều pallet; chuỗi tháo hàng còn 3 → 2 → 1 → 0 bao trong
  khi pallet bên cạnh vẫn đầy. Bộ hiện thiên về pallet đầy.
- Tất cả 105 lượt thuộc **6 chuỗi chụp** (`sequence_group`). 80 lượt chưa nhãn nằm xen giữa các lượt
  đã có nhãn, cách nhau vài giây. Muốn có tập test độc lập phải chụp thêm phiên mới.
- Hai lớp nhãn: `0=pallet_region` (đế pallet và mọi bao trên đó, kể cả bao tràn biên), `1=bag` (thân
  bao; bao bị đè chỉ khoanh phần còn thấy).

| Dữ liệu | Ở đâu |
|---|---|
| Ảnh màu/depth gốc (không vào git, tải từ Kaggle `phanbth/bao-tai-1`, `bao-tai-2`) | `bao-tai/`, `2026-09-11/` |
| Phân loại lượt chụp | `code/image_sorter/results/labels.json`, `labels.csv` |
| Danh sách ảnh, nhóm chuỗi chụp | `code/obb_labeler/selection.json` |
| Nhãn (bản gốc, gồm cả nháp) | `code/obb_labeler/results/annotations.json`; JSON còn giữ pallet mục tiêu, liên kết bao → pallet, cờ che khuất/cắt biên |
| Bản xuất YOLO-OBB của ảnh đã kiểm tra | `code/obb_labeler/results/yolo/` |

## Cách đánh giá

- **Chia theo nhóm cảnh, 4 fold phủ cả 25 ảnh** (`red` 4 ảnh, `sep3` 3, `green` 8, `sep1` 10): mỗi fold
  giữ lại một/vài `sequence_group` làm val, train trên phần còn lại. Mỗi ảnh làm val đúng một lần,
  không ảnh nào vừa train vừa val. Không chia ngẫu nhiên, vì các frame cùng chuỗi gần như trùng nhau.
- Chỉ số chính **mAP50-95** trung bình 2 lớp (IoU xoay kiểu probiou của Ultralytics). mAP50 gần như
  bão hòa; mAP50-95 phản ánh cả độ khít của box, cần cho robot gắp.
- **"Ảnh 640/960/1280"** là kích thước ảnh đưa vào model (`imgsz`): ảnh gốc 1920×1080 được thu nhỏ
  để cạnh dài bằng số này. Ảnh càng lớn thì bao nhỏ càng rõ, nhưng chạy chậm hơn.
- **Số chính chấm bằng model ở epoch cuối (`last.pt`)**. `best.pt` là epoch có mAP val cao nhất, tức
  đã dùng chính tập val để chọn, nên điểm bị thổi lên. Ví dụ fold `red` không augment: 0.880 ở
  epoch 8, còn epoch cuối chỉ 0.722. Từ vòng 4, số `best.pt` vẫn được ghi kèm để tham khảo, gồm 2
  cột: chấm lại bằng `pipeline.py` trên ảnh gốc (cùng cách với `last.pt`), và điểm Ultralytics val
  lúc train (chính tập dùng để chọn, nên cao nhất).
- Vẫn còn hạn chế: không có tập test độc lập (cấu hình được chọn trên chính 4 fold này), fold nhỏ,
  và trước vòng 4 mỗi cấu hình chỉ chạy 1 seed. Chạy lại cùng cấu hình từng lệch ~2 điểm, nên
  chênh lệch dưới ~2 điểm coi như hòa.
- Số của vòng 1 và vòng 3 bên dưới vẫn chấm bằng `best.pt` (đã ghi chú). Vòng 2 dùng trung bình 10
  epoch cuối, tương đương `last.pt`.

## Chi tiết các vòng thử nghiệm

Model: **YOLO26-OBB** (Ultralytics 8.4, pretrained DOTA), 2 lớp `pallet_region` và `bag`. Vòng 1–5
dùng bản nano (`n`, 2.6M tham số), vòng 6 thử bản `s` (10.5M). Train trên GPU T4 của Kaggle: AdamW
(Ultralytics tự chọn lr ≈ 0.0017), batch 8, 60 epoch; augment online xoay 0–180°, lật, scale ±50%,
mosaic. Code và cách chạy lại: `code/train/README.md`.

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

### Vòng 4: sửa các case kém (augment v4)

Phân tích lỗi cấu hình tốt nhất của vòng 2 (v3 960, `last.pt`, cả 25 ảnh, `runs/errors_v3_960/`).
Lỗi = bỏ sót hoặc box lệch (IoU < 0.75):

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

So sánh công bằng: v1, v3, v4 **cùng ảnh 960, cùng 30 bản augment/ảnh**, v3 và v4 chạy 2 seed (v1
chỉ 1 seed), chấm bằng `pipeline.py` (`runs/kaggle_fix/`). Số theo fold là trung bình các seed.

| Augment | red | sep3 | green | sep1 | **TB `last.pt`** | `best.pt` (chấm lại) | `best.pt` (Ultralytics val) |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | 0.802 | 0.663 | 0.790 | 0.921 | **0.794** | 0.767 | 0.822 |
| v3 | 0.826 | 0.724 | 0.796 | 0.917 | **0.815** | 0.829 | 0.851 |
| v4 | 0.810 | 0.732 | 0.781 | 0.912 | **0.809** | 0.825 | 0.848 |

- **v4 không hơn v3** (chênh 0.6 điểm, nằm trong nhiễu; 2 seed của v4 lệch nhau 2.7 điểm). Dán bao đè
  bao không làm giảm lỗi ở bao bị đè: 45% với v3, 52% với v4. v3 hơn v1 khoảng 2 điểm.
- `best.pt` trên Ultralytics val cao hơn `last.pt` 3–4 điểm. Chấm lại cùng các ảnh đó bằng ảnh gốc thì
  chỉ còn hơn 1–2 điểm, v1 còn thấp hơn 2.7 điểm. Nên vẫn lấy `last.pt` làm số chính.
- Model cuối: v4 960 trên cả 25 ảnh (`runs/kaggle_fix/runs/all_v4_960/`).

### Vòng 5: augment v5 (ánh sáng theo lớp xếp)

v5 (`--v5`) thêm vào v4: bao lớp dưới tối hơn và có bóng ở mép tiếp giáp, mỗi bao lệch màu một chút,
vệt bóng loáng và cháy sáng, bao dán bị phồng/méo. Cùng điều kiện với vòng 4 (`runs/kaggle_v5/`).

| Augment | red | sep3 | green | sep1 | **TB `last.pt`** | `best.pt` (chấm lại) | `best.pt` (Ultralytics val) |
|---|---:|---:|---:|---:|---:|---:|---:|
| v3 (vòng 4) | 0.826 | 0.724 | 0.796 | 0.917 | **0.815** | 0.829 | 0.851 |
| v5 | 0.781 | 0.720 | 0.779 | 0.902 | **0.796** | 0.806 | 0.838 |

- **v5 kém v3 khoảng 2 điểm**, cả 2 seed (0.796, 0.795) đều thấp hơn 2 seed của v3 (0.820, 0.810).
- Từ v3 đến v5, thêm augment không còn tăng điểm. Dừng hướng thêm augment.

### Vòng 6: model to hơn, ảnh lớn hơn, ensemble

Cùng augment v3 (30 bản/ảnh), 2 seed, chấm bằng `pipeline.py` (`runs/kaggle_s960/`, `runs/kaggle_n1280/`).

| Model | Ảnh | red | sep3 | green | sep1 | **TB `last.pt`** | bao | pallet | `best.pt` (chấm lại) | `best.pt` (Ultralytics val) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO26n (vòng 4) | 960 | 0.826 | 0.724 | 0.796 | 0.917 | **0.815** | 0.785 | 0.846 | 0.829 | 0.851 |
| YOLO26n | 1280 | 0.839 | 0.678 | 0.764 | 0.897 | **0.794** | 0.791 | 0.797 | 0.798 | 0.838 |
| **YOLO26s** | 960 | 0.860 | 0.825 | 0.829 | 0.943 | **0.864** | 0.835 | 0.892 | 0.866 | 0.884 |

- **YOLO26s tốt hơn rõ: +4.9 điểm.** Cả 2 seed (0.856, 0.872) đều cao hơn seed tốt nhất của nano
  (0.820). Tăng ở cả bao và pallet, nhiều nhất ở `sep3` (+10 điểm, nhiều bao nhỏ).
- So lỗi trên cả 25 ảnh (seed 0, `runs/kaggle_s960/runs/errors_s960/`): số bao sai giảm từ 56 xuống
  39. Bao bị đè sai 45% → 36%, bao 1–2% ảnh sai 43% → 28%. Báo nhầm bao 28 → 17, báo nhầm pallet
  19 → 6. Bao dưới 1% ảnh vẫn sai 11/12.
- **Ảnh 1280 không giúp:** nano ở 1280 kém nano ở 960 khoảng 2 điểm. Dừng hướng này.
- Tốc độ, đo trên laptop (RTX 3050, ảnh 960, 1 ảnh/lần): nano 20 ms GPU / 91 ms CPU; YOLO26s 22 ms
  GPU / 176 ms CPU. Cần đo lại trên máy của robot.

**Ensemble và TTA** (`code/train/ensemble.py`): chạy model trên 8 hướng xoay/lật ảnh, rồi gộp box của
các lần chạy bằng WBF (trung bình có trọng số).

| Cách chạy | nano 960 | YOLO26s 960 | Thời gian |
|---|---:|---:|---:|
| 1 model (seed 0) | 0.820 | 0.856 | 1× |
| 1 model + TTA 8 hướng | 0.813 | 0.863 | 8× |
| 2 seed | 0.807 | 0.873 | 2× |
| 2 seed + TTA 8 hướng | 0.837 | **0.877** | 16× |
| 4 model (v3 + v4, mỗi bản 2 seed) + TTA 8 hướng | 0.841 | – | 32× |

- Ensemble 2 seed + TTA tăng thêm 1–2 điểm nhưng chạy chậm 16 lần. Chỉ đáng dùng nếu robot không cần nhanh.
- Gộp bằng NMS thay vì WBF thì kém hẳn.

**Nhiễu nhãn** (`code/train/label_noise.py`): so nhãn của cùng một vật trên các cặp ảnh gần như giống
hệt nhau (45 cặp bao, 11 cặp pallet). mAP OBB của Ultralytics tính IoU bằng **probiou**, nên phải đọc
theo probiou. Hai lần khoanh cùng một bao khớp với probiou trung vị 0.959, pallet 0.980. Ước tính
trần mAP50-95 do nhiễu nhãn: **khoảng 0.93–0.96 với bao, 0.98 với pallet**. Vậy mức 0.9x vẫn đạt
được. (Bản trước ghi trần 0.81: con số đó đo bằng IoU đa giác, sai thước đo.) Mẫu còn nhỏ, số chỉ
để tham khảo.

## Chọn bao cần gắp

Rule-based, chạy sau bước nhận diện (`code/grasp/pick_xyz.py`, chi tiết và số đo nhiễu depth trong
`code/grasp/README.md`):

1. Pallet mục tiêu là pallet gần giữa ảnh; bao của nó là bao có tâm nằm trong pallet.
2. Mặt trên mỗi bao: khớp mặt phẳng lên depth ở 60% vùng giữa box. Điểm gắp là giao của tia qua tâm box
   với mặt phẳng; góc kẹp theo cạnh dài của box.
3. Độ cao đo theo hướng "lên" của chồng bao (camera có thể nghiêng tới 15°). Chỉ xét bao lớp trên cùng.
4. Bỏ bao bị đè: so depth đo thật với mặt bao ngay chỗ hai box chạm nhau.
5. Gắp bao cao nhất trước.

Depth nhiễu 4–19 mm mỗi pixel, nhỏ so với độ dày một lớp bao (130 mm). Code không đọc depth của pixel lẻ
mà khớp mặt phẳng trên hàng nghìn điểm. Không cần lọc ảnh depth.

## Model và dự đoán hiện có

- **YOLO26s 960** mới có bản theo fold: `runs/kaggle_s960/runs/<fold>_s960_s{0,1}/weights/last.pt`.
  Bản seed 0 là weights của app demo (Kaggle dataset `tuananhpham29/bao-tai-obb-weights`). Chưa train
  bản trên cả 25 ảnh.
- Model nano trên cả 25 ảnh (100 epoch, không có val): v3 640/960 `runs/kaggle/runs/all_v3*/`,
  v4 960 `runs/kaggle_fix/runs/all_v4_960/`, v5 960 `runs/kaggle_v5/runs/all_v5_960/`. Dự đoán 80
  lượt chưa nhãn: `predict_all_*` trong cùng thư mục (`predictions.json` cùng schema box với
  `annotations.json`, `vis/*.jpg` để xem).
- Bản vòng 1 (augment v2) ở `runs/kaggle_v1/`. Thư mục `runs/` không đưa lên git; tải lại bằng
  `code/train/kaggle/push.py -a <tài khoản> --stage <stage> --pull`.

## Hướng tiếp theo

1. **Train lâu hơn và model to hơn** (đang chạy trên Kaggle): YOLO26s 120 epoch, YOLO26m, YOLO26l, rồi
   ensemble giữa các model. Mục tiêu 0.9x.
2. **Sửa 80 ảnh nháp** trong labeler (`code/obb_labeler/README.md`, mục "Sửa nhãn nháp"), rồi train lại
   với ~100 ảnh.
3. **Thống nhất cách khoanh bao bị đè** và sửa lại khoảng 56 nhãn loại này. Đây là nhóm lỗi lớn nhất.
4. **Chụp thêm phiên mới** (cảnh vơi, ít bao, nhiều pallet, bao bọc màng co) để có tập test độc lập.
5. Train model cuối trên toàn bộ ảnh có nhãn, cập nhật weights cho demo.
6. Hiệu chuẩn hand-eye và thử trên robot; đo tốc độ trên máy của robot.

## Điểm cần nhớ

- **Rà `session_043`:** B4–B14 chưa liên kết P1; B3 là bao rời bên phải nhưng đang liên kết P1. `session_031/043/058` còn bao thấy được ở pallet phụ chưa khoanh: bổ sung nếu train toàn ảnh, hoặc chuẩn bị crop pallet mục tiêu cho model bao; tránh coi bao chưa có nhãn là nền.
- Nhãn cũ trong `annotations/` ở ảnh ortho 600×600; **không dùng trực tiếp tọa độ đó trên RGB 1920×1080**.
- Lượt `141341_948804` được ghi chú depth hỏng trong bộ cũ; không nằm trong seed 25 ảnh. Kiểm tra trước khi dùng cho XYZ.
- **Chuyển máy:** giữ cấu trúc thư mục và `annotations.json`. Xóa riêng bản xuất `code/obb_labeler/results/yolo/`, rồi chạy `python3 code/obb_labeler/app.py` để dựng lại; ảnh xuất đang là symlink, danh sách ảnh chứa đường dẫn tuyệt đối. App chỉ cần Python ≥3.9; chi tiết thao tác trong `code/obb_labeler/README.md`.
