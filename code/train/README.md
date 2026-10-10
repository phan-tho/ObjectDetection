# Train YOLO-OBB nano

Cần GPU NVIDIA. Môi trường riêng trong `.venv/` (đã gitignore), tạo bằng [uv](https://docs.astral.sh/uv/):

```sh
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/Scripts/python.exe ultralytics kaggle
```

(macOS/Linux: `.venv/bin/python`, bỏ `--index-url`.) Các lệnh dưới chạy từ thư mục dự án.

## 1. Tải ảnh gốc

```sh
.venv/Scripts/python code/train/download_data.py
```

Cần token Kaggle (`~/.kaggle/access_token` hoặc `kaggle.json`). Lưu đúng tên mà
labeler/sorter dùng: `phanbth/bao-tai-1` → `bao-tai/`, `phanbth/bao-tai-2` →
`2026-09-11/`. Cả hai đã trong `.gitignore`.

## 2. Dựng dataset

```sh
.venv/Scripts/python code/train/build_dataset.py --name red_v3 \
    --val-groups august_red september_pallet_4 --aug 30 --heavy
```

- Chia theo `sequence_group` (các frame cùng chuỗi gần như giống nhau, không được
  nằm cả ở train lẫn val). Val chỉ là ảnh gốc, không augment.
- `--aug N`: N bản augment offline cho mỗi ảnh train. `--copies-only`: nhân bản ảnh
  gốc (baseline cùng số bước). `--scene`: bật augment v2. `--heavy`: bật v3 (kèm v2). Bỏ `--val-groups` để đưa
  cả 25 ảnh vào train cho model cuối.
- Ảnh thu về cạnh dài 1280 px, lưu JPEG; nhãn `class x1 y1 … x4 y4` chuẩn hóa.
- `session_031/043/058`: pallet phụ có bao chưa khoanh được tô xám và bỏ nhãn, để model
  không học "bao chưa nhãn = nền".
- Kết quả trong `datasets/<name>/` (gitignore), dựng lại được bất cứ lúc nào.

### Augment

Offline (`augment.py`; nhãn giữ nguyên trừ khi dán bao hoặc biến đổi hình học v3):

| Nhóm | Phép | Mô phỏng |
|---|---|---|
| Màu bao | `recolor_bags` | Bao loại khác màu (đỏ, cam, vàng, xanh lá, xanh dương, tím, trắng, xám, nâu, đen). Cả ảnh một màu; chỉ đổi điểm ảnh gần màu thân bao, giữ chữ in và nếp nhăn |
| Ánh sáng (≤3 phép) | nhiệt độ màu, sáng/tương phản, gamma, bão hòa, đốm chói, bóng đổ, tối góc | Đèn kho vàng/trắng, nắng chói lên bao nilon, bóng người/cánh tay/xà |
| Cảm biến (≤2 phép) | nhiễu, blur, motion blur, JPEG, hạ phân giải | Camera rẻ, rung, nén |
| Cảnh (v2, `--scene`) | `paste_bags` | Bao rời trên nền: cắt bao thật từ ảnh train, dán với kích thước 0.7–1.25×, góc tùy ý, có bóng, thêm nhãn bao |
| | `arm_occluder` | Tay người/cánh tay robot/ống che một phần bao |
| | `stretch_wrap` | Màng co nilon + dây đai quấn pallet |
| Hình học (v3, `--heavy`, ≤1 phép) | `crop_zoom`, `zoom_out`, `rotate_scene`, `tilt` | Pallet bị cắt ở mép ảnh, bao nhỏ (camera cao), pallet nằm chéo, camera nghiêng. Nhãn cắt theo khung ảnh qua `clip_objects`: giữ hình chữ nhật bao phần còn thấy; bao còn <30% (pallet <15%) thì tô xám, bỏ nhãn. Chỗ trống lấp bằng nền giả, không phản chiếu ảnh |
| Mẫu âm (v3) | `paste_clutter` | Dán vật nền (cửa, kệ, người, carton) cắt từ vùng không có nhãn của ảnh train, không thêm nhãn: giảm báo nhầm ngoài pallet |
| Ảnh xấu (v3) | `stripes`, `defocus`, `grayscale`, `channel_shuffle`, `cutout` | Sọc nhấp nháy đèn LED/vạch lỗi cảm biến, mất nét, camera đen trắng, sai kênh màu, vật lạ che nhỏ |

Online (Ultralytics, trong `train.py`): xoay 0–180° (camera nhìn từ trên), lật ngang/dọc,
scale 0.5–1.5 (bao to/nhỏ, camera cao/thấp), dịch 15% (pallet lệch), perspective nhẹ,
mosaic (tắt 10 epoch cuối), HSV nhẹ.

## 3. Train + đánh giá

```sh
.venv/Scripts/python code/train/train.py --data fold_red_v2 --model yolo26n-obb.pt --name red_y26n_v2
```

Mặc định 60 epoch, imgsz 640, batch 8 (~13 phút trên RTX 3050 4GB). Model, log và ảnh
val trong `runs/<name>/`; chỉ số val của `best.pt` được nối vào
`runs/results.jsonl`.

Phân tích lỗi trên val (bỏ sót, báo nhầm, box lệch theo đặc điểm nhãn):

```sh
.venv/Scripts/python code/train/analyze_errors.py \
    --pair runs/red_v1/weights/best.pt red_v1 --pair runs/sep3_v1/weights/best.pt sep3_v1
```

Xuất `runs/errors/summary.txt`, `objects.csv` và `vis/` (xanh = đúng, đỏ = bỏ sót, vàng = box
lệch IoU < 0.75, cam = báo nhầm).

## 4. Dự đoán ảnh chưa nhãn

```sh
.venv/Scripts/python code/train/predict.py --weights runs/<run>/weights/best.pt
```

Chạy trên các lượt "dùng được" trong `image_sorter` chưa có trong bộ 25 ảnh. Xuất
`runs/predict_<run>/predictions.json` (cùng schema box với `annotations.json`,
thêm `conf`) và `vis/*.jpg` để rà nhanh.

## Chạy trên Kaggle

```sh
.venv/Scripts/python code/train/kaggle/push.py -a <user>            # upload code + nhãn, chạy kernel GPU
.venv/Scripts/python code/train/kaggle/push.py -a <user> --status
.venv/Scripts/python code/train/kaggle/push.py -a <user> --pull     # output về runs/kaggle/
```

`-a` chọn tài khoản trong `.kaggle/tokens.json` của dự án (gitignore); bỏ `-a` thì dùng
`~/.kaggle`. Mỗi tài khoản có quota GPU 30 giờ/tuần; hết thì đổi tài khoản. Code +
`annotations.json` + `labels.csv` được upload thành dataset private `<user>/bao-tai-obb-code`;
ảnh lấy thẳng từ `phanbth/bao-tai-1/2`.

Kernel `kaggle/kernel.py`: dựng dataset cho 4 fold theo nhóm cảnh (phủ cả 25 ảnh) × các cấu
hình (không augment / v1 / v3 / v3 ảnh 960), train song song trên 2 GPU, phân tích lỗi v1 và v3,
train model cuối trên cả 25 ảnh, dự đoán ảnh chưa nhãn. Bảng tổng hợp ở `runs/summary.txt`.

## Quy ước góc

Một hình chữ nhật có nhiều cách ghi: (w, h, 30°) = (w, h, 210°) = (h, w, 120°).
Labeler giờ lưu một dạng duy nhất (long-edge, như MMRotate `le90`): `w` là cạnh dài,
`angle` là hướng cạnh dài trong [-90°, 90°), độ, chiều kim đồng hồ trên ảnh (y hướng
xuống). Bao đối xứng 180° nên đây cũng là góc kẹp cho robot. YOLO train từ 4 đỉnh và tự
chuẩn hóa nội bộ, nên quy ước này không ảnh hưởng train; `predict.py` đổi output về cùng
quy ước.
