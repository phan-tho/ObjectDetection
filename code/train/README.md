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
labeler/sorter dùng: `phanbth/bao-tai-1` → `bao-tai.nosync/`, `phanbth/bao-tai-2` →
`2026-09-11.nosync/`. Cả hai đã trong `.gitignore`.

## 2. Dựng dataset

```sh
.venv/Scripts/python code/train/build_dataset.py --name fold_red_v2 \
    --val-groups august_red september_pallet_4 --aug 15 --scene
```

- Chia theo `sequence_group` (các frame cùng chuỗi gần như giống nhau, không được
  nằm cả ở train lẫn val). Val chỉ là ảnh gốc, không augment.
- `--aug N`: N bản augment offline cho mỗi ảnh train. `--copies-only`: nhân bản ảnh
  gốc (baseline cùng số bước). `--scene`: bật augment v2. Bỏ `--val-groups` để đưa
  cả 25 ảnh vào train cho model cuối.
- Ảnh thu về cạnh dài 1280 px, lưu JPEG; nhãn `class x1 y1 … x4 y4` chuẩn hóa.
- `session_031/043/058`: pallet phụ có bao chưa khoanh được tô xám và bỏ nhãn, để model
  không học "bao chưa nhãn = nền".
- Kết quả trong `datasets.nosync/<name>/` (gitignore), dựng lại được bất cứ lúc nào.

### Augment

Offline (`augment.py`, nhãn giữ nguyên trừ khi dán bao):

| Nhóm | Phép | Mô phỏng |
|---|---|---|
| Màu bao | `recolor_bags` | Bao loại khác màu (đỏ, cam, vàng, xanh lá, xanh dương, tím, trắng, xám, nâu, đen). Cả ảnh một màu; chỉ đổi điểm ảnh gần màu thân bao, giữ chữ in và nếp nhăn |
| Ánh sáng (≤3 phép) | nhiệt độ màu, sáng/tương phản, gamma, bão hòa, đốm chói, bóng đổ, tối góc | Đèn kho vàng/trắng, nắng chói lên bao nilon, bóng người/cánh tay/xà |
| Cảm biến (≤2 phép) | nhiễu, blur, motion blur, JPEG, hạ phân giải | Camera rẻ, rung, nén |
| Cảnh (v2, `--scene`) | `paste_bags` | Bao rời trên nền: cắt bao thật từ ảnh train, dán với kích thước 0.7–1.25×, góc tùy ý, có bóng, thêm nhãn bao |
| | `arm_occluder` | Tay người/cánh tay robot/ống che một phần bao |
| | `stretch_wrap` | Màng co nilon + dây đai quấn pallet |

Online (Ultralytics, trong `train.py`): xoay 0–180° (camera nhìn từ trên), lật ngang/dọc,
scale 0.5–1.5 (bao to/nhỏ, camera cao/thấp), dịch 15% (pallet lệch), perspective nhẹ,
mosaic (tắt 10 epoch cuối), HSV nhẹ.

## 3. Train + đánh giá

```sh
.venv/Scripts/python code/train/train.py --data fold_red_v2 --model yolo26n-obb.pt --name red_y26n_v2
```

Mặc định 60 epoch, imgsz 640, batch 8 (~13 phút trên RTX 3050 4GB). Model, log và ảnh
val trong `runs.nosync/<name>/`; chỉ số val của `best.pt` được nối vào
`runs.nosync/results.jsonl`.

## 4. Dự đoán ảnh chưa nhãn

```sh
.venv/Scripts/python code/train/predict.py --weights runs.nosync/<run>/weights/best.pt
```

Chạy trên các lượt "dùng được" trong `image_sorter` chưa có trong bộ 25 ảnh. Xuất
`runs.nosync/predict_<run>/predictions.json` (cùng schema box với `annotations.json`,
thêm `conf`) và `vis/*.jpg` để rà nhanh.

## Chạy trên Kaggle

```sh
.venv/Scripts/python code/train/kaggle/push.py            # upload code + nhãn, chạy kernel GPU
.venv/Scripts/python code/train/kaggle/push.py --status
.venv/Scripts/python code/train/kaggle/push.py --pull     # output về runs.nosync/kaggle/
```

Dùng tài khoản trong `~/.kaggle`. Code + `annotations.json` + `labels.csv` được upload thành
dataset private `<user>/bao-tai-obb-code`; ảnh lấy thẳng từ `phanbth/bao-tai-1/2`.
Kernel `kaggle/kernel.py` dựng mọi dataset, chạy các thí nghiệm song song trên các GPU,
train model cuối trên cả 25 ảnh rồi dự đoán ảnh chưa nhãn. Cần quota GPU (30 giờ/tuần).

## Quy ước góc

Một hình chữ nhật có nhiều cách ghi: (w, h, 30°) = (w, h, 210°) = (h, w, 120°).
Labeler giờ lưu một dạng duy nhất (long-edge, như MMRotate `le90`): `w` là cạnh dài,
`angle` là hướng cạnh dài trong [-90°, 90°), độ, chiều kim đồng hồ trên ảnh (y hướng
xuống). Bao đối xứng 180° nên đây cũng là góc kẹp cho robot. YOLO train từ 4 đỉnh và tự
chuẩn hóa nội bộ, nên quy ước này không ảnh hưởng train; `predict.py` đổi output về cùng
quy ước.
