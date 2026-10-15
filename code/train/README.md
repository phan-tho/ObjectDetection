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
.venv/Scripts/python code/train/build_dataset.py --name red_v4 \
    --val-groups august_red september_pallet_4 --aug 30 --v4
```

- Chia theo `sequence_group` (các frame cùng chuỗi gần như giống nhau, không được
  nằm cả ở train lẫn val). Val chỉ là ảnh gốc, không augment.
- `--aug N`: N bản augment offline cho mỗi ảnh train. `--copies-only`: nhân bản ảnh
  gốc (baseline cùng số bước). `--scene`: bật v2; `--heavy`: v3 (kèm v2); `--v4`: v4 (kèm v3).
  Bỏ `--val-groups` để đưa cả 25 ảnh vào train cho model cuối.
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
| Bao đè bao (v4, `--v4`) | `occlude_bags` | Dán bao thật đè một phần lên bao có sẵn; bao bên dưới đổi nhãn thành hình chữ nhật bao phần còn thấy (bỏ nhãn nếu còn < 25%); pallet chứa nó nới ra theo bao mới |
| Pallet rỗng (v4) | `paste_empty_pallet` | Dán pallet không có bao (cắt từ ảnh train) vào chỗ trống, có nhãn pallet |
| Mẫu âm dày hơn (v4) | `paste_clutter` | Xác suất 0.7, tối đa 5 vật/ảnh, mảng tới 35% cạnh ảnh |
| Sáng theo lớp xếp (v5, `--v5`) | `layer_shading` trong `occlude_bags` | Đo trên ảnh thật: bao lớp dưới tối hơn bao lớp trên 5–15%, có dải bóng dọc mép tiếp giáp. Bao dưới tối 5–25%, bóng theo hướng đèn ngẫu nhiên, bao trên sáng 0–15%; 30% không đổi sáng để model học mép bao |
| Màu từng bao (v5) | `per_bag_jitter`, `mixed_recolor` | Mỗi bao lệch sắc/sáng một chút (cũ, bụi, phai); 20% ảnh có pallet trộn 2 loại bao |
| Chói (v5) | `bag_specular`, `overexpose` | Vệt bóng loáng dọc thân bao nilon; cháy sáng mạnh (×1.3–1.7, mất vân, quầng sáng) như nhóm bao đỏ |
| Bao to/nhỏ, biến dạng (v5) | `deform_patch`, `paste_bags` | Bao dán phồng/nhăn/méo nhẹ thay vì phẳng; cỡ 0.6–1.4× |

Online (Ultralytics, trong `train.py`): xoay 0–180° (camera nhìn từ trên), lật ngang/dọc,
scale 0.5–1.5 (bao to/nhỏ, camera cao/thấp), dịch 15% (pallet lệch), perspective nhẹ,
mosaic (tắt 10 epoch cuối), HSV nhẹ.

## 3. Train + đánh giá

```sh
.venv/Scripts/python code/train/train.py --data red_v4 --model yolo26n-obb.pt --imgsz 960 --name red_v4
```

Mặc định 60 epoch, imgsz 640, batch 8, seed 0 (`--seed`). Model, log và ảnh val trong
`runs/<name>/`. Chỉ số val của **`last.pt`** được nối vào `runs/results.jsonl`; số của `best.pt` ghi
kèm với tiền tố `best_` chỉ để tham khảo, vì `best.pt` được chọn theo chính tập val nên lạc quan.
`--freeze N` đóng băng N lớp đầu (đã thử, kém hơn), `--no-val` cho model cuối train trên cả 25 ảnh.

Phân tích lỗi trên val (bỏ sót, báo nhầm, box lệch theo đặc điểm nhãn: bị che, cắt biên, kích
thước, pallet mục tiêu/khác/bao rời). Mỗi `--fold` là weights rồi các nhóm val của fold đó:

```sh
.venv/Scripts/python code/train/analyze_errors.py --imgsz 960 \
    --fold runs/red_v4/weights/last.pt august_red september_pallet_4 \
    --fold runs/sep3_v4/weights/last.pt september_pallet_3 september_pallet_2
```

Xuất `runs/errors/summary.txt` (ảnh xếp theo số lỗi), `objects.csv` và `vis/` (xanh = đúng, đỏ = bỏ
sót, vàng = box lệch IoU < 0.75, cam = báo nhầm).

So phương pháp suy luận trên ảnh gốc, chấm mAP như Ultralytics val (`pipeline.py`): `plain`,
`tiled` (thêm các ô 960×960), `two_stage` (cần `--crop-model`, train trên dataset của
`build_crops.py`: crop pallet ở độ phân giải gốc, một lớp bao), `+inpallet` (chỉ giữ bao trong
pallet). Kết quả nối vào `runs/pipeline.jsonl`:

```sh
.venv/Scripts/python code/train/pipeline.py --fold red --val-groups august_red september_pallet_4 \
    --model runs/red_v4/weights/last.pt --imgsz 960 --tag v4 --methods plain plain+inpallet
```

## 4. Dự đoán ảnh chưa nhãn

```sh
.venv/Scripts/python code/train/predict.py --weights runs/<run>/weights/last.pt --imgsz 960
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

Kernel `kaggle/kernel.py` chạy 4 fold theo nhóm cảnh (phủ cả 25 ảnh), train song song trên 2 GPU.
Chọn giai đoạn bằng `--stage`; mỗi giai đoạn là một kernel riêng, kết quả tải về `runs/kaggle[_<stage>]/`:

- `compare` (mặc định): không augment / v1 / v3 / v3 ảnh 960, phân tích lỗi, model cuối `all_v3*`.
  Bảng ở `runs/summary.txt`.
- `methods`: model crop cho two-stage, biến thể `--freeze`, chấm plain/tiled/two_stage/+inpallet
  (`summary_methods.txt`). Cần `compare` chạy xong (mount output của nó).
- `fix`: v1 / v3 / v4 cùng ảnh 960, cùng 30 bản augment, 2 seed, chấm `last.pt`; model cuối
  `all_v4_960` và dự đoán ảnh chưa nhãn (`summary_fix.txt`).
- `v5`: v5 cùng điều kiện với `fix`, gộp bảng với `fix` (`summary_v5.txt`); model cuối `all_v5_960`.
  Cần `fix` chạy xong.
- `s960` / `n1280`: YOLO26s ảnh 960 / nano ảnh 1280, dataset v3, 2 seed, chấm `last.pt` và ensemble
  2 seed (`summary_<stage>.txt`). Hai stage độc lập, chạy song song trên 2 tài khoản được.

Sau khi tải về, chấm thêm `best.pt` trên máy để ghi kèm (README chính giữ cả hai số):

```sh
.venv/Scripts/python code/train/pipeline.py --fold red --val-groups august_red september_pallet_4 \
    --model runs/kaggle_s960/runs/red_s960_s0/weights/best.pt --imgsz 960 --tag best_s960_s0 --methods plain
```

## Công cụ khác

- `ensemble.py`: TTA 8 hướng xoay/lật và ensemble nhiều weights, gộp box bằng WBF, chấm như
  `pipeline.py`. `--device 0` để chạy GPU (mặc định CPU).
- `label_noise.py`: ước lượng nhiễu nhãn bằng cách so nhãn của cùng một vật trên các cặp ảnh gần như
  giống hệt nhau.

## Quy ước góc

Một hình chữ nhật có nhiều cách ghi: (w, h, 30°) = (w, h, 210°) = (h, w, 120°).
Labeler giờ lưu một dạng duy nhất (long-edge, như MMRotate `le90`): `w` là cạnh dài,
`angle` là hướng cạnh dài trong [-90°, 90°), độ, chiều kim đồng hồ trên ảnh (y hướng
xuống). Bao đối xứng 180° nên đây cũng là góc kẹp cho robot. YOLO train từ 4 đỉnh và tự
chuẩn hóa nội bộ, nên quy ước này không ảnh hưởng train; `predict.py` đổi output về cùng
quy ước.

## Đưa weights lên cho demo

App demo (`code/demo/`) chỉ suy luận, lấy weights từ Kaggle dataset `tuananhpham29/bao-tai-obb-weights`
(private, share cho tài khoản Kaggle của người chạy demo). Sau khi train được model tốt hơn: sửa
`source` (đường dẫn `last.pt` trong `runs/`) và `name` trong `code/demo/models.json`, rồi

```sh
.venv/Scripts/python code/demo/weights.py publish -a tuananhpham29
```

Lệnh tạo version mới của dataset. Commit `models.json`; người chạy demo chạy lại `weights.py download`.
