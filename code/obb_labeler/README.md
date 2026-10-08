# Gán nhãn OBB cho 25 ảnh mẫu

Python 3.9 trở lên, không cần cài thư viện. Chạy từ thư mục dự án:

```sh
python3 code/obb_labeler/app.py
```

Trình duyệt mở tại http://127.0.0.1:8766. Nên dùng cửa sổ trình duyệt rộng để
ảnh và bảng nhãn hiển thị cạnh nhau. Dừng bằng `Ctrl+C` trong Terminal.
Cổng bận: thêm `--port 8767`. Không tự mở trình duyệt: thêm `--no-browser`.

## Bộ ảnh

`selection.json` và `selection.csv` lưu 25 ảnh đã chọn cùng lý do, đường dẫn
ảnh RGB/depth và nhóm chuỗi chụp: 18 đầy, 2 vơi, 3 ít bao, 2 rỗng.
Mỗi lượt chỉ dùng ảnh RGB đại diện, 1920×1080, giữ nguyên dữ liệu gốc.
Các ảnh cùng nhóm chuỗi không phải các cảnh độc lập khi chia tập sau này.
Nhóm tháng 8 được gom bảo thủ dựa trên cảnh đã xem, cần kiểm tra khi chia tập.

## Vẽ và sửa

1. Bấm `1` (Pallet), khoanh pallet mục tiêu trước. Vùng này gồm đế pallet và
   mọi bao của nó, kể cả bao tràn ra ngoài. Pallet đầu tiên mặc định là mục tiêu;
   có thể đổi bằng checkbox **Pallet mục tiêu**. Chỉ một pallet có cờ này.
2. Một OBB được vẽ bằng 3 lần bấm: hai đầu của một cạnh, rồi một điểm trên cạnh
   đối diện. Hai cạnh còn lại tự vuông góc, nên luôn là hình chữ nhật.
3. Bấm `2` (Bao) và khoanh từng thân bao, bỏ tai/mép gấp nhỏ. Mỗi bao tự liên kết
   với pallet đang chọn trong **Bao mới thuộc pallet**. Sửa liên kết nếu cần.
4. Bấm `V` để chọn/sửa. Kéo trong box để di chuyển; kéo góc để đổi kích thước;
   kéo nút tròn ngoài box để xoay. Có thể nhập góc ở bảng bên phải.
5. Khoanh những pallet/bao khác rõ ràng trong ảnh. Bao ngoài pallet hoặc chưa
   rõ thuộc pallet nào có thể chọn **Ngoài pallet / chưa xác định**.
6. Bấm **Đã kiểm tra & tiếp** (`Enter`) sau khi kiểm tra hết ảnh.

| Phím/thao tác | Ý nghĩa |
|---|---|
| `1`, `2` | Chế độ vẽ pallet / bao |
| `V`, `H` | Chọn/sửa box / kéo di chuyển ảnh |
| `Esc` | Hủy nét đang vẽ |
| `Delete` hoặc `Backspace` | Xóa box đang chọn |
| `Ctrl+Z` hoặc `Cmd+Z` | Hoàn tác trong ảnh hiện tại |
| `[` / `]` | Xoay box 1°, giữ `Shift` để xoay 10° |
| Các phím mũi tên | Khi chọn box: di chuyển 1 px, `Shift` = 10 px |
| `Space` | Lưu nháp và về ảnh trước |
| `→` khi không chọn box | Lưu nháp và sang ảnh sau |
| Lăn chuột | Zoom vào/ra; tối đa 8 lần |
| **Vừa ảnh** | Trở về toàn bộ ảnh |

Phím tắt không tác động khi đang nhập ghi chú/góc hoặc chọn dropdown.
Nét vẽ chưa đủ 3 điểm cần hoàn tất hoặc hủy trước khi lưu/chuyển ảnh.
Box có thể lưu nháp khi nằm ngoài biên; phải chỉnh các góc về trong ảnh trước
khi xác nhận để đảm bảo định dạng OBB chuẩn hóa 0–1.

Bao bị che nhưng vẫn phân biệt được: khoanh phần thân nhìn thấy, bật **Bị che
khuất**. Không phỏng đoán phần hoàn toàn bị che. Đối tượng bị cắt bởi biên
ảnh: bật **Bị cắt bởi biên ảnh**, kiểm tra box trước khi xuất. Các cờ này lưu
trong JSON, không tự tạo vùng ignore trong YOLO. Nếu còn đối tượng mơ hồ/chưa
khoanh xong, lưu nháp và ghi chú; chưa xác nhận ảnh để train.

Pallet rỗng vẫn cần box pallet mục tiêu; có thể không có bao trên pallet đó.
OBB mô tả vị trí/hướng đối tượng; không phải nhãn điểm gắp hoặc nhãn XYZ.

## Lưu và xuất

Thay đổi tự lưu nháp sau 0,5 giây. Đợi thông báo **Đã lưu nháp** trước khi đóng.
Mở lại sẽ bắt đầu ở ảnh chưa xác nhận đầu tiên. Mỗi ảnh có trạng thái nháp/
đã kiểm tra riêng. Sửa một ảnh đã kiểm tra sẽ đưa ảnh về nháp và bỏ bản xuất
YOLO dẫn xuất cho đến khi bạn xác nhận lại.

Kết quả trong `code/obb_labeler/results.nosync/`:

- `annotations.json`: bản gốc, nháp và nhãn đã kiểm tra; lưu tâm, kích thước,
  góc, bốn góc theo pixel gốc, lớp, liên kết pallet, cờ mục tiêu và che khuất.
- `manifest.csv`: đường dẫn RGB/depth, kích thước, nhóm chuỗi, số đối tượng,
  trạng thái đã kiểm tra và ghi chú.
- `yolo/labels/<id>.txt`: chỉ ảnh đã kiểm tra; mỗi dòng
  `class_id x1 y1 x2 y2 x3 y3 x4 y4`, tọa độ chuẩn hóa theo ảnh gốc.
- `yolo/images/<id>.png`: symlink tới RGB gốc tương ứng, không nhân bản ảnh.
- `yolo/classes.yaml`: `0: pallet_region`, `1: bag`.
- `yolo/reviewed.txt`: danh sách ảnh đã kiểm tra để chọn dữ liệu train sau này.

Đây là bộ seed để fine-tune; chưa chia train/validation/test hoặc augment.
Không dùng seed này làm test độc lập. Khi train sẽ chọn tập validation theo
nhóm cảnh và tạo `dataset.yaml` từ manifest; không dùng train làm validation.

Chỉ chạy một bản app ghi cùng thư mục kết quả. Có thể đổi nơi lưu bằng
`--output /đường/dẫn/khác`. Nhãn cũ và file phân loại không bị thay đổi.
