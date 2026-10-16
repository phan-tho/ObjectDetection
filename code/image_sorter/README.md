# Phân loại ảnh pallet

Chạy local bằng Python 3.9 trở lên, không cần cài thư viện. Ảnh gốc được giữ nguyên.

Từ thư mục dự án:

```sh
python3 code/image_sorter/app.py
```

Trình duyệt tự mở tại http://127.0.0.1:8765. Nếu không tự mở, truy cập địa chỉ này.
Nhấn `Ctrl+C` trong Terminal để dừng. Nếu cổng bận, dùng `--port 8766`.

## Thao tác

| Phím | Ý nghĩa |
|---|---|
| `1` | Đầy: không hở đáy pallet, không xét chiều cao |
| `2` | Vơi: hở đáy, còn trên 3 bao |
| `3` | Chỉ còn 1–3 bao; nhãn hở đáy chọn riêng nếu thấy |
| `4` | Pallet rỗng |
| `Q W E R T Y U I` | Bật/tắt các đặc điểm, như hiển thị trên giao diện |
| `D` / `X` / `C` | Dùng được / Loại / Cần xem lại |
| `Enter` hoặc `→` | Lưu và sang lượt tiếp theo |
| `Space` hoặc `←` | Quay lại; lưu thay đổi hợp lệ trước khi chuyển |
| `[` / `]` | Xem frame trước/sau trong cùng lượt |

Bấm ảnh để xem ở kích thước gốc, bấm lại để thu vừa khung. Ghi chú không bắt buộc.
Phím tắt không hoạt động khi đang gõ ghi chú hoặc chọn lý do loại.

Chọn một trạng thái pallet và nhiều đặc điểm nếu cần. Với 1–3 bao, ưu tiên chọn `3`.
Chọn `1` tự bỏ nhãn hở đáy; `2` và `4` tự thêm nhãn này. Ánh sáng và màu bao
"khác thường" cần được hiểu so với điều kiện vận hành thông thường của bạn;
ghi rõ màu/điều kiện trong ghi chú nếu cần.

Ảnh không đủ toàn bộ vùng pallet mục tiêu: chọn `X` và lý do tương ứng.
Ảnh khó nhưng vẫn phù hợp điều kiện vận hành vẫn có thể dùng.
Nếu không xác định được trạng thái, chọn `C`; không bắt buộc chọn trạng thái pallet.

Thay đổi hợp lệ tự lưu sau 0,4 giây; `Enter` lưu ngay. Đợi thông báo **Đã lưu**
trước khi đóng. Nếu đóng khi còn thay đổi chưa lưu, trình duyệt sẽ cảnh báo.
Mở lại bắt đầu ở lượt chưa phân loại đầu tiên. Dùng `Space` để sửa lượt trước.

## Dữ liệu và kết quả

Chương trình đọc hai thư mục `bao-tai/` và `2026-09-11/`
ở gốc dự án. Mỗi lượt chụp là một mục; nhãn áp dụng cho cả lượt.
Các frame khác chỉ để kiểm tra, không đổi ảnh đại diện. Chưa có trình xem depth.
Không tự suy ra nhãn "đầy/vơi" từ metadata cũ.

Kết quả nằm trong `code/image_sorter/results/`:

- `labels.json`: kết quả gốc để chương trình đọc lại, kèm định nghĩa nhãn.
- `labels.csv`: một dòng mỗi lượt chụp; cột nhãn là `0/1`; lượt chưa phân loại
  có `status=unreviewed`. CSV dùng UTF-8 BOM để mở tiếng Việt trong Excel.

CSV lưu đường dẫn tương đối tới ảnh RGB và depth đại diện, số frame, metadata
và `sequence_group_hint` nếu có trong manifest của bộ nhãn cũ. Đây là gợi ý
nhóm chuỗi chụp cho việc chia tập sau này, cần kiểm tra khi dùng.
Không chia train/validation/test tự động trong bước này.

Mục **Thống kê & file kết quả** đếm theo lượt chụp; các trạng thái/đặc điểm
chỉ đếm lượt **dùng được**. Một lượt có nhiều đặc điểm nên tổng số đặc điểm
có thể lớn hơn số lượt.

Chỉ chạy một bản chương trình với cùng thư mục kết quả để tránh ghi đè nhau.
Tham số `--output /đường/dẫn/khác` và `--no-browser` dành cho khi cần đổi nơi
lưu hoặc không muốn tự mở trình duyệt.
