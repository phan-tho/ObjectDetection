# Chọn bao cần gắp và tính XYZ

Rule-based, không cần train. Đầu vào: box OBB (của model hoặc nhãn tay) và `depth.png` của cùng lượt
chụp. Chạy trên CPU, vài giây mỗi ảnh.

```sh
.venv/Scripts/python code/grasp/pick_xyz.py --predictions runs/predict_<run>/predictions.json
.venv/Scripts/python code/grasp/pick_xyz.py --labels      # dùng nhãn tay, để kiểm tra luật
```

Kết quả trong `runs/grasp/<tên>/`:
- `picks.json`: mỗi ảnh có danh sách bao, `rank` 1 là bao gắp trước, `rank` = null kèm `reasons` là lý
  do chưa gắp được. Mỗi bao có `xyz_cam_mm`, `yaw_cam_deg`, `normal`, `tilt_deg`,
  `below_top_mm`, `size_mm`.
- `vis/*.jpg`: khung xanh lá = gắp trước, vàng = gắp được, đỏ = chưa gắp được.

Toạ độ camera theo quy ước OpenCV: X sang phải, Y xuống dưới, Z hướng ra trước, đơn vị mm. Muốn ra
toạ độ robot, điền ma trận 4×4 `cam_to_robot` (từ hiệu chuẩn hand-eye) vào `camera.json`.

## Các bước

1. **Pallet mục tiêu**: box pallet gần tâm ảnh nhất. Bao của nó là bao có tâm nằm trong box pallet.
2. **Mặt trên mỗi bao**: lấy vùng lõi 60% của box, đổi pixel depth sang điểm 3D, khớp mặt phẳng có loại
   điểm lệch. Điểm gắp là giao của tia qua tâm box với mặt phẳng này. Hướng kẹp là cạnh dài của box.
3. **Độ cao** đo theo hướng "lên" của chồng bao (trục Z robot nếu có hiệu chuẩn, không thì lấy hướng
   pháp tuyến trung vị của các mặt bao), vì camera có thể nghiêng tới 15°.
4. **Lớp trên cùng**: bao thấp hơn bao cao nhất không quá 65 mm (nửa độ dày lớp 130 mm).
5. **Bị đè**: xét vùng hai box chạm nhau. Nếu depth đo thật ở đó cao hơn mặt bao này quá 25 mm (và bao
   kia ở phía cao hơn) thì bao này đang bị đè. Không so độ cao trung bình, vì bao đè thường nằm chéo.
   Cũng không kéo dài mặt phẳng của bao kia ra chỗ tiếp xúc, vì bao nằm nghiêng hoặc mép bao phồng
   làm phép kéo dài sai.
6. **Loại**: thiếu depth, mặt nghiêng quá 35°, mặt gồ ghề, thấy được dưới 60% chiều dài bao.
7. **Thứ tự**: cao nhất trước; cùng độ cao thì bao thấy được nhiều hơn, conf cao hơn trước.

## Depth có nhiễu không, cần xử lý gì

Đo trên dữ liệu (Orbbec Gemini 2, depth đã khớp ảnh màu):

| Đại lượng | Giá trị |
|---|---|
| Pixel không đo được (giá trị 0) trong vùng pallet | < 1% |
| Nhiễu theo thời gian của một pixel | 4–19 mm, tăng theo khoảng cách |
| Độ lệch so với mặt phẳng trong một bao (gồm cả độ phồng của bao) | 6–22 mm |
| Chênh độ cao giữa hai lớp bao | khoảng 120–130 mm |
| Kích thước bao đo từ depth (80 ảnh) | trung bình 685×443 mm (thật: 700×450) |
| Méo ống kính | ~6 px ở góc ảnh, < 2 px ở vùng pallet |

Nhiễu đủ nhỏ so với độ dày một lớp bao, nhưng không được dùng giá trị của một pixel lẻ. Code đã làm:
- Gộp nhiều frame bằng median, bỏ pixel 0. Dữ liệu 2026-09-11 đã gộp sẵn 5 frame.
- Chỉ dùng vùng lõi của box, tránh mép bao và chỗ depth lệch với ảnh màu.
- Khớp mặt phẳng có loại điểm lệch trên hàng nghìn điểm, thay vì đọc depth ở tâm.
- Sửa méo ống kính (hệ số trong `camera.json`).

Không cần lọc ảnh depth (bilateral, lấp lỗ...).

Kiểm bằng nhãn tay trên 25 ảnh:
- 23 ảnh có bao đều chọn được bao để gắp, và bao gắp trước không bao giờ là bao gắn cờ "bị che".
- 73 bao được xếp hạng gắp được. 3 bao gắn cờ "bị che" lọt vào danh sách, đều ở ảnh `143342`. Đó là bao
  cùng lớp trên cùng, chỉ bị mép bao bên cạnh gối lên, và được xếp sau các bao không bị che (thứ 3–5).
- Hai ảnh pallet rỗng được báo đúng.

Trên dự đoán của model (80 ảnh chưa nhãn): 76 ảnh chọn được bao, 4 ảnh pallet rỗng.

## Cần làm thêm trước khi chạy trên robot

- **Hiệu chuẩn hand-eye** để có `cam_to_robot`. Khi đó độ cao cũng đo theo trục Z robot, chính xác hơn.
- **Kiểm sai số tuyệt đối của Z** bằng vật có độ cao biết trước. Camera stereo có thể lệch vài cm ở
  khoảng 2–3 m. Nên để robot hạ xuống có cảm biến (giác hút có cảm biến áp, hoặc lực) thay vì tin tuyệt
  đối vào Z.
- **Gắp xong một bao thì chụp lại** rồi chọn bao tiếp, vì cảnh thay đổi sau mỗi lần gắp.
- Thứ tự gắp giữa các bao cùng độ cao hiện chưa tính vị trí robot. Nếu cần gắp từ phía gần robot trước,
  thêm điều kiện đó vào bước 7.
