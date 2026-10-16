"""Hiệu chuẩn hand-eye cho camera gắn trên tay robot: tìm T_flange_cam (`cam_to_flange` trong camera.json).

    .venv/Scripts/python code/grasp/handeye.py board                  # bảng ChArUco để in -> calib/charuco_board.png
    .venv/Scripts/python code/grasp/handeye.py solve calib/handeye    # giải từ các lần chụp, ghi camera.json
    .venv/Scripts/python code/grasp/handeye.py selftest               # kiểm code bằng ảnh giả lập

Quy trình:
1. In bảng (A3, in đúng 100%, không co giãn), dán phẳng lên tấm cứng, đặt cố định trên bàn/sàn trong tầm
   camera. Đo lại cạnh ô vuông bằng thước; khác thì sửa `charuco.square_mm` (và `marker_mm` theo cùng tỷ
   lệ) trong camera.json.
2. Đưa tay robot tới 15–25 tư thế nhìn thấy bảng, đổi cả vị trí lẫn hướng (nghiêng 15–30° theo nhiều
   trục, xoay quanh trục camera). Mỗi tư thế: dừng tay, lưu ảnh + tư thế mặt bích (trang Camera trực tiếp >
   Hiệu chuẩn hand-eye). Bảng không được xê dịch trong suốt quá trình.
3. Chạy solve. Tốt khi vị trí bảng tính trong hệ robot từ các lần chụp lệch nhau vài mm.

Mỗi lần chụp là một thư mục con: color.png, pose.json {"pose": [...], "format": "rpy"},
intrinsics.json {"A": ma trận 3x3, "D": 8 hệ số méo hoặc null}.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from robot_pose import pose_to_matrix

ROOT = Path(__file__).resolve().parents[2]
CAMERA_JSON = Path(__file__).resolve().parent / "camera.json"
MIN_CORNERS = 8


def make_board(cfg):
    c = cfg["charuco"]
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, c["dict"]))
    return cv2.aruco.CharucoBoard(tuple(c["squares"]), float(c["square_mm"]), float(c["marker_mm"]), dictionary)


def to_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def board_pose(img, board, A, D):
    """T_cam_board (mm) từ một ảnh, số góc thấy được, sai số chiếu lại (px); None nếu không đủ góc."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    if ids is None or len(ids) < MIN_CORNERS or board.checkCharucoCornersCollinear(ids):
        return None
    obj, pts = board.matchImagePoints(corners, ids)
    D = None if D is None else np.asarray(D, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, pts, A, D)
    if not ok:
        return None
    proj = cv2.projectPoints(obj, rvec, tvec, A, D)[0]
    err = float(np.sqrt(np.mean(np.sum((proj - pts) ** 2, axis=-1))))
    return to_T(cv2.Rodrigues(rvec)[0], tvec), len(ids), err


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def solve(flange, board_in_cam):
    """Park & Martin (1994) trên mọi cặp lần chụp: A·X = X·B, A = chuyển động mặt bích, B = chuyển động
    camera. (Bản OpenCV 5 dùng ở đây không có cv2.calibrateHandEye.)
    flange: list T_base_flange; board_in_cam: list T_cam_board. Trả về X = T_flange_cam và độ ổn định của
    vị trí bảng tính trong hệ robot (bảng đứng yên nên mọi lần chụp phải ra cùng một chỗ)."""
    pairs = [(np.linalg.inv(flange[i]) @ flange[j], board_in_cam[i] @ np.linalg.inv(board_in_cam[j]))
             for i in range(len(flange)) for j in range(i + 1, len(flange))]
    M = sum(np.outer(cv2.Rodrigues(B[:3, :3])[0].ravel(), cv2.Rodrigues(A[:3, :3])[0].ravel()) for A, B in pairs)
    w, V = np.linalg.eigh(M.T @ M)
    R = V @ np.diag(1 / np.sqrt(w)) @ V.T @ M.T
    lhs = np.vstack([A[:3, :3] - np.eye(3) for A, _ in pairs])
    rhs = np.concatenate([R @ B[:3, 3] - A[:3, 3] for A, B in pairs])
    X = to_T(R, np.linalg.lstsq(lhs, rhs, rcond=None)[0])
    boards = [F @ X @ B for F, B in zip(flange, board_in_cam)]
    mean_t = np.mean([b[:3, 3] for b in boards], axis=0)
    dt = [float(np.linalg.norm(b[:3, 3] - mean_t)) for b in boards]
    dr = [rot_deg(boards[0][:3, :3].T @ b[:3, :3]) for b in boards]
    return X, {"pos_mm_median": float(np.median(dt)), "pos_mm_max": max(dt), "rot_deg_max": max(dr),
               "per_view_mm": dt}


def write_camera_json(X):
    """Chỉ thay dòng cam_to_flange, giữ nguyên định dạng còn lại của file."""
    text = CAMERA_JSON.read_text(encoding="utf-8")
    value = json.dumps(np.round(X, 4).tolist())
    text, n = re.subn(r'"cam_to_flange":\s*[^\n]*?(,?)\n', lambda m: f'"cam_to_flange": {value}{m.group(1)}\n', text,
                      count=1)
    if n != 1:
        raise SystemExit("Không thấy khoá cam_to_flange trong camera.json.")
    CAMERA_JSON.write_text(text, encoding="utf-8")


def cmd_board(cfg, out, dpi=300):
    c = cfg["charuco"]
    ppm = dpi / 25.4
    w_mm, h_mm = c["squares"][0] * c["square_mm"], c["squares"][1] * c["square_mm"]
    img = make_board(cfg).generateImage((round(w_mm * ppm), round(h_mm * ppm)), marginSize=0)
    page = np.full((round(297 * ppm), round(420 * ppm)), 255, np.uint8)  # A3 ngang
    y0, x0 = (page.shape[0] - img.shape[0]) // 2, (page.shape[1] - img.shape[1]) // 2
    page[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    note = (f"ChArUco {c['squares'][0]}x{c['squares'][1]}, o vuong {c['square_mm']} mm, marker {c['marker_mm']} mm, "
            f"{c['dict']}. In A3 dung 100%, do lai o vuong.")
    cv2.putText(page, note, (x0, page.shape[0] - round(8 * ppm)), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 0, 4)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), page)
    print(f"-> {out} (A3 ngang, {dpi} dpi; bảng {w_mm:.0f}×{h_mm:.0f} mm)")


def load_captures(folder, board):
    flange, boards, names = [], [], []
    for d in sorted(p for p in folder.iterdir() if (p / "color.png").is_file()):
        pose = json.loads((d / "pose.json").read_text(encoding="utf-8"))
        intr = json.loads((d / "intrinsics.json").read_text(encoding="utf-8"))
        res = board_pose(cv2.imread(str(d / "color.png")), board, np.array(intr["A"], np.float64), intr.get("D"))
        if res is None:
            print(f"  {d.name}: không thấy đủ bảng, bỏ qua")
            continue
        T, n, err = res
        print(f"  {d.name}: {n} góc, sai số chiếu lại {err:.2f} px, cách bảng {np.linalg.norm(T[:3, 3]):.0f} mm")
        flange.append(pose_to_matrix(pose["pose"], pose["format"]))
        boards.append(T)
        names.append(d.name)
    return flange, boards, names


def report(flange, X, stats, names=None):
    spread = max(rot_deg(a[:3, :3].T @ b[:3, :3]) for a in flange for b in flange)
    if len(flange) < 10:
        print(f"Cảnh báo: mới có {len(flange)} lần chụp, nên 15–25.")
    if spread < 20:
        print(f"Cảnh báo: hướng tay chỉ khác nhau tối đa {spread:.0f}°; nên nghiêng/xoay nhiều hơn (> 30°).")
    print(f"Vị trí bảng trong hệ robot giữa các lần chụp lệch: median {stats['pos_mm_median']:.1f} mm, "
          f"max {stats['pos_mm_max']:.1f} mm; hướng lệch max {stats['rot_deg_max']:.2f}°")
    if names:
        worst = int(np.argmax(stats["per_view_mm"]))
        print(f"Lần chụp lệch nhất: {names[worst]} ({stats['per_view_mm'][worst]:.1f} mm), lệch nhiều thì xem lại "
              "tư thế đã nhập hoặc chụp lại.")
    print(f"Camera so với mặt bích: dịch {np.round(X[:3, 3], 1).tolist()} mm, xoay {rot_deg(X[:3, :3]):.1f}°")


def cmd_selftest(cfg, n=20, seed=0):
    """Ảnh giả lập: bảng nằm trên bàn, tay robot tới n tư thế ngẫu nhiên, camera lệch mặt bích một phép biến
    đổi biết trước. Giải lại và so với đáp án."""
    rng = np.random.default_rng(seed)
    board = make_board(cfg)
    c = cfg["charuco"]
    A = np.array([[1035.27, 0, 965.67], [0, 1035.94, 539.20], [0, 0, 1]])
    ppm = 4.0
    w_mm, h_mm = c["squares"][0] * c["square_mm"], c["squares"][1] * c["square_mm"]
    tex = board.generateImage((round(w_mm * ppm), round(h_mm * ppm)), marginSize=0)
    base_board = to_T(np.diag([1.0, -1.0, -1.0]), [600, -100, 0])  # mặt bảng hướng lên, camera nhìn xuống
    X_true = to_T(cv2.Rodrigues(np.radians([2.0, -3.0, 91.0]))[0], [45.0, -80.0, 120.0])
    center = base_board @ [w_mm / 2, h_mm / 2, 0, 1]
    flange, boards = [], []
    for _ in range(n):
        target = center[:3] + rng.uniform(-60, 60, 3) * [1, 1, 0]
        eye = center[:3] + [*rng.uniform(-350, 350, 2), rng.uniform(550, 950)]
        z = (target - eye) / np.linalg.norm(target - eye)
        y = np.cross(z, [1.0, 0, 0])
        y /= np.linalg.norm(y)
        R = np.c_[np.cross(y, z), y, z] @ cv2.Rodrigues(np.array([0, 0, rng.uniform(-np.pi, np.pi)]))[0]
        base_cam = to_T(R, eye)
        cam_board = np.linalg.inv(base_cam) @ base_board
        H = A @ np.c_[cam_board[:3, 0], cam_board[:3, 1], cam_board[:3, 3]] @ np.diag([1 / ppm, 1 / ppm, 1])
        img = cv2.warpPerspective(tex, H, (1920, 1080), borderValue=128)
        img = np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
        res = board_pose(img, board, A, None)
        if res is None:
            continue
        noise = to_T(cv2.Rodrigues(np.radians(rng.normal(0, 0.02, 3)))[0], rng.normal(0, 0.1, 3))
        flange.append(base_cam @ np.linalg.inv(X_true) @ noise)  # robot báo tư thế có sai số nhỏ
        boards.append(res[0])
    print(f"Thấy bảng ở {len(boards)}/{n} ảnh giả lập.")
    X, stats = solve(flange, boards)
    report(flange, X, stats)
    dt = np.linalg.norm(X[:3, 3] - X_true[:3, 3])
    dr = rot_deg(X[:3, :3].T @ X_true[:3, :3])
    print(f"So với đáp án: lệch {dt:.2f} mm, {dr:.3f}°")
    return dt, dr


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("board", help="tạo ảnh bảng ChArUco để in")
    b.add_argument("--out", type=Path, default=ROOT / "calib/charuco_board.png")
    s = sub.add_parser("solve", help="giải hand-eye từ thư mục các lần chụp")
    s.add_argument("folder", type=Path)
    s.add_argument("--dry-run", action="store_true", help="chỉ in kết quả, không ghi camera.json")
    sub.add_parser("selftest", help="kiểm code bằng ảnh giả lập")
    args = ap.parse_args()
    cfg = json.loads(CAMERA_JSON.read_text(encoding="utf-8"))
    if args.cmd == "board":
        cmd_board(cfg, args.out)
    elif args.cmd == "selftest":
        cmd_selftest(cfg)
    else:
        flange, boards, names = load_captures(args.folder, make_board(cfg))
        if len(flange) < 3:
            raise SystemExit("Cần ít nhất 3 lần chụp thấy bảng (nên 15–25).")
        X, stats = solve(flange, boards)
        report(flange, X, stats, names)
        if not args.dry_run:
            write_camera_json(X)
            print(f"Đã ghi cam_to_flange vào {CAMERA_JSON}")


if __name__ == "__main__":
    main()
