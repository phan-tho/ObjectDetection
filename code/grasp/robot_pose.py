"""Tư thế tay robot <-> ma trận 4x4, và toạ độ camera -> toạ độ robot khi camera gắn trên tay robot.

Camera gắn trên tay (eye-in-hand): lúc chụp, camera ở T_base_cam = T_base_flange · T_flange_cam.
- T_base_flange: tư thế mặt bích (flange/tool0) do bộ điều khiển robot trả về lúc chụp. Tay phải đứng
  yên khi chụp.
- T_flange_cam: cố định, đo một lần bằng hiệu chuẩn hand-eye (code/grasp/handeye.py), lưu ở
  `cam_to_flange` trong camera.json.
Camera gắn cố định (eye-to-hand) thì dùng thẳng `cam_to_robot` trong camera.json.

Kiểu tư thế (`robot_pose_format` trong camera.json), độ dài luôn là mm:
- "rpy":      x, y, z, rx, ry, rz (độ), R = Rz(rz)·Ry(ry)·Rx(rx). Fanuc W,P,R; Yaskawa Rx,Ry,Rz; nhiều robot
              Trung Quốc (Dobot, JAKA, Elite...).
- "kuka_abc": x, y, z, A, B, C (độ), R = Rz(A)·Ry(B)·Rx(C).
- "rotvec":   x, y, z, rx, ry, rz (rad, vector quay). Universal Robots (UR trả x, y, z theo mét: nhân 1000).
- "quat":     x, y, z, qw, qx, qy, qz. ABB.
- "matrix":   16 số của ma trận 4x4, theo hàng.
"""
import math

import cv2
import numpy as np

FORMATS = {"rpy": 6, "kuka_abc": 6, "rotvec": 6, "quat": 7, "matrix": 16}


def _rz_ry_rx(z, y, x):
    cz, sz, cy, sy, cx, sx = math.cos(z), math.sin(z), math.cos(y), math.sin(y), math.cos(x), math.sin(x)
    return np.array([[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                     [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                     [-sy, cy * sx, cy * cx]])


def pose_to_matrix(values, fmt):
    v = [float(x) for x in values]
    if len(v) != FORMATS[fmt]:
        raise ValueError(f"Kiểu {fmt} cần {FORMATS[fmt]} số, nhận được {len(v)}.")
    if fmt == "matrix":
        return np.array(v).reshape(4, 4)
    T = np.eye(4)
    T[:3, 3] = v[:3]
    if fmt == "rpy":
        T[:3, :3] = _rz_ry_rx(*np.radians([v[5], v[4], v[3]]))
    elif fmt == "kuka_abc":
        T[:3, :3] = _rz_ry_rx(*np.radians(v[3:6]))
    elif fmt == "rotvec":
        T[:3, :3] = cv2.Rodrigues(np.array(v[3:6]))[0]
    else:  # quat
        w, x, y, z = np.array(v[3:7]) / np.linalg.norm(v[3:7])
        T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    return T


def parse_pose(text, fmt):
    """Chuỗi số cách nhau bởi dấu phẩy / khoảng trắng -> ma trận 4x4."""
    return pose_to_matrix(text.replace(",", " ").replace(";", " ").split(), fmt)


def cam_to_base(camera_cfg, flange_pose=None):
    """Ma trận camera -> robot của lần chụp này; None nếu chưa hiệu chuẩn (chỉ có toạ độ camera).
    flange_pose: ma trận 4x4 tư thế mặt bích lúc chụp (bắt buộc khi camera gắn trên tay)."""
    if camera_cfg.get("cam_to_flange") is not None:
        if flange_pose is None:
            return None
        return np.asarray(flange_pose, np.float64) @ np.asarray(camera_cfg["cam_to_flange"], np.float64)
    if camera_cfg.get("cam_to_robot") is not None:
        return np.asarray(camera_cfg["cam_to_robot"], np.float64)
    return None
