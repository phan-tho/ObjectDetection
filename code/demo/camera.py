"""Nguồn ảnh cho trang camera trực tiếp. Mọi nguồn có cùng giao diện:

    read()        -> (ảnh BGR, depth mm hoặc None): một khung hình để xem trực tiếp
    capture(n)    -> như read() nhưng depth là median của n khung (pixel 0 bỏ), dùng để chọn bao
    camera        -> pick_xyz.Camera (nội tham số + méo ống kính), None nếu nguồn không có depth
    label         -> mô tả ngắn khung hình hiện tại

- OrbbecSource: camera của robot (Orbbec Gemini 2), ảnh màu 1920×1080 + depth khớp ảnh màu, giống
  cấu hình lúc thu dữ liệu. Cần `pip install pyorbbecsdk2` (không có trong requirements.txt, chỉ cần
  trên máy cắm camera).
- WebcamSource: webcam / camera USB qua OpenCV, chỉ có ảnh màu nên chỉ nhận diện, không tính XYZ.
- ReplaySource: phát lại một chuỗi chụp đã lưu như camera đang chạy, để thử khi không có camera.
"""
import threading
import time

import cv2
import numpy as np

from pick_xyz import Camera, load_depth, load_meta

from common import ROOT

ORBBEC_COLOR = (1920, 1080, 30)  # giống capture_profile trong meta.json của dữ liệu đã thu
WARMUP_FRAMES = 30               # bỏ các khung đầu khi auto exposure chưa ổn định


class OrbbecSource:
    has_depth = True

    def __init__(self, default_dist=None):
        try:
            from pyorbbecsdk import AlignFilter, Config, OBFormat, OBSensorType, OBStreamType, Pipeline
        except ImportError as e:
            raise RuntimeError("Chưa cài SDK của Orbbec. Chạy `.venv\\Scripts\\python -m pip install pyorbbecsdk2` "
                               "trên máy cắm camera.") from e
        self.lock = threading.Lock()
        self.pipe = Pipeline()
        cfg = Config()
        colors = self.pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color = colors.get_video_stream_profile(*ORBBEC_COLOR[:2], OBFormat.RGB, ORBBEC_COLOR[2])
        except Exception:  # camera không có đúng profile: dùng mặc định, intrinsics vẫn đọc theo profile thật
            color = colors.get_default_video_stream_profile()
        cfg.enable_stream(color)
        cfg.enable_stream(self.pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                          .get_default_video_stream_profile())
        self.pipe.enable_frame_sync()
        self.pipe.start(cfg)
        self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        self.camera = Camera(*self._intrinsics(color, default_dist))
        for _ in range(WARMUP_FRAMES):
            self.pipe.wait_for_frames(200)
        self.label = "Orbbec"

    def _intrinsics(self, profile, default_dist):
        names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
        try:
            i, d = profile.get_intrinsic(), profile.get_distortion()
        except AttributeError:  # SDK cũ: lấy từ camera param của pipeline
            p = self.pipe.get_camera_param()
            i, d = p.rgb_intrinsic, p.rgb_distortion
        k = {"fx": i.fx, "fy": i.fy, "cx": i.cx, "cy": i.cy}
        dist = {n: float(getattr(d, n)) for n in names} if d is not None else default_dist
        return k, dist

    @staticmethod
    def _to_bgr(frame):
        from pyorbbecsdk import OBFormat
        w, h, fmt = frame.get_width(), frame.get_height(), frame.get_format()
        data = np.asanyarray(frame.get_data())
        if fmt == OBFormat.RGB:
            return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if fmt == OBFormat.BGR:
            return data.reshape(h, w, 3).copy()
        if fmt == OBFormat.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == OBFormat.YUYV:
            return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
        raise RuntimeError(f"Định dạng ảnh màu chưa hỗ trợ: {fmt}")

    def _frame(self):
        for _ in range(10):
            fs = self.pipe.wait_for_frames(1000)
            fs = self.align.process(fs) if fs is not None else None
            fs = fs.as_frame_set() if fs is not None else None
            if fs is None:
                continue
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            if c is None or d is None:
                continue
            depth = np.asanyarray(d.get_data()).view(np.uint16).reshape(d.get_height(), d.get_width())
            return self._to_bgr(c), depth.astype(np.float32) * d.get_depth_scale()
        raise RuntimeError("Không nhận được khung hình từ camera (kiểm tra cáp USB 3).")

    def read(self):
        with self.lock:
            self.label = f"Orbbec · {time.strftime('%H:%M:%S')}"
            return self._frame()

    def capture(self, n=5):
        """Ảnh màu của khung đầu, depth = median các khung khác 0 (như công cụ thu dữ liệu)."""
        with self.lock:
            frames = [self._frame() for _ in range(n)]
        stack = np.stack([d for _, d in frames])
        stack[stack == 0] = np.nan
        with np.errstate(all="ignore"):
            depth = np.nan_to_num(np.nanmedian(stack, axis=0), nan=0.0)
        self.label = f"Orbbec · {time.strftime('%H:%M:%S')} · median {n} khung"
        return frames[0][0], depth

    def close(self):
        self.pipe.stop()


class WebcamSource:
    has_depth = False
    camera = None

    def __init__(self, index=0):
        self.lock = threading.Lock()
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được webcam số {index}.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.label = f"Webcam {index}"

    def read(self):
        with self.lock:
            ok, img = self.cap.read()
        if not ok:
            raise RuntimeError("Webcam không trả về khung hình.")
        return img, None

    def capture(self, n=5):
        return self.read()

    def close(self):
        self.cap.release()


class ReplaySource:
    """Phát lại các lượt chụp của một chuỗi theo thời gian, mỗi lượt giữ `period` giây rồi sang lượt sau
    (hết chuỗi thì quay lại đầu). Depth là bản đã gộp nhiều frame lúc thu."""
    has_depth = True

    def __init__(self, shots, period=2.0, default_dist=None):
        self.shots, self.period, self.default_dist = shots, period, default_dist
        self.lock = threading.Lock()
        self.start = time.monotonic()
        self.loaded = None  # (index, ảnh, depth, Camera)
        self.camera, self.label = None, "Phát lại"

    def read(self):
        """camera và label được cập nhật theo đúng lượt vừa trả về."""
        with self.lock:
            i = int((time.monotonic() - self.start) / self.period) % len(self.shots)
            if self.loaded is None or self.loaded[0] != i:
                s = self.shots[i]
                k, dist, _ = load_meta(s["image_path"])
                self.loaded = (i, cv2.imread(str(ROOT / s["image_path"])), load_depth(s["depth_path"]),
                               Camera(k, dist or self.default_dist))
            _, img, depth, self.camera = self.loaded
            self.label = f"Phát lại lượt {i + 1}/{len(self.shots)} · {self.shots[i]['id']}"
        return img, depth

    def capture(self, n=5):
        return self.read()

    def close(self):
        pass
