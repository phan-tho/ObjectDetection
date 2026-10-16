#!/usr/bin/env python3
"""Gán nhãn hình chữ nhật xoay trên ảnh RGB gốc; không cần thư viện ngoài."""

import argparse
import csv
import errno
import io
import json
import math
import os
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CLASSES = {"0": "pallet_region", "1": "bag"}


def canonical(box):
    """Một hình chữ nhật có nhiều cách ghi (30° = 210°, hoặc 120° với w/h đổi chỗ).
    Quy ước duy nhất (long-edge, như MMRotate le90): w là cạnh dài, góc của cạnh dài
    trong [-90°, 90°), đo theo chiều kim đồng hồ trên ảnh (trục y hướng xuống)."""
    w, h, angle = box["w"], box["h"], box["angle"]
    if w < h:
        w, h, angle = h, w, angle + 90
    angle = (angle + 90) % 180 - 90
    return {**box, "w": w, "h": h, "angle": round(angle, 6)}


def corners(box):
    angle = math.radians(box["angle"])
    c, s = math.cos(angle), math.sin(angle)
    return [[box["cx"] + c * x - s * y, box["cy"] + s * x + c * y]
            for x, y in [(-box["w"] / 2, -box["h"] / 2), (box["w"] / 2, -box["h"] / 2),
                         (box["w"] / 2, box["h"] / 2), (-box["w"] / 2, box["h"] / 2)]]


def validate(body, image):
    objects = body.get("objects", [])
    reviewed = body.get("reviewed", False)
    note = body.get("note", "")
    if not isinstance(objects, list) or len(objects) > 300 or type(reviewed) is not bool:
        raise ValueError("Danh sách đối tượng không hợp lệ.")
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("Ghi chú tối đa 2000 ký tự.")
    clean, ids = [], set()
    for box in objects:
        if not isinstance(box, dict) or type(box.get("id")) is not int or box["id"] <= 0 or box["id"] in ids:
            raise ValueError("ID đối tượng phải duy nhất.")
        if type(box.get("class_id")) is not int or box["class_id"] not in (0, 1):
            raise ValueError("Chỉ có hai lớp pallet và bao.")
        geometry = {}
        for key in ("cx", "cy", "w", "h", "angle"):
            value = box.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 100000:
                raise ValueError("Tọa độ không hợp lệ.")
            geometry[key] = float(value)
        if min(geometry["w"], geometry["h"]) < 2:
            raise ValueError("Hình chữ nhật quá nhỏ.")
        parent = box.get("parent_id")
        if parent is not None and type(parent) is not int:
            raise ValueError("Pallet liên kết không hợp lệ.")
        ids.add(box["id"])
        item = {"id": box["id"], "class_id": box["class_id"], **canonical(geometry),
                "parent_id": parent if box["class_id"] == 1 else None,
                "target": bool(box.get("target", False)) if box["class_id"] == 0 else False,
                "occluded": bool(box.get("occluded", False)), "truncated": bool(box.get("truncated", False))}
        item["corners"] = corners(item)
        clean.append(item)
    pallets = {box["id"] for box in clean if box["class_id"] == 0}
    if any(box["parent_id"] is not None and box["parent_id"] not in pallets for box in clean):
        raise ValueError("Bao đang liên kết tới pallet không tồn tại.")
    targets = [box for box in clean if box["target"]]
    if len(targets) > 1:
        raise ValueError("Chỉ chọn một pallet mục tiêu.")
    if reviewed:
        if len(targets) != 1:
            raise ValueError("Cần khoanh và chọn đúng một pallet mục tiêu trước khi xác nhận.")
        for box in clean:
            if any(x < -0.001 or x > image["width"] + 0.001 or y < -0.001 or y > image["height"] + 0.001
                   for x, y in box["corners"]):
                raise ValueError(f"Đối tượng #{box['id']} có góc nằm ngoài ảnh. Chỉnh lại trước khi xác nhận.")
    return {"objects": clean, "reviewed": reviewed, "note": note,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    images = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))["images"]
    by_id = {image["id"]: image for image in images}
    allowed_paths = {image["image_path"] for image in images}
    for path in allowed_paths:
        if not (ROOT / path).is_file():
            parser.error(f"Không tìm thấy ảnh gốc: {path}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "annotations.json"
    records = json.loads(json_path.read_text(encoding="utf-8"))["records"] if json_path.exists() else {}
    # File JSON là bản gốc; TXT và manifest luôn có thể dựng lại từ nó.
    for image_id, record in list(records.items()):
        if image_id in by_id:
            records[image_id] = {**record, **{k: v for k, v in validate(record, by_id[image_id]).items() if k != "updated_at"}}
    lock = threading.Lock()
    yolo = output / "yolo"
    for folder in (yolo / "images", yolo / "labels"):
        folder.mkdir(parents=True, exist_ok=True)

    def write_results():
        document = {"schema_version": 1, "classes": CLASSES, "images": images, "records": records}
        temporary = json_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(json_path)
        buffer = io.StringIO(newline="")
        fields = ["image_id", "image_path", "depth_path", "width", "height", "sequence_group", "state",
                  "purpose", "reviewed", "pallet_count", "bag_count", "target_id", "note"]
        writer = csv.DictWriter(buffer, fieldnames=fields)
        writer.writeheader()
        reviewed_paths = []
        for image in images:
            record = records.get(image["id"], {})
            objects = record.get("objects", [])
            name = image["id"].replace("/", "__")
            label_path = yolo / "labels" / (name + ".txt")
            image_path = yolo / "images" / (name + Path(image["image_path"]).suffix)
            if record.get("reviewed"):
                lines = []
                for box in objects:
                    normalized = [value / (image["width"] if axis == 0 else image["height"])
                                  for point in corners(box) for axis, value in enumerate(point)]
                    lines.append(str(box["class_id"]) + " " + " ".join(f"{min(1, max(0, value)):.8f}" for value in normalized))
                label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                # Link tương đối để repo chuyển máy vẫn đúng; link cũ trỏ sai thì tạo lại.
                target = os.path.relpath(ROOT / image["image_path"], image_path.parent)
                if image_path.is_symlink() and os.readlink(image_path) != target:
                    image_path.unlink()
                if not image_path.exists() and not image_path.is_symlink():
                    image_path.symlink_to(target)
                reviewed_paths.append(Path(os.path.relpath(image_path, ROOT)).as_posix())
            else:
                # Chỉ xóa bản xuất dẫn xuất khi người dùng sửa một ảnh đã xác nhận.
                label_path.unlink(missing_ok=True)
                if image_path.is_symlink():
                    image_path.unlink()
            targets = [box["id"] for box in objects if box["target"]]
            writer.writerow({"image_id": image["id"],
                             **{key: image[key] for key in ("image_path", "depth_path", "width", "height", "sequence_group", "state", "purpose")},
                             "reviewed": int(bool(record.get("reviewed"))),
                             "pallet_count": sum(box["class_id"] == 0 for box in objects),
                             "bag_count": sum(box["class_id"] == 1 for box in objects),
                             "target_id": targets[0] if targets else "", "note": record.get("note", "")})
        (output / "manifest.csv").write_text(buffer.getvalue(), encoding="utf-8-sig")
        (yolo / "reviewed.txt").write_text("".join(path + "\n" for path in reviewed_paths), encoding="utf-8")
        (yolo / "classes.yaml").write_text("names:\n  0: pallet_region\n  1: bag\n", encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, value, status=200, content_type="application/json; charset=utf-8"):
            body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path)
            if route.path == "/":
                self.respond((HERE / "index.html").read_bytes(), content_type="text/html; charset=utf-8")
            elif route.path == "/api/data":
                with lock:
                    self.respond({"images": images, "records": records, "classes": CLASSES, "output": str(output)})
            elif route.path == "/image":
                path = parse_qs(route.query).get("path", [""])[0]
                if path not in allowed_paths:
                    self.respond({"error": "Ảnh không thuộc bộ được chọn."}, 404)
                    return
                try:
                    self.respond((ROOT / path).read_bytes(), content_type="image/png")
                except OSError:
                    self.respond({"error": "Không đọc được ảnh gốc."}, 404)
            else:
                self.respond({"error": "Không tìm thấy."}, 404)

        def do_POST(self):
            if self.path != "/api/save":
                self.respond({"error": "Không tìm thấy."}, 404)
                return
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.respond({"error": "Chỉ nhận thay đổi từ giao diện local."}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 500000:
                    raise ValueError("Nội dung rỗng hoặc quá dài.")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict) or body.get("image_id") not in by_id:
                    raise ValueError("Ảnh không hợp lệ.")
                record = validate(body, by_id[body["image_id"]])
                with lock:
                    records[body["image_id"]] = record
                    write_results()
                self.respond({"record": record})
            except (ValueError, KeyError, TypeError) as error:
                self.respond({"error": str(error)}, 400)
            except OSError:
                self.respond({"error": "Không ghi được kết quả. Kiểm tra quyền ghi hoặc dung lượng đĩa, rồi lưu lại."}, 500)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        hint = " Thử --port 8767." if error.errno == errno.EADDRINUSE else ""
        parser.error(f"Không mở được cổng {args.port}: {error}.{hint}")
    write_results()
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"{len(images)} ảnh để gán nhãn. Mở {url}\nKết quả: {output}\nCtrl+C để dừng.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng; nhãn đã lưu được giữ nguyên.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
