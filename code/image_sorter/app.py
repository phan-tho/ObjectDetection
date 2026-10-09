#!/usr/bin/env python3
"""Giao diện phân loại ảnh local, chỉ dùng thư viện chuẩn Python."""

import argparse
import csv
import errno
import io
import json
import mimetypes
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent.parent
SOURCES = ("bao-tai", "2026-09-11")
STATES = [
    {"id": "full", "label": "Đầy — không hở đáy", "key": "1"},
    {"id": "partial", "label": "Vơi — hở đáy, còn trên 3 bao", "key": "2"},
    {"id": "few", "label": "Chỉ còn 1–3 bao", "key": "3"},
    {"id": "empty", "label": "Pallet rỗng", "key": "4"},
]
TAGS = [
    {"id": "overlap", "label": "Bao chồng lấn mạnh", "key": "q"},
    {"id": "rotation", "label": "Bao xoay nhiều góc", "key": "w"},
    {"id": "deformed", "label": "Bao nhăn / phồng / rách / móp", "key": "e"},
    {"id": "lighting", "label": "Ánh sáng khác thường", "key": "r"},
    {"id": "bag_color", "label": "Màu / loại bao khác thường", "key": "t"},
    {"id": "pallet_offset", "label": "Pallet lệch / xoay", "key": "y"},
    {"id": "overhang", "label": "Bao tràn khỏi biên pallet", "key": "u"},
    {"id": "bottom_visible", "label": "Hở đáy pallet", "key": "i"},
]
REASONS = [
    {"id": "cropped_pallet", "label": "Không thấy toàn bộ pallet mục tiêu"},
    {"id": "blur", "label": "Mờ / rung, không phân biệt được bao"},
    {"id": "exposure", "label": "Quá tối / cháy sáng, không dùng được"},
    {"id": "depth_invalid", "label": "Depth hỏng"},
    {"id": "other", "label": "Lý do khác (ghi chú bên dưới)"},
]


def relative(path):
    return path.relative_to(ROOT).as_posix()


def scan_groups():
    # Tận dụng nhóm chuỗi chụp đã có; đây là gợi ý cho việc chia tập sau này.
    sequence_groups = {}
    manifest = ROOT / "annotations/survey_2026-09-11/manifest.csv"
    if manifest.exists():
        with manifest.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                source = "/".join(Path(row["source"]).parts[-2:])
                sequence_groups[source] = row.get("group", "")

    groups = []
    for source in SOURCES:
        folder = ROOT / source
        if not folder.exists():
            continue
        for session in sorted(folder.iterdir()):
            if not session.is_dir():
                continue
            if source == "bao-tai":
                colors = sorted(session.glob("*_color.png"))
                if not colors:
                    continue
                representative = colors[0]
                depths = [p.with_name(p.name.replace("_color.png", "_depth.png")) for p in colors]
            else:
                representative = session / "color.png"
                if not representative.exists():
                    continue
                colors = [representative, *sorted(session.glob("raw/*/color.png"))]
                depths = [p.with_name("depth.png") for p in colors]
            metadata_path = session / "meta.json"
            metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
            scene = metadata.get("scene", {})
            context = metadata.get("context", {})
            group_id = relative(session)
            groups.append({
                "id": group_id,
                "source": source,
                "representative": relative(representative),
                "frames": [
                    {"color": relative(c), "depth": relative(d) if d.exists() else ""}
                    for c, d in zip(colors, depths)
                ],
                "captured_at": metadata.get("captured_at", ""),
                "pallet_id": scene.get("pallet_id", context.get("pallet_id", "")),
                "station_id": context.get("station_id", ""),
                "bag_type": scene.get("bag_type", context.get("material", "")),
                "sequence_group_hint": sequence_groups.get(group_id, ""),
            })
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--output", type=Path, default=APP_DIR / "results")
    args = parser.parse_args()
    groups = scan_groups()
    if not groups:
        parser.error("Không tìm thấy ảnh trong hai thư mục dữ liệu.")
    by_id = {group["id"]: group for group in groups}
    image_paths = {frame["color"] for group in groups for frame in group["frames"]}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = output / "labels.json", output / "labels.csv"
    records = json.loads(json_path.read_text(encoding="utf-8"))["records"] if json_path.exists() else {}
    lock = threading.Lock()

    def write_results():
        document = {"schema_version": 1, "states": STATES, "tags": TAGS, "records": records}
        temporary = json_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(json_path)
        buffer = io.StringIO(newline="")
        columns = ["group_id", "source", "image_path", "depth_path", "frame_count", "captured_at",
                   "pallet_id", "station_id", "bag_type", "sequence_group_hint", "status", "state",
                   *[tag["id"] for tag in TAGS], "reject_reason", "note", "updated_at"]
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        for group in groups:
            record = records.get(group["id"], {})
            writer.writerow({
                "group_id": group["id"], "source": group["source"],
                "image_path": group["representative"], "depth_path": group["frames"][0]["depth"],
                "frame_count": len(group["frames"]),
                **{key: group[key] for key in ("captured_at", "pallet_id", "station_id", "bag_type", "sequence_group_hint")},
                "status": record.get("status", "unreviewed"), "state": record.get("state", ""),
                **{tag["id"]: int(tag["id"] in record.get("tags", [])) for tag in TAGS},
                **{key: record.get(key, "") for key in ("reject_reason", "note", "updated_at")},
            })
        temporary = csv_path.with_suffix(".csv.tmp")
        temporary.write_text(buffer.getvalue(), encoding="utf-8-sig")
        temporary.replace(csv_path)

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
                self.respond((APP_DIR / "index.html").read_bytes(), content_type="text/html; charset=utf-8")
            elif route.path == "/api/data":
                with lock:
                    self.respond({"groups": groups, "records": records, "states": STATES,
                                  "tags": TAGS, "reasons": REASONS, "output": str(output)})
            elif route.path == "/labels.csv":
                with lock:
                    self.respond(csv_path.read_bytes(), content_type="text/csv; charset=utf-8")
            elif route.path == "/image":
                path = parse_qs(route.query).get("path", [""])[0]
                if path not in image_paths:
                    self.respond({"error": "Ảnh không có trong danh sách."}, 404)
                    return
                file = ROOT / path
                if not file.exists():
                    self.respond({"error": "Không tìm thấy ảnh gốc."}, 404)
                    return
                self.respond(file.read_bytes(), content_type=mimetypes.guess_type(file.name)[0] or "image/png")
            else:
                self.respond({"error": "Không tìm thấy."}, 404)

        def do_POST(self):
            if self.path != "/api/labels":
                self.respond({"error": "Không tìm thấy."}, 404)
                return
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.respond({"error": "Chỉ nhận thay đổi từ giao diện local."}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16000:
                    raise ValueError("Nội dung quá dài hoặc rỗng.")
                body = json.loads(self.rfile.read(size))
                group_id = body["id"]
                status, state = body.get("status"), body.get("state", "")
                tags, reason, note = body.get("tags", []), body.get("reject_reason", ""), body.get("note", "")
                if group_id not in by_id or status not in ("usable", "rejected", "review"):
                    raise ValueError("Lượt chụp hoặc trạng thái không hợp lệ.")
                if state not in ("", *[item["id"] for item in STATES]):
                    raise ValueError("Trạng thái pallet không hợp lệ.")
                if status == "usable" and not state:
                    raise ValueError("Chọn trạng thái pallet trước khi lưu.")
                if not isinstance(tags, list) or any(tag not in [item["id"] for item in TAGS] for tag in tags):
                    raise ValueError("Nhãn không hợp lệ.")
                if status == "rejected" and reason not in [item["id"] for item in REASONS]:
                    raise ValueError("Chọn lý do loại ảnh.")
                if not isinstance(note, str) or len(note) > 2000:
                    raise ValueError("Ghi chú tối đa 2000 ký tự.")
                if state == "full" and "bottom_visible" in tags:
                    raise ValueError("Pallet đầy không thể đồng thời hở đáy.")
                if state in ("partial", "empty") and "bottom_visible" not in tags:
                    raise ValueError("Pallet vơi/rỗng cần có nhãn hở đáy.")
                record = {"status": status, "state": state,
                          "tags": [item["id"] for item in TAGS if item["id"] in tags],
                          "reject_reason": reason if status == "rejected" else "", "note": note,
                          "updated_at": datetime.now().astimezone().isoformat(timespec="seconds")}
                with lock:
                    records[group_id] = record
                    write_results()
                self.respond({"record": record})
            except (ValueError, KeyError, TypeError) as error:
                self.respond({"error": str(error)}, 400)
            except OSError:
                self.respond({"error": "Không ghi được file kết quả. Kiểm tra quyền ghi hoặc dung lượng đĩa."}, 500)

    write_results()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as error:
        hint = " Thử --port 8766." if error.errno == errno.EADDRINUSE else ""
        parser.error(f"Không mở được cổng {args.port}: {error}.{hint}")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"{len(groups)} lượt chụp. Mở {url}\nKết quả: {output}\nNhấn Ctrl+C để dừng.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng. Kết quả đã lưu vẫn được giữ.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
