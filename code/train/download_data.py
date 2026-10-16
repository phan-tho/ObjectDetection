"""Tải 2 bộ ảnh gốc từ Kaggle về đúng tên thư mục mà các tool khác đang dùng.

    .venv/Scripts/python code/train/download_data.py

Cần token Kaggle (~/.kaggle/access_token hoặc kaggle.json). Thư mục đã có thì bỏ qua.
"""
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "phanbth/bao-tai-1": "bao-tai",
    "phanbth/bao-tai-2": "2026-09-11",
}


def unwrap(folder: Path) -> None:
    """Zip thường bọc thêm một thư mục cùng tên: đưa nội dung lên một cấp."""
    while True:
        children = [p for p in folder.iterdir() if p.name != ".DS_Store"]
        if len(children) != 1 or not children[0].is_dir():
            return
        inner = children[0]
        for item in list(inner.iterdir()):
            shutil.move(str(item), folder / item.name)
        inner.rmdir()


def move(src: Path, dst: Path) -> None:
    """Windows hay khóa thư mục vừa giải nén (antivirus quét): thử lại, rồi chuyển từng file."""
    for _ in range(5):
        try:
            src.rename(dst)
            return
        except PermissionError:
            time.sleep(3)
    dst.mkdir(exist_ok=True)
    for item in list(src.iterdir()):
        shutil.move(str(item), dst / item.name)
    shutil.rmtree(src, ignore_errors=True)


def main() -> None:
    kaggle = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    for slug, name in DATASETS.items():
        target = ROOT / name
        if target.exists() and any(target.iterdir()):
            print(f"[skip] {name} đã có")
            continue
        staging = ROOT / f"_download_{name}"
        staging.mkdir(exist_ok=True)
        if not (staging / "data").exists() or any(staging.glob("*.zip")):
            print(f"[get] {slug} -> {name}")
            subprocess.run([str(kaggle), "datasets", "download", "-d", slug, "-p", str(staging)], check=True)
            for archive in staging.glob("*.zip"):
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(staging / "data")
                archive.unlink()
        unwrap(staging / "data")
        move(staging / "data", target)
        shutil.rmtree(staging, ignore_errors=True)
        print(f"[ok] {name}: {sum(1 for _ in target.rglob('*'))} file")


if __name__ == "__main__":
    main()
