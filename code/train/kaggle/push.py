"""Đẩy code + nhãn lên Kaggle (dataset private) rồi chạy kernel GPU.

    .venv/Scripts/python code/train/kaggle/push.py -a tuananhpham29            # upload code + chạy kernel
    .venv/Scripts/python code/train/kaggle/push.py -a tuananhpham29 --status   # xem trạng thái
    .venv/Scripts/python code/train/kaggle/push.py -a tuananhpham29 --pull     # tải output về runs/kaggle/

`-a` chọn tài khoản trong .kaggle/tokens.json của dự án (gitignore, dạng
[{"username": ..., "token": "KGAT_..."}]); bỏ `-a` thì dùng ~/.kaggle. Hết quota GPU thì
đổi sang tài khoản khác. Dataset/kernel đều private. Ảnh gốc lấy trực tiếp từ
phanbth/bao-tai-1, phanbth/bao-tai-2 nên không upload lại.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
KAGGLE = str(Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle"))
FILES = ["code/train/augment.py", "code/train/build_dataset.py", "code/train/train.py",
         "code/train/predict.py", "code/train/analyze_errors.py", "code/obb_labeler/app.py",
         "code/obb_labeler/selection.json",
         "code/obb_labeler/results/annotations.json",
         "code/image_sorter/results/labels.csv"]
SLUG_DATA = "bao-tai-obb-code"
SLUG_KERNEL = "bao-tai-obb-train"
TOKENS = ROOT / ".kaggle" / "tokens.json"
ENV = dict(os.environ)


def use_account(name):
    accounts = {a["username"]: a["token"] for a in json.loads(TOKENS.read_text(encoding="utf-8"))}
    if name not in accounts:
        raise SystemExit(f"Không có tài khoản {name} trong {TOKENS}: {', '.join(accounts)}")
    ENV["KAGGLE_API_TOKEN"] = accounts[name]


def kaggle(*args, check=True, capture=False):
    return subprocess.run([KAGGLE, *args], check=check, text=True, capture_output=capture, env=ENV)


def username():
    out = kaggle("config", "view", capture=True).stdout
    return next(line.split(":", 1)[1].strip() for line in out.splitlines() if "username" in line)


def push_code(user, tmp):
    folder = tmp / "data"
    folder.mkdir()
    for f in FILES:
        shutil.copy(ROOT / f, folder / f.replace("/", "__"))
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": SLUG_DATA, "id": f"{user}/{SLUG_DATA}", "licenses": [{"name": "other"}]}))
    exists = kaggle("datasets", "status", f"{user}/{SLUG_DATA}", check=False, capture=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", str(folder), "-m", "update code", "-q")
    else:
        kaggle("datasets", "create", "-p", str(folder), "-q")  # mặc định private
    # Kernel chỉ gắn được dataset khi Kaggle xử lý xong version mới.
    for _ in range(60):
        status = kaggle("datasets", "status", f"{user}/{SLUG_DATA}", check=False, capture=True).stdout
        if "ready" in status.lower():
            return
        time.sleep(5)
    raise SystemExit(f"Dataset chưa sẵn sàng: {status}")


def push_kernel(user, tmp):
    folder = tmp / "kernel"
    folder.mkdir()
    shutil.copy(HERE / "kernel.py", folder / "kernel.py")
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{SLUG_KERNEL}", "title": SLUG_KERNEL, "code_file": "kernel.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": ["phanbth/bao-tai-1", "phanbth/bao-tai-2", f"{user}/{SLUG_DATA}"],
        "competition_sources": [], "kernel_sources": []}, indent=1))
    # CLI trả mã 0 cả khi lỗi (vd hết quota GPU), nên phải đọc output.
    out = kaggle("kernels", "push", "-p", str(folder), "--accelerator", "NvidiaTeslaT4", capture=True)
    print(out.stdout, out.stderr)
    if "error" in (out.stdout + out.stderr).lower():
        raise SystemExit("Push kernel thất bại")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--account", help="username trong .kaggle/tokens.json")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--kernel-only", action="store_true", help="không upload lại code")
    args = ap.parse_args()
    if args.account:
        use_account(args.account)
    user = username()
    print(f"Tài khoản Kaggle: {user}")
    ref = f"{user}/{SLUG_KERNEL}"
    if args.status:
        kaggle("kernels", "status", ref)
    elif args.pull:
        out = ROOT / "runs" / "kaggle"
        out.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", ref, "-p", str(out), "-o")
        print(f"-> {out}")
    else:
        with tempfile.TemporaryDirectory() as tmp:
            if not args.kernel_only:
                push_code(user, Path(tmp))
            push_kernel(user, Path(tmp))
        print(f"https://www.kaggle.com/code/{ref}")


if __name__ == "__main__":
    main()
