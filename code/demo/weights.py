"""Đẩy / tải weights của app demo qua Kaggle dataset (weights không đưa lên git).

    .venv/Scripts/python code/demo/weights.py download                  # co-dev: tải về weights/demo/
    .venv/Scripts/python code/demo/weights.py publish -a tuananhpham29  # người train: đẩy bản mới

Danh sách file và dataset trong code/demo/models.json. Dataset để private: chủ dataset vào trang
dataset trên Kaggle > Settings > Sharing để thêm tài khoản Kaggle của co-dev.
download dùng token Kaggle của máy đang chạy (~/.kaggle/kaggle.json hoặc access_token), cùng token
dùng để tải ảnh bằng code/train/download_data.py.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).resolve().parent / "models.json"
sys.path.insert(0, str(ROOT / "code/train/kaggle"))
import push  # noqa: E402
from push import kaggle, use_account, username  # noqa: E402


def entries(cfg):
    """(tên, mục) của mọi weights: các fold, rồi model live nếu có (live là tuỳ chọn khi tải)."""
    return list(cfg["folds"].items()) + ([("live", cfg["live"])] if "live" in cfg else [])


def publish(cfg, account):
    if account:
        use_account(account)
    owner = username()
    slug = cfg["kaggle_dataset"].split("/")[1]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for _, f in entries(cfg):
            shutil.copy(ROOT / f["source"], folder / Path(f["weights"]).name)
        shutil.copy(CONFIG, folder / CONFIG.name)
        (folder / "dataset-metadata.json").write_text(json.dumps(
            {"title": slug, "id": f"{owner}/{slug}", "licenses": [{"name": "other"}]}))
        exists = kaggle("datasets", "status", f"{owner}/{slug}", check=False, capture=True).returncode == 0
        if exists:
            kaggle("datasets", "version", "-p", str(folder), "-m", cfg["name"], "-q")
        else:
            kaggle("datasets", "create", "-p", str(folder), "-q")  # mặc định private
    if cfg["kaggle_dataset"] != f"{owner}/{slug}":
        cfg["kaggle_dataset"] = f"{owner}/{slug}"
        CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã đẩy {len(entries(cfg))} weights -> https://www.kaggle.com/datasets/{owner}/{slug}")


def run(*args):
    """Kaggle CLI, đọc output bằng UTF-8 (thanh tiến trình làm hỏng bảng mã mặc định của Windows)."""
    return subprocess.run([push.KAGGLE, *args], env=push.ENV, capture_output=True, encoding="utf-8",
                          errors="replace")


def download(cfg, account=None):
    """Tải từng file (không cần Kaggle đóng gói xong bản zip của cả dataset)."""
    if account:
        use_account(account)
    done = 0
    for name, f in entries(cfg):
        dest = ROOT / f["weights"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            r = run("datasets", "download", cfg["kaggle_dataset"], "-f", dest.name, "-p", tmp)
            files = [p for p in Path(tmp).iterdir()]
            if (r.returncode != 0 or not files) and name == "live":
                print(f"  Bỏ qua {dest.name}: chưa có trên dataset (trang camera trực tiếp sẽ báo thiếu).")
                continue
            if r.returncode != 0 or not files:
                raise SystemExit(f"Tải {dest.name} thất bại ({cfg['kaggle_dataset']}). Kiểm tra token Kaggle và "
                                 "quyền truy cập dataset (chủ dataset cần share cho tài khoản của bạn).\n"
                                 f"{r.stdout}{r.stderr}")
            src = files[0]
            if src.suffix == ".zip":  # Kaggle nén file lớn khi tải lẻ
                with zipfile.ZipFile(src) as zf:
                    zf.extractall(tmp)
                src = Path(tmp) / dest.name
            shutil.move(str(src), dest)
        print(f"  {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")
        done += 1
    print(f"Đã tải {done} weights.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["download", "publish"])
    ap.add_argument("-a", "--account", help="username trong .kaggle/tokens.json (bỏ trống: dùng ~/.kaggle)")
    args = ap.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    publish(cfg, args.account) if args.action == "publish" else download(cfg, args.account)


if __name__ == "__main__":
    main()
