"""Kernel Kaggle: dựng dataset, chạy các thí nghiệm song song trên các GPU, train model cuối,
dự đoán ảnh chưa nhãn. Đẩy lên bằng code/train/kaggle/push.py; không chạy local.

Input (dataset_sources): phanbth/bao-tai-1, phanbth/bao-tai-2, <user>/bao-tai-obb-code.
Output (/kaggle/working): runs/ (weights, biểu đồ, results.jsonl, predict_*), logs/.
"""
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
import torch  # noqa: E402

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
PROJ = Path("/tmp/proj")  # ngoài /kaggle/working để dataset trung gian không vào output
LOGS = WORK / "logs"
FOLDS = {"red": ["august_red", "september_pallet_4"],
         "sep3": ["september_pallet_3", "september_pallet_2"]}


def find_dir(name):
    for p in INPUT.rglob(name):
        if p.is_dir():
            return p
    raise SystemExit(f"Không thấy thư mục {name} trong /kaggle/input")


def setup():
    PROJ.mkdir(parents=True, exist_ok=True)
    # Ảnh gốc: link thư mục cha của một lượt chụp đã biết, đúng tên mà annotations.json dùng.
    for link, probe in [("bao-tai", "session_031"), ("2026-09-11", "140624_459950_0019fdc8")]:
        target = PROJ / link
        if not target.exists():
            target.symlink_to(find_dir(probe).parent)
    # Code: push.py upload file phẳng, "__" thay cho "/" (code__train__augment.py).
    code_dir = next(INPUT.rglob("code__train__build_dataset.py")).parent
    for f in code_dir.iterdir():
        if f.name.startswith("code__"):
            dest = PROJ / f.name.replace("__", "/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dest)
    print("PROJ:", sorted(str(p.relative_to(PROJ)) for p in (PROJ / "code").rglob("*")), flush=True)


def sh(args, log):
    LOGS.mkdir(exist_ok=True)
    with open(LOGS / f"{log}.log", "w") as f:
        r = subprocess.run([sys.executable, *args], cwd=PROJ, stdout=f, stderr=subprocess.STDOUT)
    print(f"[{'ok' if r.returncode == 0 else 'FAIL'}] {log}", flush=True)
    return r.returncode == 0


def build_all():
    jobs = []
    for fold, groups in FOLDS.items():
        base = ["code/train/build_dataset.py", "--val-groups", *groups, "--aug", "15"]
        jobs += [(base + ["--name", f"fold_{fold}_base", "--copies-only"], f"build_{fold}_base"),
                 (base + ["--name", f"fold_{fold}_aug"], f"build_{fold}_aug"),
                 (base + ["--name", f"fold_{fold}_v2", "--scene"], f"build_{fold}_v2")]
    jobs.append((["code/train/build_dataset.py", "--name", "all_v2", "--aug", "20", "--scene"], "build_all_v2"))
    threads = [threading.Thread(target=sh, args=j) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def train_all(runs):
    """Mỗi GPU một luồng lấy job từ hàng đợi chung."""
    q = queue.Queue()
    for r in runs:
        q.put(r)
    n_gpu = max(torch.cuda.device_count(), 1)

    def worker(dev):
        while True:
            try:
                name, extra = q.get_nowait()
            except queue.Empty:
                return
            sh(["code/train/train.py", "--name", name, "--device", str(dev), "--workers", "2", *extra], name)

    threads = [threading.Thread(target=worker, args=(d,)) for d in range(n_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main():
    t0 = time.time()
    setup()
    build_all()
    print(f"build xong sau {time.time() - t0:.0f}s", flush=True)
    y26, y11 = ["--model", "yolo26n-obb.pt"], ["--model", "yolo11n-obb.pt"]
    runs = []
    for fold in FOLDS:
        runs += [(f"{fold}_y26n_base", ["--data", f"fold_{fold}_base", *y26]),
                 (f"{fold}_y26n_aug", ["--data", f"fold_{fold}_aug", *y26]),
                 (f"{fold}_y26n_v2", ["--data", f"fold_{fold}_v2", *y26]),
                 (f"{fold}_y11n_v2", ["--data", f"fold_{fold}_v2", *y11]),
                 (f"{fold}_y26n_v2_960", ["--data", f"fold_{fold}_v2", *y26, "--imgsz", "960"])]
    # Model cuối: cả 25 ảnh, không validate (không còn ảnh độc lập); dùng last.pt.
    runs += [("all_y26n_v2", ["--data", "all_v2", *y26, "--epochs", "100", "--no-val"]),
             ("all_y26n_v2_960", ["--data", "all_v2", *y26, "--epochs", "100", "--imgsz", "960", "--no-val"])]
    train_all(runs)
    print(f"train xong sau {time.time() - t0:.0f}s", flush=True)
    for name, imgsz in [("all_y26n_v2", "640"), ("all_y26n_v2_960", "960")]:
        weights = PROJ / "runs" / name / "weights" / "last.pt"
        if weights.exists():
            sh(["code/train/predict.py", "--weights", str(weights), "--imgsz", imgsz], f"predict_{name}")
    shutil.copytree(PROJ / "runs", WORK / "runs", dirs_exist_ok=True)
    print(open(WORK / "runs" / "results.jsonl").read() if (WORK / "runs/results.jsonl").exists() else "")
    print(f"xong sau {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
