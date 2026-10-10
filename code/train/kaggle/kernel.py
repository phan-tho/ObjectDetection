"""Kernel Kaggle: dựng dataset, chạy các thí nghiệm song song trên các GPU, train model cuối,
dự đoán ảnh chưa nhãn. Đẩy lên bằng code/train/kaggle/push.py; không chạy local.

Input (dataset_sources): phanbth/bao-tai-1, phanbth/bao-tai-2, <user>/bao-tai-obb-code.
Output (/kaggle/working): runs/ (weights, biểu đồ, results.jsonl, predict_*), logs/.
"""
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
import torch  # noqa: E402

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
PROJ = Path("/tmp/proj")  # ngoài /kaggle/working để dataset trung gian không vào output
LOGS = WORK / "logs"
# 4 fold theo nhóm cảnh, phủ cả 25 ảnh: mỗi ảnh làm val đúng một lần.
FOLDS = {"red": ["august_red", "september_pallet_4"],          # 4 ảnh: bao đỏ, bao lẫn loại
         "sep3": ["september_pallet_3", "september_pallet_2"],  # 3 ảnh: nhiều pallet, màng co
         "green": ["august_green_and_empty"],                   # 8 ảnh: bao xanh, vơi, rỗng
         "sep1": ["september_pallet_1"]}                         # 10 ảnh: chuỗi tháo hàng đến rỗng
# (tên cấu hình, tham số build_dataset, tham số train)
CONFIGS = [("base", ["--aug", "15", "--copies-only"], []),
           ("v1", ["--aug", "15"], []),
           ("v3", ["--aug", "30", "--heavy"], []),
           ("v3_960", None, ["--imgsz", "960"])]  # dùng lại dataset v3


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
    jobs = [(["code/train/build_dataset.py", "--name", f"{fold}_{cfg}", "--val-groups", *groups, *bargs],
             f"build_{fold}_{cfg}")
            for fold, groups in FOLDS.items() for cfg, bargs, _ in CONFIGS if bargs is not None]
    jobs.append((["code/train/build_dataset.py", "--name", "all_v3", "--aug", "30", "--heavy"], "build_all_v3"))
    with ThreadPoolExecutor(4) as pool:  # Kaggle có 4 CPU
        list(pool.map(lambda j: sh(*j), jobs))


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


def summarize():
    """Bảng mAP50-95 theo cấu hình x fold, trung bình theo fold và theo số ảnh val."""
    path = PROJ / "runs" / "results.jsonl"
    if not path.exists():
        return ""
    res = {r["run"]: r for r in map(json.loads, path.read_text().splitlines())}
    n_val = {f: len(list((PROJ / "datasets" / f"{f}_v1" / "val" / "images").glob("*.jpg"))) for f in FOLDS}
    lines = ["config | " + " | ".join(f"{f}({n_val[f]})" for f in FOLDS) + " | mean | weighted | bag | pallet"]
    for cfg, _, _ in CONFIGS:
        rs = [res.get(f"{f}_{cfg}") for f in FOLDS]
        if not all(rs):
            continue
        m = [r["mAP50-95"] for r in rs]
        w = sum(v * n_val[f] for v, f in zip(m, FOLDS)) / sum(n_val.values())
        bag = sum(r["per_class_mAP50-95"]["bag"] for r in rs) / len(rs)
        pal = sum(r["per_class_mAP50-95"]["pallet_region"] for r in rs) / len(rs)
        lines.append(f"{cfg} | " + " | ".join(f"{v:.3f}" for v in m) + f" | {sum(m) / len(m):.3f} | {w:.3f}"
                     f" | {bag:.3f} | {pal:.3f}")
    return "\n".join(lines)


def main():
    t0 = time.time()
    setup()
    build_all()
    print(f"build xong sau {time.time() - t0:.0f}s", flush=True)
    runs = []
    for fold in FOLDS:
        for cfg, bargs, targs in CONFIGS:
            data = f"{fold}_{cfg if bargs is not None else 'v3'}"
            runs.append((f"{fold}_{cfg}", ["--data", data, "--model", "yolo26n-obb.pt", *targs]))
    # Model cuối: cả 25 ảnh, không validate (không còn ảnh độc lập); dùng last.pt.
    final = ["--data", "all_v3", "--model", "yolo26n-obb.pt", "--epochs", "100", "--no-val"]
    runs += [("all_v3", final), ("all_v3_960", final + ["--imgsz", "960"])]
    train_all(runs)
    print(f"train xong sau {time.time() - t0:.0f}s", flush=True)
    # Phân tích lỗi trên val của từng fold, so v1 với v3.
    for cfg in ("v1", "v3"):
        pairs = [a for f in FOLDS for a in ("--pair", f"runs/{f}_{cfg}/weights/best.pt", f"{f}_{cfg}")]
        sh(["code/train/analyze_errors.py", *pairs, "--out", f"runs/errors_{cfg}"], f"errors_{cfg}")
    for name, imgsz in [("all_v3", "640"), ("all_v3_960", "960")]:
        weights = PROJ / "runs" / name / "weights" / "last.pt"
        if weights.exists():
            sh(["code/train/predict.py", "--weights", str(weights), "--imgsz", imgsz], f"predict_{name}")
    table = summarize()
    (PROJ / "runs" / "summary.txt").write_text(table)
    shutil.copytree(PROJ / "runs", WORK / "runs", dirs_exist_ok=True)
    print(table)
    print(f"xong sau {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
