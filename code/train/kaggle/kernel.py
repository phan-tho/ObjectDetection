"""Kernel Kaggle, đẩy lên bằng code/train/kaggle/push.py; không chạy local.

STAGE "compare": so cấu hình augment (4 fold), phân tích lỗi, train model cuối, dự đoán ảnh
chưa nhãn. STAGE "methods" (kernel riêng, mount output của "compare"): train model crop cho
two-stage và biến thể đóng băng backbone, chấm plain/tiled/two_stage/+inpallet bằng pipeline.py.
STAGE "fix" (kernel riêng, mount "compare"): augment v4 cho case kém, so v1/v3/v4 ở 960 với
cùng số bản augment và 2 seed, chấm last.pt; model cuối all_v4_960.
STAGE "v5" (mount "fix"): augment v5 cùng điều kiện, gộp bảng với fix; model cuối all_v5_960.

Input (dataset_sources): phanbth/bao-tai-1, phanbth/bao-tai-2, <user>/bao-tai-obb-code.
Output (/kaggle/working): runs/ (weights, biểu đồ, results.jsonl, summary.txt...), logs/.
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
import numpy as np  # noqa: E402
import torch  # noqa: E402

STAGE = "compare"  # push.py --stage methods thay dòng này
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
        pairs = [a for f, g in FOLDS.items() for a in ("--fold", f"runs/{f}_{cfg}/weights/last.pt", *g)]
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


def summarize_methods():
    """Trung bình mAP50-95 qua 4 fold cho mỗi (model, phương pháp)."""
    path = PROJ / "runs" / "pipeline.jsonl"
    if not path.exists():
        return ""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    groups = {}
    for r in rows:
        groups.setdefault((r["tag"], r["method"]), []).append(r)
    lines = ["model | method | " + " | ".join(FOLDS) + " | mean | weighted | bag | pallet"]
    for (tag, method), rs in sorted(groups.items()):
        by = {r["fold"]: r for r in rs}
        if set(by) != set(FOLDS):
            continue
        m = [by[f]["mAP50-95"] for f in FOLDS]
        n = [by[f]["n_val"] for f in FOLDS]
        bag = sum(by[f]["per_class_mAP50-95"].get("bag", 0) for f in FOLDS) / len(FOLDS)
        pal = sum(by[f]["per_class_mAP50-95"].get("pallet_region", 0) for f in FOLDS) / len(FOLDS)
        lines.append(f"{tag} | {method} | " + " | ".join(f"{v:.3f}" for v in m)
                     + f" | {sum(m) / len(m):.3f} | {sum(a * b for a, b in zip(m, n)) / sum(n):.3f}"
                     f" | {bag:.3f} | {pal:.3f}")
    return "\n".join(lines)


def methods():
    t0 = time.time()
    setup()
    # Weights vòng "compare" (kernel_sources): thư mục runs có red_v3.
    prev = next(p.parent for p in INPUT.rglob("red_v3") if (p.parent / "results.jsonl").exists())
    print("prev runs:", prev, flush=True)
    jobs = []
    for fold, groups in FOLDS.items():
        jobs.append((["code/train/build_crops.py", "--name", f"{fold}_crop", "--val-groups", *groups,
                      "--aug", "15", "--heavy"], f"build_{fold}_crop"))
        # Cùng seed với vòng compare nên dataset v3 giống hệt: so freeze công bằng.
        jobs.append((["code/train/build_dataset.py", "--name", f"{fold}_v3", "--val-groups", *groups,
                      "--aug", "30", "--heavy"], f"build_{fold}_v3"))
    jobs.append((["code/train/build_crops.py", "--name", "all_crop", "--aug", "20", "--heavy"], "build_all_crop"))
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda j: sh(*j), jobs))
    print(f"build xong sau {time.time() - t0:.0f}s", flush=True)
    runs = []
    for fold in FOLDS:
        runs += [(f"{fold}_crop", ["--data", f"{fold}_crop", "--model", "yolo26n-obb.pt"]),
                 (f"{fold}_v3_freeze", ["--data", f"{fold}_v3", "--model", "yolo26n-obb.pt", "--freeze", "11"])]
    runs.append(("all_crop", ["--data", "all_crop", "--model", "yolo26n-obb.pt", "--epochs", "100", "--no-val"]))
    train_all(runs)
    print(f"train xong sau {time.time() - t0:.0f}s", flush=True)
    for fold, groups in FOLDS.items():
        crop = PROJ / "runs" / f"{fold}_crop" / "weights" / "best.pt"
        for tag, model in [("v1", prev / f"{fold}_v1"), ("v3", prev / f"{fold}_v3"),
                           ("v3_960", prev / f"{fold}_v3_960"), ("v3_freeze", PROJ / "runs" / f"{fold}_v3_freeze")]:
            weights = model / "weights" / "best.pt"
            if not weights.exists():
                continue
            imgsz = "960" if tag.endswith("960") else "640"
            extra = ["--crop-model", str(crop)] if crop.exists() and tag in ("v1", "v3") else []
            sh(["code/train/pipeline.py", "--fold", fold, "--val-groups", *groups, "--model", str(weights),
                "--imgsz", imgsz, "--tag", tag, *extra], f"pipeline_{fold}_{tag}")
    table = summarize_methods()
    (PROJ / "runs" / "summary_methods.txt").write_text(table)
    shutil.copytree(PROJ / "runs", WORK / "runs", dirs_exist_ok=True)
    print(table)
    print(f"xong sau {time.time() - t0:.0f}s", flush=True)


def summarize_fix():
    """mAP50-95 (last.pt, chấm bằng pipeline.py) theo cấu hình: từng seed rồi trung bình các seed."""
    rows = [json.loads(line) for line in (PROJ / "runs" / "pipeline.jsonl").read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault((r["tag"], r["method"]), {})[r["fold"]] = r
    lines = ["config | seed | method | " + " | ".join(FOLDS) + " | mean | bag | pallet"]
    agg = {}
    for (tag, method), fr in sorted(by.items()):
        if set(fr) != set(FOLDS):
            continue
        cfg, seed = tag.rsplit("_s", 1)
        m = [fr[f]["mAP50-95"] for f in FOLDS]
        bag = sum(fr[f]["per_class_mAP50-95"].get("bag", 0) for f in FOLDS) / len(FOLDS)
        pal = sum(fr[f]["per_class_mAP50-95"].get("pallet_region", 0) for f in FOLDS) / len(FOLDS)
        lines.append(f"{cfg} | {seed} | {method} | " + " | ".join(f"{v:.3f}" for v in m)
                     + f" | {sum(m) / len(m):.3f} | {bag:.3f} | {pal:.3f}")
        agg.setdefault((cfg, method), []).append((sum(m) / len(m), bag, pal))
    lines.append("\nTrung bình các seed:")
    for (cfg, method), v in sorted(agg.items()):
        a = np.array(v)
        lines.append(f"{cfg} | {method} | n_seed={len(v)} | mean={a[:, 0].mean():.3f} "
                     f"(±{a[:, 0].std():.3f}) | bag={a[:, 1].mean():.3f} | pallet={a[:, 2].mean():.3f}")
    return "\n".join(lines)


def fix():
    """Vòng 4: sửa case kém. So v1/v3/v4 cùng ảnh 960, cùng 30 bản augment, 2 seed, chấm last.pt."""
    t0 = time.time()
    setup()
    prev = next(p.parent for p in INPUT.rglob("red_v3_960") if (p.parent / "results.jsonl").exists())
    print("prev runs:", prev, flush=True)
    jobs = []
    for fold, groups in FOLDS.items():
        base = ["code/train/build_dataset.py", "--val-groups", *groups, "--aug", "30"]
        jobs += [(base + ["--name", f"{fold}_v1"], f"build_{fold}_v1"),
                 (base + ["--name", f"{fold}_v3", "--heavy"], f"build_{fold}_v3"),
                 (base + ["--name", f"{fold}_v4", "--v4"], f"build_{fold}_v4")]
    jobs.append((["code/train/build_dataset.py", "--name", "all_v4", "--aug", "30", "--v4"], "build_all_v4"))
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda j: sh(*j), jobs))
    print(f"build xong sau {time.time() - t0:.0f}s", flush=True)
    y = ["--model", "yolo26n-obb.pt", "--imgsz", "960"]
    runs = []
    for fold in FOLDS:
        runs += [(f"{fold}_v4_960_s0", ["--data", f"{fold}_v4", *y]),
                 (f"{fold}_v1_960_s0", ["--data", f"{fold}_v1", *y]),
                 (f"{fold}_v4_960_s1", ["--data", f"{fold}_v4", *y, "--seed", "1"]),
                 (f"{fold}_v3_960_s1", ["--data", f"{fold}_v3", *y, "--seed", "1"])]
    runs.append(("all_v4_960", ["--data", "all_v4", *y, "--epochs", "100", "--no-val"]))
    train_all(runs)
    print(f"train xong sau {time.time() - t0:.0f}s", flush=True)
    for fold, groups in FOLDS.items():
        models = {f"v3_960_s0": prev / f"{fold}_v3_960"}  # seed 0 của v3 lấy từ vòng compare
        models.update({t: PROJ / "runs" / f"{fold}_{t}" for t in ("v4_960_s0", "v1_960_s0", "v4_960_s1", "v3_960_s1")})
        for tag, run in models.items():
            weights = run / "weights" / "last.pt"
            if weights.exists():
                sh(["code/train/pipeline.py", "--fold", fold, "--val-groups", *groups, "--model", str(weights),
                    "--imgsz", "960", "--tag", tag, "--methods", "plain", "plain+inpallet"],
                   f"pipeline_{fold}_{tag}")
    for tag, root in [("v3", prev), ("v4", PROJ / "runs")]:
        name = "v3_960" if tag == "v3" else "v4_960_s0"
        pairs = [a for f, g in FOLDS.items() for a in ("--fold", str(root / f"{f}_{name}" / "weights" / "last.pt"), *g)]
        sh(["code/train/analyze_errors.py", "--imgsz", "960", *pairs, "--out", f"runs/errors_{tag}_960"],
           f"errors_{tag}_960")
    final = PROJ / "runs" / "all_v4_960" / "weights" / "last.pt"
    if final.exists():
        sh(["code/train/predict.py", "--weights", str(final), "--imgsz", "960"], "predict_all_v4_960")
    table = summarize_fix()
    (PROJ / "runs" / "summary_fix.txt").write_text(table)
    shutil.copytree(PROJ / "runs", WORK / "runs", dirs_exist_ok=True)
    print(table)
    print(f"xong sau {time.time() - t0:.0f}s", flush=True)


def v5():
    """Vòng 5: augment v5 (ánh sáng theo lớp xếp/từng bao, bóng loáng, cháy sáng, bao biến dạng)
    cùng điều kiện với vòng fix (960, 30 bản augment, seed 0 và 1, last.pt); gộp bảng với fix."""
    t0 = time.time()
    setup()
    prev = next(p.parent for p in INPUT.rglob("pipeline.jsonl") if (p.parent / "summary_fix.txt").exists())
    print("fix runs:", prev, flush=True)
    (PROJ / "runs").mkdir(exist_ok=True)
    shutil.copy(prev / "pipeline.jsonl", PROJ / "runs" / "pipeline.jsonl")  # số v1/v3/v4 của vòng fix
    jobs = [(["code/train/build_dataset.py", "--name", f"{fold}_v5", "--val-groups", *groups, "--aug", "30",
              "--v5"], f"build_{fold}_v5") for fold, groups in FOLDS.items()]
    jobs.append((["code/train/build_dataset.py", "--name", "all_v5", "--aug", "30", "--v5"], "build_all_v5"))
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda j: sh(*j), jobs))
    print(f"build xong sau {time.time() - t0:.0f}s", flush=True)
    y = ["--model", "yolo26n-obb.pt", "--imgsz", "960"]
    runs = [(f"{fold}_v5_960_s{seed}", ["--data", f"{fold}_v5", *y, "--seed", str(seed)])
            for fold in FOLDS for seed in (0, 1)]
    runs.append(("all_v5_960", ["--data", "all_v5", *y, "--epochs", "100", "--no-val"]))
    train_all(runs)
    print(f"train xong sau {time.time() - t0:.0f}s", flush=True)
    for fold, groups in FOLDS.items():
        for seed in (0, 1):
            weights = PROJ / "runs" / f"{fold}_v5_960_s{seed}" / "weights" / "last.pt"
            if weights.exists():
                sh(["code/train/pipeline.py", "--fold", fold, "--val-groups", *groups, "--model", str(weights),
                    "--imgsz", "960", "--tag", f"v5_960_s{seed}", "--methods", "plain", "plain+inpallet"],
                   f"pipeline_{fold}_v5_s{seed}")
    pairs = [a for f, g in FOLDS.items()
             for a in ("--fold", str(PROJ / "runs" / f"{f}_v5_960_s0" / "weights" / "last.pt"), *g)]
    sh(["code/train/analyze_errors.py", "--imgsz", "960", *pairs, "--out", "runs/errors_v5_960"], "errors_v5_960")
    final = PROJ / "runs" / "all_v5_960" / "weights" / "last.pt"
    if final.exists():
        sh(["code/train/predict.py", "--weights", str(final), "--imgsz", "960"], "predict_all_v5_960")
    table = summarize_fix()
    (PROJ / "runs" / "summary_v5.txt").write_text(table)
    shutil.copytree(PROJ / "runs", WORK / "runs", dirs_exist_ok=True)
    print(table)
    print(f"xong sau {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    {"methods": methods, "fix": fix, "v5": v5}.get(STAGE, main)()
