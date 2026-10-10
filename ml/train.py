"""Train the CropCurve leaf-disease classifier (transfer learning) and export it for the backend.

Built for the Kaggle "New Plant Diseases Dataset (Augmented)": 38 classes, 14 crops, with
ready-made train/ and valid/ folders. Point --data-dir anywhere above those folders; the
script finds them. A plain folder of class folders also works (it splits 80/20 itself).

Each class folder becomes a disease id by "slugifying" its name, e.g.
"Corn_(maize)___Common_rust_" -> "corn_maize_common_rust". These ids must match
data/diseases.csv and data/pesticides.csv.

Kaggle (GPU on, Internet on):
    python train.py --data-dir /kaggle/input/new-plant-diseases-dataset \
                    --test-dir /kaggle/input/new-plant-diseases-dataset/test --out models

Only some crops:   --crops tomato potato
Faster test run:   --max-per-class 200 --epochs 1

Outputs in --out:
    cropcurve_model.onnx   used by the backend (onnxruntime, no PyTorch needed)
    labels.json            class order + preprocessing settings
    cropcurve_model.pt     PyTorch weights, for further fine-tuning
    metrics.json           accuracy, per-class accuracy, confusion matrix
    samples/               2 confidently-recognised validation photos per class; the web app
                           shows these as "try a sample" leaves
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def crop_of(folder: str) -> str:
    return slugify(folder.split("___")[0]).split("_")[0]


def images_in(d: Path) -> list[Path]:
    return sorted(f for f in d.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)


def class_dirs(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_dir() and any(
        f.suffix.lower() in IMG_EXT for f in list(p.iterdir())[:50]))


def find_layout(root: Path, max_depth: int = 5):
    """Return (train_dir, valid_dir) if a folder with train/ and valid/ (or val/) exists below
    root, else (class_root, None) for the first folder that directly holds class folders."""
    queue = [(root, 0)]
    first_class_root = None
    while queue:
        d, depth = queue.pop(0)
        try:
            subs = sorted(p for p in d.iterdir() if p.is_dir())
        except PermissionError:
            continue
        names = {p.name.lower(): p for p in subs}
        if "train" in names and ("valid" in names or "val" in names):
            return names["train"], names.get("valid") or names["val"]
        if first_class_root is None and len(class_dirs(d)) >= 2:
            first_class_root = d
        if depth < max_depth:
            queue += [(p, depth + 1) for p in subs]
    if first_class_root is None:
        raise SystemExit(f"Couldn't find class folders under {root}. Check the path with `ls`.")
    return first_class_root, None


def collect(train_dir: Path, valid_dir: Path | None, crops, classes, max_per_class, seed):
    rng = random.Random(seed)
    folders = [p for p in class_dirs(train_dir)
               if (not crops or crop_of(p.name) in crops) and (not classes or slugify(p.name) in classes)]
    if not folders:
        raise SystemExit("No classes left after --crops/--classes filtering.")
    ids = [slugify(p.name) for p in folders]
    train, val = [], []
    for label, folder in enumerate(folders):
        tr = images_in(folder)
        rng.shuffle(tr)
        if valid_dir is not None and (valid_dir / folder.name).is_dir():
            va = images_in(valid_dir / folder.name)
            rng.shuffle(va)
        else:  # no separate validation folder: hold out 20%
            n_val = max(1, len(tr) // 5)
            va, tr = tr[:n_val], tr[n_val:]
        if max_per_class:
            tr, va = tr[:max_per_class], va[:max(50, max_per_class // 4)]
        train += [(f, label) for f in tr]
        val += [(f, label) for f in va]
    print(f"{len(folders)} classes, {len(train)} training and {len(val)} validation images")
    for i, f in zip(ids, folders):
        print(f"  {i:52s} <- {f.name}")
    return ids, [p.name for p in folders], train, val


class LeafDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        with Image.open(path) as im:
            img = im.convert("RGB")
        return self.tf(img), label, i


def build_model(arch: str, n_classes: int) -> nn.Module:
    if arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    else:
        raise SystemExit(f"Unknown arch {arch}")
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
    return m


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, log_every=0):
    training = optimizer is not None
    model.train(training)
    total, correct, loss_sum = 0, 0, 0.0
    all_probs, all_labels, all_idx = [], [], []
    for step, (x, y, idx) in enumerate(loader, 1):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training), torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x)
            loss = criterion(logits, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if log_every and step % log_every == 0:
                print(f"    step {step}/{len(loader)}  loss {loss.item():.3f}", flush=True)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        if not training:
            all_probs.append(logits.float().softmax(1).cpu())
            all_labels.append(y.cpu())
            all_idx.append(idx)
    extra = (torch.cat(all_probs), torch.cat(all_labels), torch.cat(all_idx)) if not training else None
    return loss_sum / total, correct / total, extra


def export_onnx(model, path: Path, size: int):
    model = model.eval().cpu()
    dummy = torch.randn(1, 3, size, size)
    kw = dict(input_names=["input"], output_names=["logits"],
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo argument
        torch.onnx.export(model, dummy, str(path), **kw)
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ref = model(dummy).detach().numpy()
        got = sess.run(None, {"input": dummy.numpy()})[0]
        print(f"ONNX check: max difference vs PyTorch = {np.abs(ref - got).max():.2e}")
    except ImportError:
        print("onnxruntime not installed; skipped the ONNX check")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--test-dir", type=Path, help="optional folder of loose test photos to predict (e.g. Kaggle's test/)")
    ap.add_argument("--out", default="models", type=Path)
    ap.add_argument("--arch", default="mobilenet_v2", choices=["mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--crops", nargs="+", help="only these crops, e.g. tomato potato corn")
    ap.add_argument("--classes", nargs="+", help="only these class ids")
    ap.add_argument("--epochs", type=int, default=4, help="fine-tuning epochs after the head warm-up")
    ap.add_argument("--head-epochs", type=int, default=1, help="epochs training only the new head")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-per-class", type=int, default=1000, help="cap training images per class (0 = all ~1,800)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + ("" if device.type == "cuda" else
          "  (no GPU: this will be very slow. On Kaggle: Settings > Accelerator > GPU)"))

    train_dir, valid_dir = find_layout(args.data_dir)
    print(f"Training images from {train_dir}\nValidation images from {valid_dir or '(20% split of training)'}")
    crops = {c.lower() for c in args.crops} if args.crops else None
    classes = set(args.classes) if args.classes else None
    ids, folders, train_items, val_items = collect(train_dir, valid_dir, crops, classes, args.max_per_class, args.seed)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(0.35, 0.35, 0.3, 0.04),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(round(args.size * 256 / 224))), transforms.CenterCrop(args.size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    pin = device.type == "cuda"
    train_dl = DataLoader(LeafDataset(train_items, train_tf), args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=pin, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(LeafDataset(val_items, eval_tf), args.batch_size * 2,
                        num_workers=args.workers, pin_memory=pin)

    counts = Counter(l for _, l in train_items)
    weights = torch.tensor([len(train_items) / (len(ids) * counts[i]) for i in range(len(ids))], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)

    model = build_model(args.arch, len(ids)).to(device)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_acc, best_state = -1.0, None
    log_every = max(1, len(train_dl) // 5)

    def train_phase(name, epochs, params, lr):
        nonlocal best_acc, best_state
        if epochs <= 0:
            return
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, _ = run_epoch(model, train_dl, device, criterion, opt, scaler, log_every)
            va_loss, va_acc, _ = run_epoch(model, val_dl, device, criterion)
            sched.step()
            flag = ""
            if va_acc > best_acc:
                best_acc, flag = va_acc, "  * best so far"
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{name} {ep}/{epochs}] train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time() - t0:.0f}s{flag}", flush=True)

    for p in model.features.parameters():
        p.requires_grad = False
    train_phase("head", args.head_epochs, model.classifier.parameters(), 1e-3)
    for p in model.parameters():
        p.requires_grad = True
    train_phase("fine-tune", args.epochs, model.parameters(), args.lr)

    model.load_state_dict(best_state)
    _, val_acc, (probs, labels, idx) = run_epoch(model, val_dl, device, criterion)
    preds = probs.argmax(1)
    n = len(ids)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(labels.tolist(), preds.tolist()):
        cm[t][p] += 1
    per_class = {ids[i]: round(cm[i][i] / max(1, sum(cm[i])), 4) for i in range(n)}
    print(f"\nValidation accuracy (best epoch): {val_acc:.4f}")
    for c, a in sorted(per_class.items(), key=lambda kv: kv[1])[:8]:
        print(f"  weakest: {c:52s} {a:.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": args.arch, "classes": ids, "state_dict": model.state_dict()}, args.out / "cropcurve_model.pt")
    (args.out / "labels.json").write_text(json.dumps(
        {"classes": ids, "folders": folders, "arch": args.arch, "input_size": args.size,
         "mean": MEAN, "std": STD}, indent=2))
    (args.out / "metrics.json").write_text(json.dumps(
        {"val_acc": val_acc, "per_class_val_acc": per_class, "confusion_matrix": cm,
         "class_order": ids, "n_train": len(train_items), "n_val": len(val_items)}, indent=2))
    export_onnx(model, args.out / "cropcurve_model.onnx", args.size)

    # Sample photos for the web app: the 2 most confidently correct validation images per class
    sample_dir = args.out / "samples"
    shutil.rmtree(sample_dir, ignore_errors=True)
    sample_dir.mkdir()
    for ci, cid in enumerate(ids):
        cand = [(probs[k, ci].item(), idx[k].item()) for k in range(len(labels))
                if labels[k] == ci and preds[k] == ci]
        for rank, (_, i) in enumerate(sorted(cand, reverse=True)[:2], 1):
            src = val_items[i][0]
            with Image.open(src) as im:
                im.convert("RGB").save(sample_dir / f"{cid}__{rank}.jpg", quality=90)

    if args.test_dir and args.test_dir.exists():
        print("\nPredictions on the loose test photos:")
        model.eval()
        for f in sorted(p for p in args.test_dir.rglob("*") if p.suffix.lower() in IMG_EXT):
            with Image.open(f) as im:
                x = eval_tf(im.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                pr = model(x).softmax(1)[0]
            k = int(pr.argmax())
            print(f"  {f.name:32s} -> {ids[k]:52s} {pr[k].item():.0%}")

    print(f"\nDone. Download the '{args.out}' folder and put its contents in your project's models/ folder.")


if __name__ == "__main__":
    main()
