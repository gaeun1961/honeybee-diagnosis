"""유충 다중 라벨 분류 베이스라인 (ResNet18, CPU)

실행: python src/train.py [--epochs 8] [--size 384] [--batch 16] [--lr 1e-3]
입력: data/subset/images, labels.csv(file, 응애, 석고병, 부저병), split.csv(file, split)
출력: models/best.pt (val 평균 AP 최고), docs/baseline.md (test 지표)

- 출력 3개(응애·석고병·부저병), BCEWithLogitsLoss + pos_weight(neg/pos). 셋 다 0 = 정상
- 입력 16:9 유지 (size x size*9/16), 학습 증강은 좌우 반전 + 약한 색 변화
- 처음 2 epoch은 fc만 lr로, 이후 전체를 lr/10으로 미세조정
"""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["응애", "석고병", "부저병"]
FC_ONLY_EPOCHS = 2
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class Larva(Dataset):
    def __init__(self, img_dir, rows, tf):
        self.img_dir, self.rows, self.tf = img_dir, rows, tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        x = self.tf(Image.open(self.img_dir / r["file"]).convert("RGB"))
        return x, torch.tensor([float(r[c]) for c in LABELS])


def load_rows(sub):
    with open(sub / "labels.csv", encoding="utf-8") as f:
        labels = {r["file"]: r for r in csv.DictReader(f)}
    with open(sub / "split.csv", encoding="utf-8") as f:
        split = {r["file"]: r["split"] for r in csv.DictReader(f)}
    out = {"train": [], "val": [], "test": []}
    for file, s in split.items():
        out[s].append(labels[file])
    return out


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x)))
        ys.append(y)
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    ap = [average_precision_score(y[:, k], p[:, k]) if y[:, k].any() else math.nan for k in range(len(LABELS))]
    f1 = f1_score(y, p >= 0.5, average=None, zero_division=0).tolist()
    return {"ap": ap, "map": float(np.nanmean(ap)) if not all(map(math.isnan, ap)) else math.nan,
            "f1": f1, "pos": y.sum(0).astype(int).tolist(), "n": len(y)}


def fmt(m):
    aps = " ".join(f"{c} {a:.3f}" for c, a in zip(LABELS, m["ap"]))
    f1s = " ".join(f"{c} {f:.3f}" for c, f in zip(LABELS, m["f1"]))
    return f"AP[{aps}] 평균AP {m['map']:.3f} | F1@0.5[{f1s}]"


def set_phase(model, epoch, lr):
    fc_only = epoch < FC_ONLY_EPOCHS
    for name, p in model.named_parameters():
        p.requires_grad = not fc_only or name.startswith("fc.")
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr if fc_only else lr / 10)


def write_report(path, m, a, best_epoch, best_val, counts):
    L = ["# 베이스라인 (ResNet18 다중 라벨)", "",
         f"`python src/train.py --epochs {a.epochs} --size {a.size} --batch {a.batch} --lr {a.lr}`로 생성.",
         f"이미지 train {counts['train']} / val {counts['val']} / test {counts['test']}장. "
         f"val 평균 AP 최고 epoch {best_epoch} ({best_val:.3f})의 가중치로 test 평가. F1은 임계값 0.5.", "",
         "| 클래스 | test 양성 수 | AP | F1@0.5 |", "|---|---|---|---|"]
    for c, pos, ap, f1 in zip(LABELS, m["pos"], m["ap"], m["f1"]):
        L.append(f"| {c} | {pos} | {ap:.3f} | {f1:.3f} |")
    L.append(f"| **평균** | {m['n']}장 | {m['map']:.3f} | {sum(m['f1']) / len(LABELS):.3f} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--size", type=int, default=384, help="입력 가로 (세로 = size*9/16)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-root", type=Path, default=ROOT, help="models/best.pt, docs/baseline.md를 쓸 위치")
    ap.add_argument("--no-pretrained", action="store_true", help="ImageNet 가중치 없이 (시험용)")
    a = ap.parse_args()
    torch.manual_seed(0)

    sub = a.data_dir / "subset"
    rows = load_rows(sub)
    hw = (round(a.size * 9 / 16), a.size)
    norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    train_tf = transforms.Compose([transforms.Resize(hw), transforms.RandomHorizontalFlip(),
                                   transforms.ColorJitter(0.1, 0.1, 0.1), *norm])
    eval_tf = transforms.Compose([transforms.Resize(hw), *norm])
    loaders = {s: DataLoader(Larva(sub / "images", rows[s], train_tf if s == "train" else eval_tf),
                             batch_size=a.batch, shuffle=s == "train") for s in rows}

    y = np.array([[float(r[c]) for c in LABELS] for r in rows["train"]])
    pos = y.sum(0)
    pos_weight = torch.tensor([(len(y) - p) / p if p else 1.0 for p in pos], dtype=torch.float32)
    print("train 양성 수", dict(zip(LABELS, pos.astype(int).tolist())), "pos_weight", [round(w, 2) for w in pos_weight.tolist()])

    model = models.resnet18(weights=None if a.no_pretrained else models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_path = a.out_root / "models/best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch = -math.inf, 0
    for epoch in range(a.epochs):
        if epoch in (0, FC_ONLY_EPOCHS):
            opt = set_phase(model, epoch, a.lr)
        model.train()
        t, total = time.time(), 0.0
        for x, yb in loaders["train"]:
            opt.zero_grad()
            loss = loss_fn(model(x), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        m = evaluate(model, loaders["val"])
        phase = "fc" if epoch < FC_ONLY_EPOCHS else "all"
        print(f"epoch {epoch + 1}/{a.epochs} [{phase}] loss {total / len(rows['train']):.4f} "
              f"({time.time() - t:.0f}s) | val {fmt(m)}")
        score = -1.0 if math.isnan(m["map"]) else m["map"]
        if score > best:
            best, best_epoch = score, epoch + 1
            torch.save({"state_dict": model.state_dict(), "labels": LABELS, "size": a.size, "epoch": best_epoch}, best_path)
            print(f"  -> {best_path} 저장")

    model.load_state_dict(torch.load(best_path)["state_dict"])
    m = evaluate(model, loaders["test"])
    print(f"test {fmt(m)}")
    report = a.out_root / "docs/baseline.md"
    write_report(report, m, a, best_epoch, best, {s: len(r) for s, r in rows.items()})
    print("wrote", report)


if __name__ == "__main__":
    main()
