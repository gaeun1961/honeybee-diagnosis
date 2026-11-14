"""val/test 전체 추론 -> 임계값 조정, 라벨 조합별 오답, 오차 큰 이미지 목록

실행: python src/evaluate.py [--weights models/best.pt]
- 전처리는 train.py 평가용과 같음 (16:9 리사이즈, ImageNet 정규화)
- val에서 클래스별 F1이 최대인 임계값을 찾고, test에서 0.5 기준과 비교
- test 정답 라벨 조합별로 클래스마다 잘못 켠 장수(+)와 놓친 장수(-)
- 확률 오차(|정답-확률| 합)가 큰 test 이미지 15장 -> docs/eval_errors.md (파일명만, 이미지는 저장 안 함)
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from predict import load_model
from train import LABELS, MEAN, STD, Larva, load_rows

ROOT = Path(__file__).resolve().parent.parent
TOP_ERRORS = 15


@torch.no_grad()
def infer(model, img_dir, rows, size, batch):
    tf = transforms.Compose([transforms.Resize((round(size * 9 / 16), size)),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ys, ps = [], []
    for x, y in DataLoader(Larva(img_dir, rows, tf), batch_size=batch):  # shuffle 없음 -> rows 순서 유지
        ps.append(torch.sigmoid(model(x)))
        ys.append(y)
    return torch.cat(ys).numpy().astype(int), torch.cat(ps).numpy()


def best_thresholds(y, p):
    """클래스별 F1 최대 임계값. 양성이 없는 클래스는 0.5"""
    out = []
    for k in range(y.shape[1]):
        if not y[:, k].any():
            out.append(0.5)
            continue
        cands = np.unique(np.append(p[:, k], 0.5))
        f1s = [f1_score(y[:, k], p[:, k] >= t, zero_division=0) for t in cands]
        out.append(float(cands[int(np.argmax(f1s))]))
    return out


def combo(row):
    on = [c for c, v in zip(LABELS, row) if v]
    return "+".join(on) if on else "정상"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models/best.pt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-root", type=Path, default=ROOT, help="docs/eval_errors.md를 쓸 위치")
    a = ap.parse_args()

    model, size = load_model(a.weights)
    sub = a.data_dir / "subset"
    rows = load_rows(sub)
    yv, pv = infer(model, sub / "images", rows["val"], size, a.batch)
    yt, pt = infer(model, sub / "images", rows["test"], size, a.batch)

    th = best_thresholds(yv, pv)
    f1_05 = f1_score(yt, pt >= 0.5, average=None, zero_division=0)
    f1_th = f1_score(yt, pt >= np.array(th), average=None, zero_division=0)
    print(f"## test F1 (val {len(yv)}장으로 임계값 조정, test {len(yt)}장)\n")
    print(table(["클래스", "test 양성", "F1@0.5", "val 최적 임계값", "F1@최적"],
                [[c, int(yt[:, k].sum()), f"{f1_05[k]:.3f}", f"{th[k]:.3f}", f"{f1_th[k]:.3f}"]
                 for k, c in enumerate(LABELS)]
                + [["**평균**", "", f"{f1_05.mean():.3f}", "", f"{f1_th.mean():.3f}"]]))

    pred = (pt >= np.array(th)).astype(int)
    groups = defaultdict(list)
    for yi, pi in zip(yt, pred):
        groups[combo(yi)].append((yi, pi))
    out = []
    for name, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        yy, pp = np.array([i[0] for i in items]), np.array([i[1] for i in items])
        cells = [f"+{int(((pp[:, k] == 1) & (yy[:, k] == 0)).sum())} / -{int(((pp[:, k] == 0) & (yy[:, k] == 1)).sum())}"
                 for k in range(len(LABELS))]
        out.append([name, len(items), int((yy == pp).all(1).sum()), *cells])
    print("\n## test 정답 조합별 오답 (최적 임계값, +잘못 켬 / -놓침)\n")
    print(table(["정답 조합", "장수", "전부 맞춤", *LABELS], out))

    err = np.abs(yt - pt).sum(1)
    idx = np.argsort(-err)[:TOP_ERRORS]
    files = [r["file"] for r in rows["test"]]
    L = ["# 확률 오차가 큰 test 이미지", "",
         f"`python src/evaluate.py`로 생성. 오차 = 클래스별 |정답 - 확률|의 합. 상위 {len(idx)}장 (파일명만, 이미지는 저장 안 함).",
         "val 최적 임계값: " + ", ".join(f"{c} {t:.3f}" for c, t in zip(LABELS, th)), "",
         table(["파일", "정답", *[f"{c} 확률" for c in LABELS], "오차"],
               [[files[i], combo(yt[i]), *[f"{v:.3f}" for v in pt[i]], f"{err[i]:.3f}"] for i in idx])]
    path = a.out_root / "docs/eval_errors.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n오차 상위 {len(idx)}장 -> {path}")


if __name__ == "__main__":
    main()
