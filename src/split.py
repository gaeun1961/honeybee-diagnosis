"""data/subset/labels.csv -> data/subset/split.csv (file, split)

실행: python src/split.py [--seed 0]
파일명(stem) 자연 정렬(숫자는 숫자 크기로) 후 연속 50장씩 묶고, 묶음 단위로 무작위 80/10/10 배정.
이웃 프레임(거의 같은 사진)이 train과 val/test에 나뉘어 들어가는 누수를 막기 위함.
"""
import argparse
import csv
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GROUP = 50
LABELS = ["응애", "석고병", "부저병"]


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def assign(files, seed):
    files = sorted(files, key=lambda f: natural_key(Path(f).stem))
    groups = [files[i:i + GROUP] for i in range(0, len(files), GROUP)]
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    n = len(groups)
    n_val = n_test = max(1, round(n * 0.1)) if n >= 3 else 0  # 묶음이 3개 미만이면 전부 train
    split = {}
    for rank, g in enumerate(order):
        name = "val" if rank < n_val else "test" if rank < n_val + n_test else "train"
        for f in groups[g]:
            split[f] = name
    return split


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    a = ap.parse_args()

    sub = a.data_dir / "subset"
    with open(sub / "labels.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split = assign([r["file"] for r in rows], a.seed)
    with open(sub / "split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "split"])
        w.writerows(sorted(split.items(), key=lambda kv: natural_key(Path(kv[0]).stem)))

    print(f"{sub / 'split.csv'} ({GROUP}장 묶음 {-(-len(rows) // GROUP)}개)")
    print("split  장수  " + "  ".join(LABELS + ["정상"]))
    for s in ["train", "val", "test"]:
        rs = [r for r in rows if split[r["file"]] == s]
        counts = [sum(r[c] == "1" for r in rs) for c in LABELS]
        normal = sum(all(r[c] == "0" for c in LABELS) for r in rs)
        print(f"{s:<6} {len(rs):>4}  " + "  ".join(f"{v:>4}" for v in counts + [normal]))
