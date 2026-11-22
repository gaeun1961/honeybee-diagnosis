"""val/test 전체 추론 -> 임계값 조정, 라벨 조합별 오답, 오차 큰 이미지 목록

실행: python src/evaluate.py [--weights models/best.pt] [--by-site]
--by-site: test만 추론해 파일명 첫 글자(촬영 장소/장비, 예: B/C)별 클래스 AP -> docs/site_eval.md (다른 분석은 건너뜀)
- 전처리는 train.py 평가용과 같음 (16:9 리사이즈, ImageNet 정규화)
- val에서 클래스별 F1이 최대인 임계값을 찾고, test에서 0.5 기준과 비교
- test 정답 라벨 조합별로 클래스마다 잘못 켠 장수(+)와 놓친 장수(-)
- 확률 오차(|정답-확률| 합)가 큰 test 이미지 15장 -> docs/eval_errors.md (파일명만, 이미지는 저장 안 함)
- 클래스별 양성 test 이미지를 최대 박스 비율 구간(작음 <1%, 중간 1~5%, 큼 >=5%)으로 나눈 재현율 -> 출력 + eval_errors.md 끝
  (labels.csv에 area_* 열이 필요: make_subset.py로 표본을 다시 만들 것)
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from predict import load_model
from train import LABELS, MEAN, STD, Larva, load_rows

ROOT = Path(__file__).resolve().parent.parent
TOP_ERRORS = 15
MIN_POS = 5  # site별 AP: 양성이 이보다 적으면 "표본 부족"
SIZE_BINS = [("작음 <1%", 0.0, 0.01), ("중간 1~5%", 0.01, 0.05), ("큼 >=5%", 0.05, float("inf"))]


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


def size_recall(rows, y, pred):
    """클래스별·박스 비율 구간별 [클래스, 구간, 양성 장수, 재현율, 놓침 비율]"""
    out = []
    for k, c in enumerate(LABELS):
        area = np.array([float(r[f"area_{c}"]) for r in rows])
        for name, lo, hi in SIZE_BINS:
            m = (y[:, k] == 1) & (area >= lo) & (area < hi)
            n = int(m.sum())
            rec = pred[m, k].mean() if n else float("nan")
            out.append([c, name, n, "-" if not n else f"{rec:.3f}", "-" if not n else f"{1 - rec:.3f}"])
    return out


def site_ap(files, y, p):
    """파일명 첫 글자별 [site, 장수, (양성 수, AP 또는 사유) x 클래스, 평균 AP]"""
    site = np.array([f[0] for f in files])
    rows = []
    for s in sorted(set(site)):
        m = site == s
        cells, aps = [], []
        for k in range(len(LABELS)):
            pos = int(y[m, k].sum())
            if pos < MIN_POS:
                cells += [pos, "표본 부족"]
            elif pos == m.sum():
                cells += [pos, "음성 없음"]
            else:
                ap = average_precision_score(y[m, k], p[m, k])
                aps.append(ap)
                cells += [pos, f"{ap:.3f}"]
        rows.append([s, int(m.sum()), *cells, f"{np.mean(aps):.3f} ({len(aps)}개 클래스)" if aps else "-"])
    return rows


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models/best.pt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out-root", type=Path, default=ROOT, help="docs/eval_errors.md를 쓸 위치")
    ap.add_argument("--by-site", action="store_true", help="test를 파일명 첫 글자별로 나눠 클래스별 AP만 계산")
    a = ap.parse_args()

    model, size = load_model(a.weights)
    sub = a.data_dir / "subset"
    rows = load_rows(sub)
    if a.by_site:
        yt, pt = infer(model, sub / "images", rows["test"], size, a.batch)
        headers = ["첫 글자", "test 장수"] + [h for c in LABELS for h in (f"{c} 양성", f"{c} AP")] + ["평균 AP"]
        md = "\n".join(["# 파일명 첫 글자(촬영 장소)별 test AP", "",
                        f"`python src/evaluate.py --by-site`로 생성. 가중치 `{a.weights.name}` (입력 {size}px), 기본 분할 test {len(yt)}장.",
                        f"양성이 {MIN_POS}장 미만인 클래스는 \"표본 부족\", 전부 양성이면 \"음성 없음\"으로 AP를 내지 않음. 평균 AP는 AP를 낸 클래스만 평균.", "",
                        table(headers, site_ap([r["file"] for r in rows["test"]], yt, pt))]) + "\n"
        path = a.out_root / "docs/site_eval.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        print(md)
        print("wrote", path)
        return
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
    if rows["test"] and all(f"area_{c}" in rows["test"][0] for c in LABELS):
        sec = ["## 박스 크기별 재현율 (test, val 최적 임계값)", "",
               "정답이 양성인 이미지를 그 클래스의 최대 박스 비율(박스 넓이 / 이미지 넓이)로 나눔.", "",
               table(["클래스", "박스 비율", "장수", "재현율", "놓침 비율"], size_recall(rows["test"], yt, pred))]
    else:
        sec = ["## 박스 크기별 재현율", "", "labels.csv에 area_* 열이 없어 건너뜀 (make_subset.py로 표본을 다시 만들 것)."]
    print("\n" + "\n".join(sec))
    L += [""] + sec
    path = a.out_root / "docs/eval_errors.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\n오차 상위 {len(idx)}장 -> {path}")


if __name__ == "__main__":
    main()
