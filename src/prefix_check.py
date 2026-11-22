"""파일명 접두어(앞 3칸, 벌통 추정)별 클래스 구성 확인 -> docs/prefix_check.md (집계값만)

실행: python src/prefix_check.py
입력: data/subset/labels.csv, split.csv(기본 50장 묶음 분할), split_prefix.csv(sweep prefix384의 접두어 분할)
- 접두어 수, 접두어별 클래스 개수표(이미지 수 기준, 다중 라벨이면 각 클래스에 셈, 정상 = 병 없음)
- 클래스가 하나뿐인 접두어 비율 (접두어 안 이미지들의 클래스 합집합이 1개)
- 기본 분할(split.csv)에서 한 접두어가 train/val/test 중 몇 곳에 걸치는지
- prefix384 test에 들어간 접두어 목록과 클래스 구성
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from split import LABELS, natural_key, prefix_of

ROOT = Path(__file__).resolve().parent.parent
CLASSES = LABELS + ["정상"]


def classes_of(r):
    on = [c for c in LABELS if r[c] == "1"]
    return on or ["정상"]


def read(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def class_rows(groups, extra=None):
    rows = []
    for p in sorted(groups, key=natural_key):
        cnt = Counter(c for r in groups[p] for c in classes_of(r))
        rows.append([p, len(groups[p]), *(cnt[c] for c in CLASSES), *(extra(p) if extra else [])])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/prefix_check.md")
    a = ap.parse_args()
    sub = a.data_dir / "subset"

    labels = {r["file"]: r for r in read(sub / "labels.csv")}
    split = {r["file"]: r["split"] for r in read(sub / "split.csv")}
    groups = defaultdict(list)
    for f in split:
        groups[prefix_of(f)].append(labels[f])

    n_img = len(split)
    single = [p for p, rs in groups.items() if len({c for r in rs for c in classes_of(r)}) == 1]
    single_img = sum(len(groups[p]) for p in single)
    spans = {p: len({split[r["file"]] for r in rs}) for p, rs in groups.items()}
    span_cnt = Counter(spans.values())

    L = ["# 파일명 접두어(앞 3칸, 벌통 추정)별 클래스 구성", "",
         "`python src/prefix_check.py`로 생성. 기준: `data/subset/labels.csv` + `split.csv`. 집계값만 기록.",
         "클래스 개수는 이미지 수 (한 이미지에 병이 여러 개면 각 클래스에 셈, 정상 = 병 없음).", "",
         "## 요약", "",
         f"- 이미지 {n_img:,}장, 접두어 {len(groups):,}개",
         f"- 클래스가 하나뿐인 접두어: {len(single)}/{len(groups)}개 ({len(single) / len(groups):.1%}), "
         f"그 접두어의 이미지 {single_img:,}장 ({single_img / n_img:.1%})",
         "- 기본 분할(split.csv)에서 접두어가 걸친 split 수: "
         + ", ".join(f"{k}곳 {span_cnt[k]}개" for k in sorted(span_cnt)), ""]

    single_by_class = Counter(classes_of(groups[p][0])[0] for p in single)
    L += ["클래스가 하나뿐인 접두어의 클래스: " + (", ".join(f"{c} {single_by_class[c]}개" for c in CLASSES if single_by_class[c]) or "없음"), ""]

    L += ["## 접두어별 클래스 개수 (기본 분할 기준)", "",
          table(["접두어", "이미지", *CLASSES, "train/val/test"],
                class_rows(groups, lambda p: ["/".join(str(sum(split[r["file"]] == s for r in groups[p])) for s in ["train", "val", "test"])])), ""]

    pp = sub / "split_prefix.csv"
    if pp.exists():
        sp = {r["file"]: r["split"] for r in read(pp)}
        test = defaultdict(list)
        for f, s in sp.items():
            if s == "test":
                test[prefix_of(f)].append(labels[f])
        tot = Counter(c for rs in test.values() for r in rs for c in classes_of(r))
        n_test = sum(len(v) for v in test.values())
        n_pre = len({prefix_of(f) for f in sp})
        L += ["## prefix384 test의 접두어와 클래스 구성", "",
              f"`split_prefix.csv` 기준 test {n_test}장, 접두어 {len(test)}개 (전체 접두어 {n_pre}개 중).", "",
              table(["접두어", "이미지", *CLASSES], class_rows(test)
                    + [["**합계**", n_test, *(tot[c] for c in CLASSES)]]), ""]
    else:
        L += ["## prefix384 test의 접두어와 클래스 구성", "", "`split_prefix.csv`가 없어 건너뜀 (sweep의 prefix384를 먼저 실행).", ""]

    md = "\n".join(L)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
