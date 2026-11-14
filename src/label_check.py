"""VS.zip의 유충_석고병/유충_부저병 폴더 이미지 각 N장 -> 대응 VL.zip JSON의 category_id 교차표 -> docs/label_check.md

실행: python src/label_check.py [--vs ...] [--vl ...] [--n 50] [--seed 0]
이미지와 JSON은 파일명(확장자 제외)으로 매칭한다. data/는 읽기만 한다.
"""
import argparse
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ["유충_석고병", "유충_부저병"]
CAT = {2: "석고병", 3: "부저병"}
COLS = ["2만", "3만", "2+3", "기타 포함", "JSON 없음"]
IMG_EXTS = (".jpg", ".jpeg", ".png")


def load(z, name):
    raw = z.read(name)
    k = raw.find(b'"environment"')  # 맨 끝 센서 시계열(~2MB)은 잘라내고 파싱
    return json.loads(raw[:k].rstrip().rstrip(b",") + b"}" if k >= 0 else raw)


def stem(path):
    return path.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def run(vs_path, vl_path, n, seed):
    with zipfile.ZipFile(vs_path) as vs:
        imgs = {f: sorted(p for p in vs.namelist()
                          if p.lower().endswith(IMG_EXTS) and f in p.split("/"))
                for f in FOLDERS}
    rng = random.Random(seed)
    picked = {f: rng.sample(v, min(n, len(v))) for f, v in imgs.items()}

    img_tab = {f: Counter() for f in FOLDERS}
    box_tab = {f: Counter() for f in FOLDERS}
    with zipfile.ZipFile(vl_path) as vl:
        by_stem = {stem(p): p for p in vl.namelist() if p.endswith(".json")}
        for f, paths in picked.items():
            for p in paths:
                j = by_stem.get(stem(p))
                if j is None:
                    img_tab[f]["JSON 없음"] += 1
                    continue
                ids = [a.get("category_id") for a in load(vl, j).get("annotations", [])]
                box_tab[f].update(ids)
                s = set(ids)
                key = "2만" if s == {2} else "3만" if s == {3} else "2+3" if s == {2, 3} else "기타 포함"
                img_tab[f][key] += 1
    return imgs, picked, img_tab, box_tab


def report(imgs, picked, img_tab, box_tab, n, seed):
    other = sorted({k for t in box_tab.values() for k in t if k not in CAT}, key=str)
    L = ["# 폴더명 vs JSON category_id 확인 (VS.zip 표본)", "",
         f"`python src/label_check.py --n {n} --seed {seed}`로 생성. "
         "VS.zip 폴더별 이미지를 무작위로 골라, 파일명이 같은 VL.zip JSON의 annotations[].category_id를 집계.",
         "category_id: 2=유충_석고병, 3=유충_부저병.", "",
         "## 이미지 단위 (이미지 안 category_id 조합)", "",
         "| VS 폴더 | 표본/전체 | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
    for f in FOLDERS:
        L.append(f"| {f} | {len(picked[f])}/{len(imgs[f]):,} | " + " | ".join(str(img_tab[f][c]) for c in COLS) + " |")
    cols = [2, 3] + other
    L += ["", "## 박스 단위 (category_id별 박스 수)", "",
          "| VS 폴더 | " + " | ".join(f"{c}({CAT.get(c, '기타')})" for c in cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for f in FOLDERS:
        L.append(f"| {f} | " + " | ".join(str(box_tab[f][c]) for c in cols) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vs", default=ROOT / "data/01.원천데이터/VS.zip")
    ap.add_argument("--vl", default=ROOT / "data/02.라벨링데이터/VL.zip")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=ROOT / "docs/label_check.md")
    a = ap.parse_args()
    res = run(a.vs, a.vl, a.n, a.seed)
    md = report(*res, a.n, a.seed)
    Path(a.out).write_text(md, encoding="utf-8")
    print(md)
