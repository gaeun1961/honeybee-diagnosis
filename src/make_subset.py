"""유충 다중 라벨 학습용 표본 만들기.

실행: python src/make_subset.py [--per-class 600] [--width 960] [--seed 0] [--rebuild]

1단계: VS.zip의 유충 이미지마다 파일명이 같은 VL.zip JSON의 category_id로 라벨 생성 -> data/labels_larva.csv
       (1=응애, 2=석고병, 3=부저병, 병 번호 없음=정상. 0·4~6은 무시. 폴더명은 신뢰하지 않음: 석고병/부저병이 뒤바뀜)
       area_응애/석고병/부저병 = 클래스별 최대 박스 비율(JSON area / 이미지 넓이), 없으면 0. subset/labels.csv에도 전달.
       이미 있으면 재사용, --rebuild면 다시 만듦 (박스 비율 열이 없는 예전 파일이면 자동으로 다시 만듦).
2단계: 석고병 -> 부저병 -> 응애 -> 정상 순으로 클래스별 최대 N장 무작위 표본 (중복 없음, 앞에서 뽑힌 이미지도 그 클래스 몫으로 셈)
       정상 = 폴더가 유충_정상이고 병 번호가 없는 이미지.
       가로 --width로 줄여 data/subset/images/{stem}.jpg (JPEG 90), data/subset/labels.csv 작성.
       data/subset/은 매번 지우고 새로 만든다.

zip은 압축을 풀지 않고 읽기만 한다.
"""
import argparse
import csv
import io
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image

from label_check import load, stem

ROOT = Path(__file__).resolve().parent.parent
DISEASE = {1: "응애", 2: "석고병", 3: "부저병"}
ORDER = ["석고병", "부저병", "응애", "정상"]
IMG_EXTS = (".jpg", ".jpeg", ".png")
AREA_COLS = [f"area_{n}" for n in DISEASE.values()]  # 클래스별 최대 박스 비율 (JSON area / 이미지 넓이), 없으면 0


def larva_folder(path):
    return next((p for p in path.split("/")[:-1] if p.startswith("유충_")), None)


def build_labels(vs_path, vl_path, csv_path):
    rows, missing = [], 0
    with zipfile.ZipFile(vs_path) as vs, zipfile.ZipFile(vl_path) as vl:
        by_stem = {stem(p): p for p in vl.namelist() if p.endswith(".json")}
        for p in vs.namelist():
            folder = larva_folder(p)
            if folder is None or not p.lower().endswith(IMG_EXTS):
                continue
            j = by_stem.get(stem(p))
            if j is None:  # 라벨 없는 이미지는 정상으로 치지 않고 뺀다
                missing += 1
                continue
            d = load(vl, j)
            anns = d.get("annotations", [])
            ids = {a.get("category_id") for a in anns}
            img = d.get("image") or {}
            img_area = (img.get("width") or 0) * (img.get("height") or 0)
            area = {name: 0.0 for name in DISEASE.values()}
            for a in anns:
                name = DISEASE.get(a.get("category_id"))
                if name and img_area:
                    area[name] = max(area[name], (a.get("area") or 0) / img_area)
            rows.append({"zip_path": p, "stem": stem(p), "folder": folder,
                         **{name: int(cid in ids) for cid, name in DISEASE.items()},
                         "n_boxes": len(anns),
                         **{f"area_{n}": f"{v:.6f}" for n, v in area.items()}})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["zip_path", "stem", "folder", *DISEASE.values(), "n_boxes", *AREA_COLS])
        w.writeheader()
        w.writerows(rows)
    print(f"라벨 {len(rows):,}장 -> {csv_path}" + (f" (JSON 없는 이미지 {missing:,}장 제외)" if missing else ""))


def read_labels(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for name in DISEASE.values():
            r[name] = int(r[name])
    return rows


def has(r, cls):
    if cls == "정상":
        return r["folder"] == "유충_정상" and not any(r[n] for n in DISEASE.values())
    return r[cls] == 1


def pick(rows, per_class, seed):
    rng = random.Random(seed)
    chosen = {}  # stem -> row, 뽑힌 순서 유지
    for cls in ORDER:
        need = per_class - sum(has(r, cls) for r in chosen.values())
        pool = sorted((r for r in rows if has(r, cls) and r["stem"] not in chosen), key=lambda r: r["stem"])
        for r in rng.sample(pool, max(0, min(need, len(pool)))):
            chosen[r["stem"]] = r
    return list(chosen.values())


def write_subset(vs_path, sample, out_dir, width):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True)
    with zipfile.ZipFile(vs_path) as vs, open(out_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", *DISEASE.values(), *AREA_COLS])
        for r in sample:
            im = Image.open(io.BytesIO(vs.read(r["zip_path"]))).convert("RGB")
            if im.width > width:
                im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
            name = f"{r['stem']}.jpg"
            im.save(img_dir / name, "JPEG", quality=90)
            w.writerow([name, *(r[n] for n in DISEASE.values()), *(r[c] for c in AREA_COLS)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=600)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--vs", type=Path, default=ROOT / "data/01.원천데이터/VS.zip")
    ap.add_argument("--vl", type=Path, default=ROOT / "data/02.라벨링데이터/VL.zip")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data", help="labels_larva.csv, subset/을 쓸 위치")
    a = ap.parse_args()

    csv_path = a.data_dir / "labels_larva.csv"
    if not a.rebuild and csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            if not set(AREA_COLS) <= set(next(csv.reader(f))):
                print(f"{csv_path}에 박스 비율 열이 없어 다시 만듭니다")
                a.rebuild = True
    if a.rebuild or not csv_path.exists():
        build_labels(a.vs, a.vl, csv_path)
    else:
        print(f"기존 라벨 재사용: {csv_path} (다시 만들려면 --rebuild)")
    sample = pick(read_labels(csv_path), a.per_class, a.seed)
    write_subset(a.vs, sample, a.data_dir / "subset", a.width)

    print(f"표본 {len(sample):,}장 -> {a.data_dir / 'subset'}")
    for cls in ORDER:
        print(f"  {cls}: {sum(has(r, cls) for r in sample):,}장")
