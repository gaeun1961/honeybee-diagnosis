"""Grad-CAM 최고점이 라벨 박스 안에 드는 비율 (test) -> docs/cam_check.md

실행: python src/cam_check.py [--weights models/best.pt]
- predict.grad_cam 그대로 사용: 확률 최고 클래스 기준 (서비스 화면과 같은 조건)
- 최고 클래스가 정답 양성인 test 이미지만 집계. 그 클래스의 박스 중 하나라도 최고점을 포함하면 적중
- 좌표는 정규화해서 비교 (cam 칸 중심 / bbox [x1,y1,x2,y2] / JSON image width·height)
- 우연 기준선 = 그 클래스 박스 넓이 비율 합(최대 1)의 평균. 최고점을 무작위로 찍었을 때의 적중률 근사 (박스 겹침은 무시)
VL.zip은 풀지 않고 읽는다. 히트맵·이미지는 저장하지 않는다.
"""
import argparse
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from label_check import load, stem
from predict import grad_cam, load_model, preprocess
from train import LABELS, load_rows

ROOT = Path(__file__).resolve().parent.parent
CAT = {1: "응애", 2: "석고병", 3: "부저병"}


def peak_xy(cam):
    """cam 최고점 칸의 중심 -> 정규화 (x, y)"""
    i, j = np.unravel_index(int(np.argmax(cam)), cam.shape)
    h, w = cam.shape
    return (j + 0.5) / w, (i + 0.5) / h


def inside(pt, box):
    x, y = pt
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def boxes_by_class(d):
    """JSON -> {클래스명: [정규화 bbox]}"""
    img = d.get("image") or {}
    w, h = img.get("width"), img.get("height")
    out = {}
    for a in d.get("annotations", []):
        name, b = CAT.get(a.get("category_id")), a.get("bbox")
        if name and b and w and h:
            out.setdefault(name, []).append([b[0] / w, b[1] / h, b[2] / w, b[3] / h])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models/best.pt")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--vl", type=Path, default=ROOT / "data/02.라벨링데이터/VL.zip")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/cam_check.md")
    a = ap.parse_args()

    model, size = load_model(a.weights)
    tf = preprocess(size)
    sub = a.data_dir / "subset"
    test = load_rows(sub)["test"]
    recs, skipped = [], 0  # (클래스, 접두어, 적중, 우연)
    with zipfile.ZipFile(a.vl) as vl:
        by_stem = {stem(p): p for p in vl.namelist() if p.endswith(".json")}
        for r in test:
            img = Image.open(sub / "images" / r["file"]).convert("RGB")
            _, top, cam = grad_cam(model, tf(img).unsqueeze(0))
            c = LABELS[top]
            boxes = boxes_by_class(load(vl, by_stem[stem(r["file"])])).get(c) if int(r[c]) else None
            if not boxes:
                skipped += 1
                continue
            pt = peak_xy(cam)
            chance = min(1.0, sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes))
            recs.append((c, r["file"][0], any(inside(pt, b) for b in boxes), chance))

    rows = []
    for c in LABELS:
        for p in sorted({x[1] for x in recs}) + ["전체"]:
            m = [x for x in recs if x[0] == c and (p == "전체" or x[1] == p)]
            hit = sum(x[2] for x in m)
            rows.append([c, p, len(m), hit, f"{hit / len(m):.3f}" if m else "-",
                         f"{np.mean([x[3] for x in m]):.3f}" if m else "-"])
    hit = sum(x[2] for x in recs)
    rows.append(["**전체**", "전체", len(recs), hit, f"{hit / len(recs):.3f}" if recs else "-",
                 f"{np.mean([x[3] for x in recs]):.3f}" if recs else "-"])
    L = ["# Grad-CAM 최고점 박스 적중률 (test)", "",
         f"`python src/cam_check.py`로 생성. 가중치 `{a.weights.name}` (입력 {size}px), test {len(test)}장 중 "
         f"확률 최고 클래스가 정답 양성인 {len(recs)}장만 집계 (나머지 {skipped}장 제외).",
         "적중 = layer4 Grad-CAM 최고점(칸 중심)이 그 클래스 라벨 박스 중 하나 안에 듦. "
         "우연 기준선 = 박스 넓이 비율 합의 평균 (무작위 점이 박스에 들 확률).", "",
         "| 클래스 | 접두어 | 장수 | 적중 | 적중률 | 우연 기준선 |", "|---|---|---|---|---|---|"]
    L += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("wrote", a.out)


def _selfcheck():
    cam = np.zeros((4, 8))
    cam[1, 6] = 1
    assert peak_xy(cam) == (6.5 / 8, 1.5 / 4)
    assert inside((0.5, 0.5), [0.4, 0.4, 0.6, 0.6]) and not inside((0.7, 0.5), [0.4, 0.4, 0.6, 0.6])
    d = {"image": {"width": 200, "height": 100},
         "annotations": [{"category_id": 2, "bbox": [20, 10, 100, 50]}, {"category_id": 5, "bbox": [0, 0, 1, 1]}]}
    assert boxes_by_class(d) == {"석고병": [[0.1, 0.1, 0.5, 0.5]]}


if __name__ == "__main__":
    _selfcheck()
    main()
