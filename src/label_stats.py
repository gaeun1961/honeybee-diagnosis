"""VL.zip 라벨 집계 -> docs/label_stats.md. 압축을 풀지 않고 JSON을 하나씩 읽는다.

실행: python src/label_stats.py [--zip data/02.라벨링데이터/VL.zip] [--out docs/label_stats.md]
bbox는 [x1, y1, x2, y2] (area == (x2-x1)*(y2-y1)로 확인함).
"""
import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BBOX_TOL = 1.0  # area 필드와 bbox 계산값 허용 오차(px^2)


def pct(vals, qs=(0, 5, 25, 50, 75, 95, 100)):
    s = sorted(vals)
    if not s:
        return ["-"] * len(qs)
    return [f"{s[round(q / 100 * (len(s) - 1))]:.3f}" for q in qs]


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def collect(zip_path):
    s = {
        "files": 0,
        "folder_imgs": Counter(), "folder_boxes": Counter(),   # 폴더명 기준
        "json_imgs": Counter(), "json_boxes": Counter(),       # JSON category_id 기준
        "ratios": defaultdict(list), "per_img": defaultdict(list),
        "res": Counter(), "meta": defaultdict(Counter),
        "anom": defaultdict(list), "cross": Counter(), "sym": Counter(), "combos": Counter(), "cats": None, "cats_diff": 0, "env_keys": None,
    }

    def anom(kind, name):
        s["anom"][kind].append(name)

    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            parts = name.split("/")
            s["files"] += 1
            raw = z.read(info)
            if s["env_keys"] is None:
                s["env_keys"] = list(json.loads(raw)["environment"])
            # environment(센서 시계열)는 JSON 맨 끝 키이고 파일당 ~2MB라 파싱이 느림 -> 그 앞에서 자름
            k = raw.find(b'"environment"')
            d = json.loads(raw[:k].rstrip().rstrip(b",") + b"}" if k >= 0 else raw)
            if k < 0:
                anom("environment 키 없음", name)
            folder_cls = parts[1] if len(parts) > 1 else "?"
            if parts[0] != "성충" and parts[0] != "유충":
                anom("상위 폴더가 성충/유충 아님", name)

            cats = {c["id"]: c["name"] for c in d.get("categories", [])}
            if s["cats"] is None:
                s["cats"] = cats
            elif cats != s["cats"]:
                s["cats_diff"] += 1

            img = d.get("image") or {}
            w, h = img.get("width"), img.get("height")
            if not (w and h):
                anom("image 크기 누락", name)
                w = h = None
            else:
                s["res"][(w, h)] += 1
            if img.get("filename", "")[:-4] != parts[-1][:-5]:
                anom("image.filename != JSON 파일명", name)
            s["meta"]["hive"][img.get("hive")] += 1
            for k in ("weather", "device", "resolution"):
                s["meta"][f"collection.{k}"][(d.get("collection") or {}).get(k)] += 1
            for k in ("type", "floor", "plate_num"):
                s["meta"][f"colony.{k}"][(d.get("colony") or {}).get(k)] += 1
            s["meta"]["파일명 접두(첫 글자)"][parts[-1][:1]] += 1

            anns = d.get("annotations")
            if anns is None:
                anom("annotations 키 누락", name)
                anns = []
            if not anns:
                anom("라벨 없음(빈 annotations)", name)
            s["folder_imgs"][folder_cls] += 1
            s["folder_boxes"][folder_cls] += len(anns)
            s["per_img"][folder_cls].append(len(anns))

            img_cls = set()
            for a in anns:
                cid = a.get("category_id")
                cname = cats.get(cid)
                if cname is None:
                    anom("category_id가 categories에 없음", name)
                    cname = f"?{cid}"
                img_cls.add(cname)
                s["json_boxes"][cname] += 1
                s["cross"][(folder_cls, cname)] += 1
                s["sym"][(cname, a.get("symptoms"))] += 1
                if cname != folder_cls:
                    anom("폴더명과 박스 클래스 불일치", name)
                s["meta"]["state"][a.get("state")] += 1
                s["meta"]["symptoms"][a.get("symptoms")] += 1
                b = a.get("bbox")
                if not b or len(b) != 4:
                    anom("bbox 누락/형식 이상", name)
                    continue
                x1, y1, x2, y2 = b
                bw, bh = x2 - x1, y2 - y1
                if bw <= 0 or bh <= 0:
                    anom("빈 박스(너비/높이 <= 0)", name)
                    continue
                if abs(bw * bh - a.get("area", -1e9)) > BBOX_TOL + 0.01 * bw * bh:
                    anom("area != bbox 계산값", name)
                if w and (x1 < 0 or y1 < 0 or x2 > w or y2 > h):
                    anom("박스가 이미지 밖", name)
                if w:
                    s["ratios"][cname].append(bw * bh / (w * h))
            for c in img_cls:
                s["json_imgs"][c] += 1
            s["combos"][(folder_cls, tuple(sorted(img_cls)))] += 1
            if len(img_cls) > 1:
                anom("한 이미지에 여러 클래스", name)
    return s


def report(s, zip_path):
    L = []
    classes = sorted(s["folder_imgs"], key=lambda c: -s["folder_imgs"][c])
    n = s["files"]
    L += ["# 라벨 통계 (Validation, VL.zip)", "",
          f"`python src/label_stats.py`로 생성. 대상: `{zip_path}`, JSON {n:,}개 (이미지 1장 = JSON 1개).",
          "bbox는 `[x1,y1,x2,y2]`(area 필드로 검증). 박스 비율 = 박스 넓이 / 이미지 넓이.", ""]

    L += ["## 클래스별 이미지·박스 수 (폴더명 vs JSON 기준)", ""]
    nb = sum(s["folder_boxes"].values())
    rows = []
    for c in classes:
        rows.append([c, f"{s['folder_imgs'][c]:,}", f"{s['folder_imgs'][c] / n:.1%}",
                     f"{s['json_imgs'].get(c, 0):,}", f"{s['folder_boxes'][c]:,}",
                     f"{s['folder_boxes'][c] / nb:.1%}", f"{s['json_boxes'].get(c, 0):,}"])
    rows.append(["**합계**", f"{n:,}", "100%", "", f"{nb:,}", "100%", f"{sum(s['json_boxes'].values()):,}"])
    L += [table(["클래스", "이미지(폴더)", "비율", "이미지(JSON)", "박스(폴더)", "비율", "박스(JSON)"], rows), ""]
    larva = [c for c in classes if c.startswith("유충")]
    nl = sum(s["folder_imgs"][c] for c in larva)
    L += [f"유충 4클래스 존재 여부: {sorted(larva) == sorted(['유충_정상', '유충_응애', '유충_부저병', '유충_석고병'])}. "
          f"유충 이미지 {nl:,}장({nl / n:.1%}), 성충 {n - nl:,}장.", "",
          "유충만 본 비율:", "",
          table(["클래스", "이미지", "유충 내 비율"],
                [[c, f"{s['folder_imgs'][c]:,}", f"{s['folder_imgs'][c] / nl:.1%}"] for c in larva]), ""]

    jc = sorted(s["json_boxes"], key=lambda c: -s["json_boxes"][c])
    L += ["## 폴더명 vs JSON 클래스 (박스 수, 행=폴더, 열=JSON category)", "",
          table(["폴더 / JSON"] + jc, [[f] + [f"{s['cross'].get((f, c), 0):,}" for c in jc] for f in classes]), "",
          "## 폴더별 이미지의 JSON 클래스 조합 (이미지 수, 상위 25)", "",
          table(["폴더", "이미지 내 클래스 조합", "이미지 수"],
                [[f, " + ".join(c), f"{v:,}"] for (f, c), v in s["combos"].most_common(25)]), ""]

    sy = sorted({k[1] for k in s["sym"]}, key=str)
    L += ["## JSON category vs symptoms 필드 (박스 수)", "",
          table(["JSON category / symptoms"] + [str(x) for x in sy],
                [[c] + [f"{s['sym'].get((c, x), 0):,}" for x in sy] for c in jc]), ""]

    L += ["## 이미지당 박스 수 (폴더 기준)", ""]
    rows = []
    for c in classes:
        v = s["per_img"][c]
        cnt = Counter(v)
        rows.append([c, f"{sum(v) / len(v):.2f}", max(v), ", ".join(f"{k}:{cnt[k]}" for k in sorted(cnt)[:6])])
    L += [table(["클래스", "평균", "최대", "분포(박스수:이미지수, 앞 6개)"], rows), ""]

    L += ["## 박스 크기 (이미지 대비 넓이 비율, JSON 클래스 기준)", "",
          table(["클래스", "박스 수", "min", "5%", "25%", "중앙값", "75%", "95%", "max"],
                [[c, f"{len(s['ratios'][c]):,}"] + pct(s["ratios"][c]) for c in classes]), ""]

    L += ["## 해상도", "",
          table(["가로x세로", "이미지 수"], [[f"{w}x{h}", f"{v:,}"] for (w, h), v in s["res"].most_common()]), ""]

    L += ["## 메타 정보 (파일명/JSON)", ""]
    for k, c in s["meta"].items():
        L += [f"**{k}**: " + ", ".join(f"{a} {b:,}" for a, b in c.most_common(10))
              + (f" 외 {len(c) - 10}종" if len(c) > 10 else ""), ""]

    L += [f"환경 센서 키(첫 파일 기준, 시계열이라 집계 제외): {', '.join(s['env_keys'] or [])}", ""]

    L += ["## 이상값", "", f"- categories 목록이 첫 파일과 다른 JSON: {s['cats_diff']:,}개"]
    if not s["anom"]:
        L += ["- 그 외 이상값 없음"]
    for k, v in sorted(s["anom"].items(), key=lambda kv: -len(kv[1])):
        L += [f"- {k}: {len(v):,}건 (예: `{v[0]}`)"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ROOT / "data/02.라벨링데이터/VL.zip")
    ap.add_argument("--out", default=ROOT / "docs/label_stats.md")
    a = ap.parse_args()
    stats = collect(a.zip)
    Path(a.out).write_text(report(stats, Path(a.zip).relative_to(ROOT).as_posix()
                                  if Path(a.zip).is_relative_to(ROOT) else a.zip), encoding="utf-8")
    print("wrote", a.out)
