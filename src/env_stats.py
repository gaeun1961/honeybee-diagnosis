"""유충 라벨 JSON의 environment(센서 시계열) 집계 -> docs/env_stats.md

실행: python src/env_stats.py [--vl data/02.라벨링데이터/VL.zip]
VL.zip은 압축을 풀지 않고 zipfile로 하나씩 읽는다. 문서에는 집계값만 쓰고 원본 값·이미지 경로는 쓰지 않는다.

- 필드별 존재율(키 있음), 빈 배열 비율, 원소 결측(null/NaN) 비율, 이미지당 시계열 길이
- 이미지마다 필드별 시계열 평균을 구한 뒤
  - category_id별(그 번호가 있는 이미지) 평균·표준편차, 병 번호(1~3)가 없는 이미지는 "병 없음"
  - 파일명 접두어(앞 3칸, 벌통 추정)별 평균과 접두어 안 표준편차 (같은 접두어 안에서 값이 거의 같은지)
"""
import argparse
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from label_check import stem

ROOT = Path(__file__).resolve().parent.parent
CATS = {0: "유충_정상", 1: "유충_응애", 2: "유충_석고병", 3: "유충_부저병"}
PREFIX_PARTS = 3
MAX_PREFIX_ROWS = 60
ARRAY_RE = re.compile(rb'"(\w+)"\s*:\s*\[([^\]]*)\]')


def parse_env(raw):
    """environment 부분만 빠르게 파싱 -> {필드: (값 배열, 결측 수)}. 없으면 None"""
    k = raw.find(b'"environment"')
    if k < 0:
        return None
    env = {}
    for key, body in ARRAY_RE.findall(raw[k:]):
        body = body.strip()
        if not body:
            env[key.decode()] = (np.empty(0), 0)
            continue
        if b"null" in body or b"NaN" in body:  # 결측이 섞이면 느린 경로
            vals = [v for v in json.loads(b"[" + body + b"]")]
            arr = np.array([math.nan if v is None else float(v) for v in vals])
        else:
            arr = np.array([float(v) for v in body.split(b",")])
        env[key.decode()] = (arr[~np.isnan(arr)], int(np.isnan(arr).sum()))
    return env


def prefix_of(name):
    return "_".join(stem(name).split("_")[:PREFIX_PARTS])


def collect(vl_path):
    n = 0
    present, empty, missing_el, total_el = Counter(), Counter(), Counter(), Counter()
    lengths = defaultdict(list)
    img_means = []  # (category set, prefix, {필드: 평균})
    no_env = 0
    with zipfile.ZipFile(vl_path) as z:
        names = [p for p in z.namelist() if p.endswith(".json") and p.split("/")[0] == "유충"]
        for i, name in enumerate(names):
            raw = z.read(name)
            env = parse_env(raw)
            k = raw.find(b'"environment"')  # label_check.load와 같은 방식, 한 번 읽은 raw를 재사용
            d = json.loads(raw[:k].rstrip().rstrip(b",") + b"}" if k >= 0 else raw)
            cats = {a.get("category_id") for a in d.get("annotations", [])}
            n += 1
            if env is None:
                no_env += 1
                continue
            means = {}
            for f, (arr, miss) in env.items():
                present[f] += 1
                lengths[f].append(len(arr) + miss)
                missing_el[f] += miss
                total_el[f] += len(arr) + miss
                if len(arr) + miss == 0:
                    empty[f] += 1
                if len(arr):
                    means[f] = float(arr.mean())
            img_means.append((cats, prefix_of(name), means))
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1:,}/{len(names):,}", flush=True)
    return dict(n=n, no_env=no_env, present=present, empty=empty, missing_el=missing_el, total_el=total_el,
                lengths=lengths, img_means=img_means)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
                     + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def ms(vals):
    if not vals:
        return "-"
    a = np.array(vals)
    return f"{a.mean():.2f} ± {a.std():.2f}"


def report(s, vl_rel):
    fields = sorted(s["present"], key=lambda f: (not f.startswith("in_"), f))
    n = s["n"]
    L = ["# 유충 environment(센서 시계열) 집계", "",
         f"`python src/env_stats.py`로 생성. 대상: `{vl_rel}`의 유충 JSON {n:,}개. 집계값만 기록 (원본 값·이미지 경로 없음).",
         f"environment 키가 없는 JSON: {s['no_env']:,}개. 필드 이름은 JSON 그대로 (예: in_carbon_monoxide는 이름상 일산화탄소).", "",
         "## 필드별 존재율·결측률", "",
         table(["필드", "존재율 (키 있음)", "빈 배열", "원소 결측 (null/NaN)"],
               [[f, f"{s['present'][f] / n:.1%}", f"{s['empty'][f] / n:.1%}",
                 f"{s['missing_el'][f] / s['total_el'][f]:.2%}" if s["total_el"][f] else "-"] for f in fields]), "",
         "## 이미지당 시계열 길이", "",
         table(["필드", "최소", "중앙값", "최대", "가장 흔한 길이 (이미지 수)"],
               [[f, min(s["lengths"][f]), int(np.median(s["lengths"][f])), max(s["lengths"][f]),
                 ", ".join(f"{k} ({v:,})" for k, v in Counter(s["lengths"][f]).most_common(3))] for f in fields]), ""]

    L += ["## category_id별 평균 ± 표준편차", "",
          "이미지마다 필드별 시계열 평균을 구하고, 그 category_id가 있는 이미지끼리 평균·표준편차를 냄 (한 이미지가 여러 번호에 들어갈 수 있음). "
          "병 없음 = 병 번호(1~3)가 없는 이미지.", ""]
    groups = [(f"{cid} {name}", lambda c, cid=cid: cid in c) for cid, name in CATS.items()]
    groups.append(("병 없음", lambda c: not c & {1, 2, 3}))
    rows = []
    for gname, pred in groups:
        sel = [m for c, _, m in s["img_means"] if pred(c)]
        rows.append([gname, f"{len(sel):,}"] + [ms([m[f] for m in sel if f in m]) for f in fields])
    L += [table(["category_id", "이미지 수", *fields], rows), ""]

    by_pre = defaultdict(list)
    for _, p, m in s["img_means"]:
        by_pre[p].append(m)
    overall = {f: np.std([m[f] for _, _, m in s["img_means"] if f in m]) for f in fields}
    within = {f: [np.std([m[f] for m in ms_ if f in m]) for ms_ in by_pre.values() if sum(f in m for m in ms_) >= 2]
              for f in fields}
    L += ["## 파일명 접두어(앞 3칸, 벌통 추정)별", "",
          f"접두어 {len(by_pre):,}개. 접두어 안 표준편차가 전체 표준편차보다 훨씬 작으면 같은 접두어끼리 센서 값이 거의 같다는 뜻.", "",
          table(["필드", "전체 이미지 간 표준편차", "접두어 안 표준편차 중앙값", "비율 (안/전체)"],
                [[f, f"{overall[f]:.3f}", f"{np.median(within[f]):.3f}" if within[f] else "-",
                  f"{np.median(within[f]) / overall[f]:.2f}" if within[f] and overall[f] else "-"] for f in fields]), ""]
    pres = sorted(by_pre, key=lambda p: -len(by_pre[p]))
    shown = pres[:MAX_PREFIX_ROWS]
    rows = []
    for p in shown:
        ms_ = by_pre[p]
        rows.append([p, len(ms_)] + [ms([m[f] for m in ms_ if f in m]) for f in fields])
    note = f" (이미지 수 상위 {MAX_PREFIX_ROWS}개만, 나머지 {len(pres) - len(shown)}개 생략)" if len(pres) > len(shown) else ""
    L += [f"접두어별 평균 ± 접두어 안 표준편차{note}:", "", table(["접두어", "이미지 수", *fields], rows)]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vl", type=Path, default=ROOT / "data/02.라벨링데이터/VL.zip")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/env_stats.md")
    a = ap.parse_args()
    s = collect(a.vl)
    rel = a.vl.resolve().relative_to(ROOT).as_posix() if a.vl.resolve().is_relative_to(ROOT) else a.vl.name
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(report(s, rel), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
