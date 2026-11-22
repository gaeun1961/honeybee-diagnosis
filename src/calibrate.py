"""확률 보정 (temperature scaling): val로 T를 맞추고 보정 전후 ECE·신뢰도 표 -> docs/calibration.md, T -> models/temperature.json

실행: python src/calibrate.py [--weights models/best.pt]
- 보정 확률 = sigmoid(logit / T). bias가 없어 logit 부호가 그대로 -> 0.5 판정·최고 클래스·AP는 안 바뀌고 확률 값만 바뀜
- T 두 방식: 전체 공통 1개, 클래스별 3개. 각각 val BCE 최소 T를 격자 탐색. test ECE 평균이 낮은 쪽을 json에 저장
- ECE: 클래스별, 등간격 10구간, sum(구간 장수/전체 * |평균 확률 - 양성 비율|)
- test는 측정만 (T 맞추기에 안 씀)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import infer, table
from predict import load_model
from train import LABELS, load_rows

ROOT = Path(__file__).resolve().parent.parent
GRID = np.exp(np.linspace(np.log(0.1), np.log(10), 401))  # T 후보 0.1~10 (로그 간격)
BINS = 10


def sig(z):
    return 1 / (1 + np.exp(-z))


def bce(y, z, t):
    p = np.clip(sig(z / t), 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit(y, z):
    """[공통 T]*3, [클래스별 T] 반환"""
    common = float(GRID[np.argmin([bce(y, z, t) for t in GRID])])
    per = [float(GRID[np.argmin([bce(y[:, k], z[:, k], t) for t in GRID])]) for k in range(y.shape[1])]
    return [common] * y.shape[1], per


def bins(y, p):
    """등간격 10구간 [(구간, 장수, 평균 확률, 양성 비율)] (빈 구간 제외)"""
    idx = np.minimum((p * BINS).astype(int), BINS - 1)
    return [(b, int((idx == b).sum()), p[idx == b].mean(), y[idx == b].mean()) for b in range(BINS) if (idx == b).any()]


def ece(y, p):
    return sum(n / len(y) * abs(mp - pos) for _, n, mp, pos in bins(y, p))


def apply(z, ts):
    return sig(z / np.array(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=ROOT / "models/best.pt")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/calibration.md")
    ap.add_argument("--t-out", type=Path, default=ROOT / "models/temperature.json")
    a = ap.parse_args()

    model, size = load_model(a.weights)
    sub = a.data_dir / "subset"
    rows = load_rows(sub)
    yv, zv = infer(model, sub / "images", rows["val"], size, a.batch, logits=True)
    yt, zt = infer(model, sub / "images", rows["test"], size, a.batch, logits=True)
    common, per = fit(yv, zv)
    ways = {"보정 전": [1.0] * len(LABELS), "공통 T": common, "클래스별 T": per}

    ece_rows, mean_test = [], {}
    for name, ts in ways.items():
        ev = [ece(yv[:, k], apply(zv, ts)[:, k]) for k in range(len(LABELS))]
        et = [ece(yt[:, k], apply(zt, ts)[:, k]) for k in range(len(LABELS))]
        mean_test[name] = np.mean(et)
        ece_rows.append([name, ", ".join(f"{t:.3f}" for t in ts),
                         *[f"{v:.3f} / {w:.3f}" for v, w in zip(ev, et)], f"{np.mean(ev):.3f} / {np.mean(et):.3f}"])
    chosen = min(["공통 T", "클래스별 T"], key=lambda k: mean_test[k])

    L = ["# 확률 보정 (temperature scaling)", "",
         f"`python src/calibrate.py`로 생성. 가중치 `{a.weights.name}` (입력 {size}px), val {len(yv)}장으로 T 맞춤, test {len(yt)}장은 측정만.",
         "보정 확률 = sigmoid(logit / T). 0.5 판정·AP는 그대로이고 확률 값만 바뀜 (최고 클래스는 공통 T일 때만 그대로).",
         *[f"주의: {c} 클래스별 T가 탐색 범위 끝({t:.1f})에 걸림 (val에서 거의 완전히 분리돼 BCE가 T를 끝까지 밀어냄)."
           for c, t in zip(LABELS, per) if t in (GRID[0], GRID[-1])],
         f"ECE: 클래스별, 등간격 {BINS}구간. 서버용으로 test ECE 평균이 낮은 **{chosen}** 를 `models/temperature.json`에 저장.",
         "한계: val은 체크포인트 선택·임계값 조정에도 쓰였으므로 val ECE는 낙관적일 수 있음. "
         "또 방식(공통/클래스별) 선택에 test ECE를 썼으므로 선택한 쪽의 test ECE도 낙관적일 수 있음. "
         "각 200장이라 ECE 차이 0.01 수준은 개선으로 단정하기 어려움.", "",
         "## ECE (val / test)", "",
         table(["방식", "T (응애, 석고병, 부저병)", *LABELS, "평균"], ece_rows), "",
         f"## 신뢰도 표 (test, 보정 전 → {chosen})", "",
         "빈 구간은 생략. 평균 확률과 양성 비율이 가까울수록 잘 맞음.", ""]
    for k, c in enumerate(LABELS):
        for name in ["보정 전", chosen]:
            L += [f"### {c} — {name}", "",
                  table(["구간", "장수", "평균 확률", "양성 비율"],
                        [[f"{b / BINS:.1f}~{(b + 1) / BINS:.1f}", n, f"{mp:.3f}", f"{pos:.3f}"]
                         for b, n, mp, pos in bins(yt[:, k], apply(zt, ways[name])[:, k])]), ""]
    a.out.write_text("\n".join(L), encoding="utf-8")
    a.t_out.parent.mkdir(parents=True, exist_ok=True)
    a.t_out.write_text(json.dumps({"labels": LABELS, "T": ways[chosen], "method": chosen}, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L[:12]))
    print("wrote", a.out, a.t_out)


def _selfcheck():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 3, (100, 3))
    for ts in ([0.3, 0.3, 0.3], [2.0, 0.5, 4.0]):  # 0.5 판정은 어떤 T든 그대로
        assert ((apply(z, ts) >= 0.5) == (z >= 0)).all()
    assert (apply(z, [2.0] * 3).argmax(1) == z.argmax(1)).all()  # 최고 클래스는 공통 T일 때만 보장
    y = np.array([0, 0, 1, 1])
    assert abs(ece(y, np.array([0.05, 0.05, 0.95, 0.95])) - 0.05) < 1e-9


if __name__ == "__main__":
    _selfcheck()
    main()
