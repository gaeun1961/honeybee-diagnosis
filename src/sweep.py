"""학습 실험 여러 개를 순서대로 실행하고 결과를 한 표로 모은다.

실행: python src/sweep.py [--only 이름 ...]
실험 (순서대로):
  base384     384px 기준 재현 (train.py 기본 설정)
  aug384      384px + 증강 강화 (회전 ±15도, 강한 색 변화)
  size512_e12 512px + 12 epoch
  prefix384   파일명 접두어(앞 3칸) 단위 분할 + 384px (분할: data/subset/split_prefix.csv)

- 가중치: models/sweep/{이름}/best.pt (models/best.pt는 건드리지 않음)
- 실험별 로그·test 보고서: out/sweep/{이름}.log, out/sweep/{이름}.md, 실패는 out/sweep/errors.log에도 기록
- 실험 하나가 실패해도 다음 실험을 계속하고, 결과 표 out/sweep_results.md를 실험이 끝날 때마다 갱신
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
LABELS = ["응애", "석고병", "부저병"]

EXPERIMENTS = [
    # 이름, 설명, train.py 인자, 분할 (None = 기존 split.csv)
    ("base384", "384px 기준 재현", ["--size", "384", "--epochs", "8"], None),
    ("aug384", "384px + 증강 강화(회전·색상)", ["--size", "384", "--epochs", "8", "--aug", "strong"], None),
    ("size512_e12", "512px + 12 epoch", ["--size", "512", "--epochs", "12"], None),
    ("prefix384", "접두어(앞 3칸) 단위 분할 + 384px", ["--size", "384", "--epochs", "8"], "prefix"),
]


def run(cmd, log):
    log.write(f"$ {' '.join(map(str, cmd))}\n")
    log.flush()
    r = subprocess.run([str(c) for c in cmd], stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        raise RuntimeError(f"{Path(cmd[1]).name} 종료 코드 {r.returncode}")


def write_results(path, results):
    L = ["# 학습 실험 비교 (sweep)", "",
         "`python src/sweep.py`로 생성. test 지표, 각 실험은 val 평균 AP가 가장 높은 epoch의 가중치로 평가.",
         "test가 200장 안팎이라 평균 AP는 ±0.05 정도 흔들릴 수 있음. prefix384는 분할이 달라 test 이미지도 다름.", "",
         "| 실험 | 설정 | " + " | ".join(f"{c} AP" for c in LABELS) + " | 평균 AP | best epoch | test 장수 | 상태 |",
         "|" + "---|" * (len(LABELS) + 6)]
    for name, desc, r in results:
        if "metrics" in r:
            m = r["metrics"]
            L.append(f"| {name} | {desc} | " + " | ".join(f"{v:.3f}" for v in m["ap"])
                     + f" | {m['map']:.3f} | {m['best_epoch']} | {m['n']} | 완료 ({r['min']:.0f}분) |")
        else:
            L.append(f"| {name} | {desc} | " + " | ".join("-" for _ in LABELS) + f" | - | - | - | 실패: {r['error']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=[e[0] for e in EXPERIMENTS], help="일부 실험만")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--root", type=Path, default=ROOT, help="models/sweep, out/을 쓸 위치")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="모든 train.py 호출 끝에 붙일 인자 (시험용)")
    a = ap.parse_args()

    py = sys.executable
    out_dir = a.root / "out/sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_md = a.root / "out/sweep_results.md"
    protected = (ROOT / "models/best.pt").resolve()
    results = []

    for name, desc, train_args, split_by in EXPERIMENTS:
        if a.only and name not in a.only:
            continue
        print(f"[{name}] {desc} 시작", flush=True)
        t = time.time()
        weights = a.root / "models/sweep" / name / "best.pt"
        metrics_path = out_dir / f"{name}.json"
        metrics_path.unlink(missing_ok=True)
        try:
            assert weights.resolve() != protected, "models/best.pt는 덮어쓰지 않음"
            with open(out_dir / f"{name}.log", "w", encoding="utf-8") as log:
                split_args = []
                if split_by:
                    split_csv = a.data_dir / "subset" / f"split_{split_by}.csv"
                    run([py, SRC / "split.py", "--data-dir", a.data_dir, "--by", split_by, "--out", split_csv], log)
                    split_args = ["--split", split_csv]
                run([py, SRC / "train.py", "--data-dir", a.data_dir, *train_args, *split_args,
                     "--weights-out", weights, "--report", out_dir / f"{name}.md",
                     "--metrics-json", metrics_path, *a.extra], log)
            r = {"metrics": json.loads(metrics_path.read_text(encoding="utf-8")), "min": (time.time() - t) / 60}
            print(f"[{name}] 완료: 평균 AP {r['metrics']['map']:.3f} ({r['min']:.0f}분)", flush=True)
        except Exception as e:
            r = {"error": str(e).replace("|", "/")[:120]}
            with open(out_dir / "errors.log", "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {name}: {e}\n{traceback.format_exc()}"
                        f"자세한 출력: {out_dir / (name + '.log')}\n\n")
            print(f"[{name}] 실패: {e} (로그: {out_dir / (name + '.log')})", flush=True)
        results.append((name, desc, r))
        write_results(results_md, results)

    print(f"결과 -> {results_md}")


if __name__ == "__main__":
    main()
