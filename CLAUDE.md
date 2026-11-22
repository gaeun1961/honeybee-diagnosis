# honeybee-diagnosis

폰으로 찍은 꿀벌 유충 사진 1장으로 질병을 분류하고 병변 위치를 표시하는 웹 서비스 (개인 포트폴리오).

- 데이터: AI Hub 「꿀벌 질병 진단 이미지」(dataSetSn=71667). **Validation만 사용** (사진 VS.zip, 라벨 VL.zip)
- MVP: 유충 이미지 다중 라벨 분류(응애·석고병·부저병 확률, 병 없음=정상). 라벨은 폴더명이 아니라 JSON category_id 기준. 병변 위치(검출)는 분류 다음 단계
- 환경: 윈도우 노트북, GPU 없음, RAM 16GB. CPU 전이학습, 필요하면 Colab

## 폴더
- `data/` 데이터 (git 제외, 직접 받아서 둠)
- `notebooks/` 실험·분석 노트북
- `src/` 학습·추론·집계 코드 (`label_stats.py` → `docs/label_stats.md`)
- `web/` 웹 서비스
- `docs/` 문서

## 규칙
- 답변은 한국어로
- git·PR: `.claude/rules/git-workflow.md`
- 데이터·비밀값: `.claude/rules/data-safety.md`
- 모르는 건 추측하지 말고 물어볼 것




## 작업 흐름 (docs/[tasks.md](http://tasks.md)가 있는 프로젝트만)

"다음 작업 진행해"를 받으면:

1. planner에게 다음 작업 지시문을 받는다.
2. GitHub 이슈 → 브랜치 생성 후 구현한다.
3. reviewer에게 검토를 맡긴다. "수정 필요"면 고치고 다시 맡긴다 (최대 2회, 이후 나에게 보고).
4. docs/[tasks.md](http://tasks.md)에 체크하고 PR 번호를 적는다.
5. 커밋 → PR (Closes #이슈번호).
6. 나에게 3줄로 보고: 한 일 / 리뷰 결과 / 다음 작업.

기획과 다르게 가야 할 것 같으면 구현 전에 멈추고 나에게 묻는다.

Notion 페이지: [https://app.notion.com/p/41673aa7bb8d81798746d99a8f5b7ced](https://app.notion.com/p/41673aa7bb8d81798746d99a8f5b7ced)

"일지 써"라고 하면 devlog에게 맡긴다. PR마다 자동으로 쓰지 않는다.