# honeybee-diagnosis

꿀벌 유충 사진 1장으로 질병 확률을 내고 병변 위치를 표시하는 개인 포트폴리오 프로젝트입니다. (AI Hub 꿀벌 질병 진단 이미지, CPU 학습)

**데모**: 로컬 서버(`src/serve.py`)를 켜고 `web/index.html`에서 사진을 올리면 응애·석고병·부저병 확률과 Grad-CAM 히트맵을 보여줍니다.

## MVP
- 유충 사진 다중 라벨 분류: 응애·석고병·부저병 확률을 각각 출력하고, 병이 없으면(모두 0.5 미만) 정상
- Grad-CAM으로 확률이 가장 높은 병의 병변 위치를 히트맵으로 표시
- 모델: ImageNet 사전학습 ResNet18 전이학습 (CPU)

## 실행 순서

설치 (Windows). CPU용 torch를 먼저 설치하고 나머지 패키지를 설치합니다. `pip install -r requirements.txt`로 한 번에 설치하면 CPU 전용이 아닌 torch가 설치될 수 있으니 쓰지 마세요.
```
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pandas scikit-learn pillow matplotlib jupyter fastapi uvicorn python-multipart
```

데이터를 `data/`에 둔 뒤([데이터](#데이터) 참고) 순서대로 실행합니다.
```
python src/make_subset.py                       # 유충 라벨 생성 + 클래스별 표본 → data/subset/
python src/split.py                             # 50장 묶음 단위 train/val/test 분할
python src/train.py --size 512                  # 학습 → models/best.pt, docs/baseline.md
python src/evaluate.py                          # 임계값 조정·오답 분석 → docs/eval_errors.md
python src/serve.py                             # 진단 API (http://127.0.0.1:8000)
python -m http.server 5500 --directory web      # 다른 터미널에서 웹 페이지 서버
```
512px 학습은 384px보다 입력 픽셀이 약 1.8배 많아 CPU에서 더 오래 걸립니다. 빨리 돌려 보려면 `--size 384`(train.py 기본값)로 학습하고 `serve.py --size 384`로 맞추세요.

브라우저에서 http://localhost:5500 을 열고 "서비스 미리보기"에서 사진을 올립니다.

## 성능

test 200장 기준 평균 AP는 384px 입력에서 0.898, 512px 입력에서 0.916입니다. 두 결과의 차이는 아래 [한계](#한계)의 점수 흔들림(±0.05) 범위 안입니다.

384px (`docs/baseline_384.md`)

| 클래스 | test 양성 수 | AP | F1@0.5 |
|---|---|---|---|
| 응애 | 66 | 0.983 | 0.985 |
| 석고병 | 67 | 0.926 | 0.780 |
| 부저병 | 48 | 0.785 | 0.780 |
| **평균** | 200장 | 0.898 | 0.848 |

512px (`docs/baseline.md`)

| 클래스 | test 양성 수 | AP | F1@0.5 |
|---|---|---|---|
| 응애 | 66 | 0.986 | 0.985 |
| 석고병 | 67 | 0.924 | 0.803 |
| 부저병 | 48 | 0.839 | 0.773 |
| **평균** | 200장 | 0.916 | 0.854 |

둘 다 `train.py --epochs 8 --batch 16 --lr 0.001`, train 1,603 / val 200 / test 200장, val 평균 AP가 가장 높은 epoch의 가중치로 test를 평가했습니다.

## 한계
- 학습 표본이 2,003장, 테스트가 200장뿐이라 점수가 ±0.05 정도 흔들립니다.
- 석고병과 부저병을 서로 혼동합니다.
- 출력 확률은 교정(calibration)하지 않았습니다. 0.8이 "80% 확률"을 뜻하지 않습니다.
- 진단 도구가 아닙니다. 참고용입니다.
- 파일명 접두어(벌통으로 추정) 단위의 분할은 검증하지 않았습니다. 지금은 연속 50장 묶음 단위로 나누므로 같은 벌통 사진이 train과 test에 함께 들어갈 수 있습니다.

## 데이터
- 데이터는 이 저장소에 없습니다. [AI Hub 「꿀벌 질병 진단 이미지」(dataSetSn=71667)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71667)에서 직접 받으세요.
- Validation 데이터만 사용합니다: `data/01.원천데이터/VS.zip`(사진), `data/02.라벨링데이터/VL.zip`(라벨)
- zip은 압축을 풀지 않고 코드가 zipfile로 직접 읽습니다.
- 폴더명의 석고병·부저병이 서로 바뀌어 있어, 라벨은 폴더명이 아니라 JSON의 `category_id`(1=응애, 2=석고병, 3=부저병)를 기준으로 합니다. 근거: [docs/label_check.md](docs/label_check.md)
- AI Hub 이용 약관상 원본 사진은 재배포할 수 없습니다. 그래서 이 저장소와 README에는 원본 사진이나 그 사진으로 만든 이미지를 넣지 않습니다.

## 폴더 구조
```
data/        데이터 (git 제외)
models/      학습 가중치 best.pt (git 제외)
out/         predict.py 결과 그림 (git 제외)
notebooks/   01_data_check.ipynb: 라벨 zip 구조 확인
src/
  label_stats.py   VL.zip 라벨 전체 통계 → docs/label_stats.md
  label_check.py   폴더명 vs JSON category_id 확인 → docs/label_check.md
  make_subset.py   유충 다중 라벨 생성 + 클래스별 표본 → data/subset/
  split.py         50장 묶음 단위 train/val/test 분할
  train.py         ResNet18 다중 라벨 학습
  evaluate.py      임계값 조정, 라벨 조합별·박스 크기별 오답 분석
  predict.py       사진 1장 확률 + Grad-CAM (명령줄)
  serve.py         진단 API (FastAPI)
web/         index.html: 소개 페이지 + 서비스 미리보기
docs/        라벨 분석, 베이스라인 결과
```
