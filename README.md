# honeybee-diagnosis

폰으로 찍은 꿀벌 유충 사진 1장으로 질병을 분류하고 병변 위치를 표시하는 웹 서비스 (개인 포트폴리오).

## 배경
- AI Hub "꿀벌 질병 진단 이미지" 데이터 사용. 학교 기계학습·영상처리 수업의 개인 후속 활동.
- **MVP**: 유충 4클래스(정상 / 응애 / 부저병 / 석고병) 분류. 병변 위치 표시(검출)는 분류가 된 뒤에 진행.
- **환경**: 윈도우 노트북, GPU 없음, RAM 16GB. 소형 사전학습 모델 전이학습을 CPU로 학습하고, 필요하면 Colab 사용.
- **데이터**: Validation 파일(사진 `VS.zip`, 라벨 `VL.zip`)만 사용.

## 폴더 구조
```
data/       데이터 (git 제외)
notebooks/  실험·분석 노트북
src/        학습·추론 코드
web/        웹 서비스
docs/       문서
```

## 시작하기
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 데이터
데이터는 이 저장소에 없습니다. [AI Hub](https://aihub.or.kr)에서 직접 받아 `data/`에 두세요
(원본 사진 재배포는 이용 약관상 금지).
`VS.zip` → `data/images/`, `VL.zip` → `data/labels/` 에 압축 해제.
