# 빅데이터 분석 과제 - 특성 공학 파이프라인

Titanic 데이터로 결측치 처리, 인코딩, 스케일링, 파생 변수, 변수 선택 방법에 따른 성능 차이를 비교한 과제입니다.

- 학번/이름: 2242973 이창민
- 데이터: Kaggle Titanic train.csv (891건)
- 보고서: `report/2242973_이창민_빅데이터분석_과제.pdf`

## 실행 방법

```
pip install -r requirements.txt
python 01_eda.py           # STEP 01~02 데이터 확인, EDA
python 02_experiments.py   # STEP 03~05 비교 실험, 변수 선택, 실험 비교표
python 03_bonus.py         # GridSearchCV, Feature Importance, SHAP, AutoML
```

결과 표는 `outputs/tables`, 그래프는 `outputs/figures`에 저장됩니다.
random_state는 42로 고정했고, AutoML(FLAML)은 시간 제한으로 탐색하기 때문에 실행 환경에 따라 결과가 조금 다를 수 있습니다.

## 파일 구성

- `src/fe.py` : 전처리 클래스, 파이프라인 생성 함수, 모델, 평가 함수
- `src/plot.py` : 그래프 한글 폰트 설정
- `01_eda.py` ~ `03_bonus.py` : 단계별 실행 코드
