# 매출 예측 자동화 시스템

월별 단어-점수 데이터와 시계열 매출 데이터를 학습하여 다음 달 매출을 예측하는 자동화 머신러닝 시스템입니다.

## 📋 목차

- [개요](#개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [설치 및 설정](#설치-및-설정)
- [데이터 형식](#데이터-형식)
- [사용 방법](#사용-방법)
- [모델 구조](#모델-구조)
- [자동화 프로세스](#자동화-프로세스)
- [설정 파일](#설정-파일)
- [문제 해결](#문제-해결)

## 🎯 개요

이 시스템은 다음과 같은 특징을 가집니다:

- **하이브리드 모델**: 시계열 LSTM과 텍스트 특성(Dense)을 결합한 딥러닝 모델
- **자동화 파이프라인**: 새 데이터가 들어오면 자동으로 모델 재학습 및 예측
- **유연한 입력**: 다양한 형식의 단어-점수 데이터 자동 처리
- **시계열 분석**: 과거 매출 패턴을 학습하여 미래 예측

### 핵심 가정

- **단어-점수 반영**: 각 월의 단어-점수 데이터는 해당 월의 매출에 영향을 미침
- **시계열 의존성**: 과거 매출 패턴이 미래 매출에 영향을 미침
- **자동 재학습**: 새로운 데이터가 추가되면 모델이 자동으로 재학습

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐
│  Input Data     │
│  (단어-점수)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Data Loader    │────▶│  Feature Eng.   │
│  (데이터 결합)   │     │  (특성 추출)     │
└────────┬────────┘     └────────┬────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐     ┌─────────────────┐
│  Sales Data     │     │  Time Series    │
│  (시계열 매출)   │     │  Features       │
└─────────────────┘     └─────────────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
         ┌──────────────────┐
         │  Hybrid Model    │
         │  (LSTM + Dense)  │
         └─────────┬────────┘
                   ▼
         ┌──────────────────┐
         │  Prediction      │
         │  (다음 달 매출)    │
         └──────────────────┘
```

## 🚀 설치 및 설정

### 1. 사전 요구사항

- Python 3.8 이상
- pip 패키지 관리자

### 2. 프로젝트 클론/다운로드

```bash
cd Demand_Analysis
```

### 3. 가상환경 설정

**Windows PowerShell:**

```powershell
# 가상환경 생성 및 패키지 설치
.\scripts\setup_venv.ps1

# 가상환경 활성화
.\scripts\activate.ps1
```

**Linux/Mac:**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. 패키지 설치 확인

```bash
pip list
```

필요한 주요 패키지:
- pandas, numpy
- tensorflow, keras
- scikit-learn
- pyyaml

## 📊 데이터 형식

### 1. 단어-점수 데이터 (`data/input_data/`)

월별 JSON 파일 형식:

**파일명**: `YYYY-MM.json` (예: `2024-01.json`)

**내용**:
```json
{
  "프로모션": 0.9,
  "할인": 0.8,
  "신제품": 0.7,
  "마케팅": 0.85
}
```

- **키**: 단어 (문자열)
- **값**: 점수 (0.0 ~ 1.0 사이의 실수)
- 각 월마다 별도 파일 생성

### 2. 매출 데이터 (`data/sales/`)

CSV 파일 형식:

**파일명**: `sales.csv` (또는 다른 이름)

**내용**:
```csv
month,sales
2024-01-01,1500000
2024-02-01,1650000
2024-03-01,1800000
2024-04-01,1750000
```

- **month**: 날짜 (YYYY-MM-DD 형식)
- **sales**: 매출액 (숫자)

## 💻 사용 방법

### 기본 실행 (자동 학습 및 예측)

```bash
# 가상환경 활성화 후
python main.py
```

또는 스크립트 사용:

```powershell
.\scripts\run.ps1
```

### 강제 재학습

```bash
python main.py --retrain
```

### 예측만 수행 (학습 생략)

```bash
python main.py --predict-only
```

### 설정 파일 지정

```bash
python main.py --config config/my_config.yaml
```

## 🧠 모델 구조

### 하이브리드 아키텍처

1. **시계열 모델 (LSTM)**
   - 입력: 과거 N개월의 시계열 데이터
   - 레이어: LSTM → Dropout → LSTM → Dropout
   - 출력: 시계열 특성 벡터

2. **텍스트 모델 (Dense)**
   - 입력: 단어-점수 특성 벡터
   - 레이어: Dense → Dropout → Dense → Dropout
   - 출력: 텍스트 특성 벡터

3. **결합 모델**
   - 입력: 시계열 특성 + 텍스트 특성
   - 레이어: Dense → Dropout → Dense → Dropout → Dense
   - 출력: 예측 매출 (1개 값)

### 모델 파라미터

- **시퀀스 길이**: 기본 12개월 (설정 가능)
- **LSTM 유닛**: [64, 32] (설정 가능)
- **Dropout 비율**: 0.2 (설정 가능)
- **학습률**: 0.001 (설정 가능)

## 🔄 자동화 프로세스

### 1. 데이터 확인
- 새 데이터가 있는지 확인
- 최소 데이터 개월 수 확인 (기본: 6개월)

### 2. 데이터 로딩 및 전처리
- 단어-점수 데이터 로딩
- 매출 데이터 로딩
- 데이터 결합 및 정렬

### 3. 특성 엔지니어링
- 시계열 시퀀스 생성
- 단어 특성 추출
- 데이터 정규화

### 4. 모델 학습
- 기존 모델 확인
- 새 데이터가 있으면 재학습
- 모델 저장

### 5. 예측 수행
- 다음 달 매출 예측
- 결과 저장 (JSON 형식)

### 6. 결과 저장
- 모델: `models/model_YYYYMMDD_HHMMSS.h5`
- 예측: `predictions/prediction_YYYYMMDD_HHMMSS.json`

## ⚙️ 설정 파일

`config/config.yaml` 파일에서 다음 항목을 설정할 수 있습니다:

### 모델 설정

```yaml
model:
  time_series:
    sequence_length: 12  # 시계열 시퀀스 길이
    lstm_units: [64, 32]  # LSTM 레이어 유닛 수
    dropout_rate: 0.2     # Dropout 비율
  
  text_features:
    dense_units: [32, 16]  # Dense 레이어 유닛 수
  
  training:
    epochs: 100            # 학습 에포크
    batch_size: 32         # 배치 크기
    learning_rate: 0.001   # 학습률
```

### 데이터 경로

```yaml
data:
  input_data_path: "data/input_data/"
  sales_data_path: "data/sales/"
  model_save_path: "models/"
  prediction_save_path: "predictions/"
```

### 자동화 설정

```yaml
automation:
  auto_retrain: true      # 자동 재학습 여부
  min_data_months: 6      # 최소 학습 데이터 개월 수
```

## 📁 프로젝트 구조

```
Demand_Analysis/
├── scripts/                  # 유틸리티 스크립트
│   ├── setup_venv.ps1       # 가상환경 설정
│   ├── activate.ps1          # 가상환경 활성화
│   ├── deactivate.ps1        # 가상환경 비활성화
│   ├── run.ps1               # 메인 스크립트 실행
│   └── README.md             # 스크립트 사용법
├── src/                      # 소스 코드
│   ├── data_loader.py        # 데이터 로딩 및 전처리
│   ├── feature_engineering.py # 특성 엔지니어링
│   ├── model.py              # 모델 정의
│   └── pipeline.py           # 자동화 파이프라인
├── config/                   # 설정 파일
│   └── config.yaml           # 모델 및 파이프라인 설정
├── data/                     # 데이터 저장
│   ├── input_data/           # 단어-점수 데이터 (JSON)
│   └── sales/                # 매출 데이터 (CSV)
├── models/                   # 학습된 모델 저장
├── predictions/              # 예측 결과 저장
├── logs/                     # 로그 파일
├── main.py                   # 메인 실행 스크립트
├── requirements.txt          # 패키지 목록
└── README.md                 # 이 파일
```

## 🔍 문제 해결

### 데이터 부족 오류

**문제**: "시퀀스 데이터가 부족합니다"

**해결**:
- 최소 `sequence_length + 1`개월의 데이터 필요
- `config/config.yaml`에서 `min_data_months` 확인
- 더 많은 데이터 수집

### 메모리 부족

**문제**: 학습 중 메모리 부족 오류

**해결**:
- `batch_size` 줄이기 (예: 32 → 16)
- `sequence_length` 줄이기 (예: 12 → 6)
- GPU 사용 (가능한 경우)

### 모델 로딩 실패

**문제**: "모델을 찾을 수 없습니다"

**해결**:
- 먼저 `python main.py` 실행하여 모델 학습
- `models/` 폴더에 모델 파일 확인

### 가상환경 활성화 실패

**문제**: PowerShell 실행 정책 오류

**해결**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 📈 성능 최적화

### 모델 성능 향상

1. **더 많은 데이터**: 학습 데이터가 많을수록 성능 향상
2. **하이퍼파라미터 튜닝**: `config.yaml`에서 조정
3. **특성 추가**: 더 많은 단어-점수 특성 사용

### 학습 속도 향상

1. **GPU 사용**: TensorFlow GPU 버전 설치
2. **배치 크기 조정**: 메모리에 맞게 조정
3. **조기 종료**: Early stopping 활용

## 📝 예시

### 데이터 준비 예시

1. **단어-점수 데이터 생성** (`data/input_data/2024-01.json`):
```json
{
  "프로모션": 0.9,
  "할인": 0.8,
  "신제품": 0.7
}
```

2. **매출 데이터 준비** (`data/sales/sales.csv`):
```csv
month,sales
2024-01-01,1500000
2024-02-01,1650000
```

3. **실행**:
```bash
python main.py
```

4. **결과 확인**:
- 모델: `models/model_20241216_143022.h5`
- 예측: `predictions/prediction_20241216_143022.json`

## 🔗 참고 자료

- [TensorFlow 공식 문서](https://www.tensorflow.org/)
- [Keras 가이드](https://keras.io/)
- [시계열 예측 기법](https://en.wikipedia.org/wiki/Time_series)

## 📄 라이선스

이 프로젝트는 내부 사용을 위한 것입니다.

---

**문의사항이나 문제가 있으면 이슈를 등록해주세요.**
