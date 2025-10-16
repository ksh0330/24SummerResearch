# 24SummerResearch

> **Note:** 이 저장소는 VITAS 프로젝트의 스포크 저장소입니다. 전체 프로젝트 개요와 문서는 **[Project VITAS](https://github.com/ksh0330/Project-VITAS)** 허브 저장소를 참조하세요.

## 📌 개요

이 저장소는 **2024년 연구참여A**에서 진행한 VITAS 프로젝트의 연구 과정과 실험 결과를 담고 있습니다.  
Vision AI 기반 교차로 충돌 방지 시스템의 핵심 알고리즘 개발과 성능 평가를 위한 다양한 실험들이 포함되어 있습니다.

---

## 🧠 주요 연구 영역

### 1. 객체 탐지 및 추적 (Object Detection & Tracking)
- YOLOv8 기반 실시간 객체 탐지
- 다중 객체 추적 알고리즘
- 교차로 영역별 객체 분류

### 2. 속도 추정 (Speed Estimation)
- 픽셀 기반 속도 계산
- 좌표계 변환을 통한 실제 속도 추정
- MLP 모델을 활용한 속도 예측

### 3. 위험도 분류 (Risk Classification)
- GRU + InceptionV3 기반 비디오 분류
- MLP를 활용한 위험도 예측
- 실시간 위험 상황 감지

---

## 📁 저장소 구조

```
24SummerResearch/
├── code/                    # 핵심 구현 코드
│   ├── mlp/                # MLP 모델 관련
│   │   ├── feature_extraction/  # 특징 추출
│   │   └── training/       # 훈련 및 시각화
│   ├── gru/                # GRU 모델 관련
│   │   ├── training/       # 훈련 코드
│   │   ├── realtime_classification.py
│   │   ├── realtime_classification_videotest.py
│   │   └── VideoClassification.py
│   └── yolo/               # YOLO 관련
│       └── lstm/          # LSTM 코드 
├── datasets/               # 모든 데이터
│   ├── mlp_data/          # MLP CSV 데이터
│   ├── gru_data/          # GRU NPY 데이터
│   └── test_videos/       # 테스트 비디오
├── results/               # 실험 결과
│   ├── mlp_results/       # MLP 결과 그래프
│   ├── gru_results/       # GRU 결과 그래프
│   └── yolo_results/      # YOLO 결과 그래프 
├── docs/                  # 연구참여일지 
│   └── research_participation_log(1-13).docx
└── README.md              # 프로젝트 문서
```

---

## 🚀 주요 파일 설명

### MLP 관련
- `mlp_main_feature_extraction.py`: A, B 영역 추적을 통한 속도 측정 및 CSV 변환
- `mlp_improved_feature_extraction.py`: 평균 속도 제거 버전 (개선된 특징 추출)
- `mlp_danger_feature_extraction.py`: 위험 상황 특징 추출
- `mlp_safe_feature_extraction.py`: 안전 상황 특징 추출
- `mlp_warn_feature_extraction.py`: 경고 상황 특징 추출
- `mlp_classifier_training.py`: MLP 분류기 훈련

### GRU 관련
- `GRU_Train.py`: GRU 모델 훈련
- `realtime_classification.py`: 실시간 분류
- `realtime_classification_videotest.py`: 비디오 테스트용 실시간 분류
- `VideoClassification.py`: 비디오 분류 메인 코드

### YOLO 관련
- `LSTM_train.py`: LSTM 모델 훈련
- `Realtime_YOLO_LSTM_Video_roof.py`: 실시간 YOLO + LSTM 비디오 처리
- `train.py`: YOLO 훈련
- `VITAS.py`: VITAS 시스템 메인 코드

---

## 🔗 관련 저장소

- **[Project VITAS](https://github.com/ksh0330/Project-VITAS)** - 메인 허브 저장소
- **[Speed-Estimation-YOLOv8](https://github.com/ksh0330/Speed-Estimation-YOLOv8)** - 핵심 속도 추정 알고리즘
- **[2024 Hanium ICT Contest](https://github.com/ksh0330/2024_Hanium_ICT_Contest)** - 공모전 참가작