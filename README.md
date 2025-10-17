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
├── code/                    
│   ├── mlp/                
│   │   ├── feature_extraction/  
│   │   └── training/       
│   ├── gru/                
│   │   ├── training/       
│   │   ├── realtime_classification.py
│   │   ├── realtime_classification_videotest.py
│   │   └── VideoClassification.py
│   ├── yolo/               
│   └── lstm/               
├── datasets/               
│   ├── mlp_data/          
│   ├── gru_data/          
│   └── test_videos/       
├── results/               
│   ├── mlp_results/       
│   ├── gru_results/       
│   └── yolo_results/      
├── docs/                  
│   └── research_participation_log(1-13).docx
└── README.md              
```

---

## 📋 주요 구성

- **MLP**: 속도 추정 및 위험도 분류 모델
- **GRU**: 비디오 기반 실시간 위험 상황 분류
- **YOLO**: 객체 탐지 모델
- **LSTM**: 시계열 데이터 기반 위험도 예측
- **Results**: 각 모델별 성능 분석 그래프

---