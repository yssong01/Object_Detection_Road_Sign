# 🚦 Road Sign Detection Project (using YOLOv8s)

**딥러닝 기반 도로 표지판 4종 실시간 탐지 및 통계적 성능 분석**

본 연습 프로젝트에서는 Kaggle의 Road Sign 데이터셋을 활용하여 주요 표지판을 정밀하게 탐지하고 학습 결과를 테스트했습니다.

---

## 1. YOLOv8 채택 이유 (Why YOLOv8?)

본 프로젝트에서 **YOLOv8(You Only Look Once v8)**을 선정하여 학습을 진행한 이유는 다음과 같습니다:

- **실시간 처리 최적화**: 고해상도 이미지에서도 매우 높은 FPS(초당 프레임 수)를 보장하여 자율주행과 같은 실시간 응답이 필수적인 시스템에 가장 적합한 SOTA(State-of-the-Art) 모델입니다.

- **강력한 데이터 증강(Augmentation)**: Mosaic, Blur, Mixup 등 물리적으로 발생할 수 있는 다양한 외부 환경 변화(조도, 노이즈, 각도)를 학습 데이터에 자동으로 반영하여 모델의 강건성(Robustness)을 극대화합니다.

- **체계적인 사후 분석**: 학습 과정의 모든 수치를 CSV로 기록하고 시각화 도구를 제공하여, 단순 학습을 넘어 지표의 통계적 유의성을 검토하기 용이합니다.

---

## 2. 코드 흐름 및 시스템 구조 (Workflow)

전체 데이터 처리 파이프라인은 아래와 같이 정밀하게 설계되었습니다:

### 상세 프로세스

1. **환경 구성**
   - Google Colab (T4 GPU) 환경
   - Google Drive 연동으로 데이터셋 영속성 확보

2. **포맷 변환 (XML to YOLO)**
   - Pascal VOC 형식 XML → YOLO TXT 형식 변환
   - 좌표 정규화: `[x_c, y_c, w, h]` (0~1 상대값)

3. **데이터 무작위 분할**
   - Random Seed: `42` # <= 가장 일반적으로 사용되는 선택지
   - 분할 비율: Train:Val:Test = **7:1:2**

4. **모델 학습**
   - 베이스 모델: `yolov8s.pt` (Small)
   - 학습 Epoch: **30**
   - Batch Size: Auto

5. **성능 지표 가공**
   - 출력물: `results.png`, `test.png`
   - 커스터마이징: Epoch 간격 10, Y축 고정

---

## 3. 학습 결과 및 물리적 지표 분석

모델의 수렴 성능을 객관적으로 비교하기 위해 X축 10단위 Epoch 및 Y축 최대값 1.0 고정 설정을 적용하여 시각화하였습니다.

*Road Sign Detection: Customized Training Metrics (Epoch Interval: 10)*

<img width="7470" height="3686" alt="results" src="https://github.com/user-attachments/assets/76546646-e3d8-42e8-b2e4-aa5487179d36" />


### 손실 함수 (Loss Functions)

| Metric | 초기값 | 최종값 | 분석 |
|--------|--------|--------|------|
| **Box Loss** | 0.85 | 0.50 | 학습 초기(1~5 Epoch) 급격 하강, 10 Epoch 이후 안정 구간 진입 |
| **Classification Loss** | 3.5 | 0.4 | 클래스 분류 능력 지속적 향상 |
| **DFL Loss** | 1.2 | 0.85 | Bounding Box 위치 정확도 개선 |

### 정확도 지표 (Accuracy Metrics)

| Metric | 초기값 | 최종값 | 분석 |
|--------|--------|--------|------|
| **Precision(B)** | 0.65-0.85 | 0.95+ | 20 Epoch부터 고정밀도 유지, False Positive 최소화 |
| **Recall(B)** | 0.45 | 0.90+ | 표지판 누락 비율 크게 감소 |
| **mAP50(B)** | 0.55 | 0.95 | 선형적 상승, 성공적 학습 완료 |
| **mAP50-95(B)** | 0.35 | 0.80 | 다양한 IoU 임계값에서 안정적 성능 |

### 핵심 인사이트

- **과적합 방지**: Train/Val Loss가 함께 감소하며 과적합 없이 안정적으로 수렴
- **빠른 학습**: 초기 10 Epoch 내 급격한 성능 향상
- **높은 재현율**: Recall 0.90+ 달성으로 표지판 미탐지 최소화
- **정밀 탐지**: Precision 0.95+ 달성으로 오탐지 억제

---

## 4. 최종 탐지 성능 검증

학습에 전혀 사용되지 않은 테스트 데이터 16장을 샘플링하여 4행 4열 그리드로 추론 결과를 시각화하였습니다.

*YOLOv8 Road Sign Detection: 16 Test Samples*

<img width="7141" height="7359" alt="test" src="https://github.com/user-attachments/assets/74afba9a-29e7-4779-9e64-482be2501914" />


### 🎯 주요 성능 지표

#### 완벽한 탐지 사례
- **road103.jpg**: `speedlimit` 신뢰도 **1.00** (100%)
- **road120.jpg**: `speedlimit` 신뢰도 **0.99**
- **road125.jpg**: `crosswalk` 신뢰도 **0.94**
- **road138.jpg**: `crosswalk` 신뢰도 **0.95**

#### 복합 환경 대응
- **어두운 조명** (road15.jpg): `trafficlight` **0.81** 탐지
- **원거리 표지판** (road108.jpg): `speedlimit` **0.81** 탐지
- **비스듬한 각도** (road100.jpg): `speedlimit` **0.82** 탐지

#### 다중 객체 탐지
- **road127.jpg**: `trafficlight` (0.82) + `crosswalk` (0.83) 동시 탐지
- **road170.jpg**: `trafficlight` (0.88) + `crosswalk` (0.96) + `trafficlight` (0.44) 3개 객체 탐지

### 📊 클래스별 성능 요약

| 클래스 | 평균 Confidence | 탐지 성공률 | 특이사항 |
|--------|----------------|-------------|----------|
| Speed Limit | 0.90+ | 100% | 원형 형태 완벽 인식 |
| Crosswalk | 0.85+ | 100% | 삼각형 형태 정확 탐지 |
| Traffic Light | 0.80+ | 95% | 복잡한 배경에서도 안정적 |
| Stop | 0.90+ | 100% | 팔각형 특징 명확히 학습 |

---

## Quick Start

### 1. 환경 설정

```bash
# YOLOv8 설치
pip install ultralytics

# 필요한 라이브러리 설치
pip install opencv-python matplotlib pandas
```

### 2. 데이터셋 준비

```python
# Google Drive 마운트 (Colab)
from google.colab import drive
drive.mount('/content/drive')

# 데이터셋 경로 설정
dataset_path = '/content/drive/MyDrive/Data'
```

### 3. 학습 실행

```bash
yolo detect train \
  data=data.yaml \
  model=yolov8s.pt \
  epochs=30 \
  imgsz=640 \
  batch=-1
```

### 4. 추론 실행

```bash
# 단일 이미지 추론
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=test/images/road100.jpg

# 전체 테스트 세트 추론
yolo detect predict \
  model=runs/detect/train/weights/best.pt \
  source=test/images/ \
  save=True
```

---

## 📊 성능 요약표

| 항목 | 지표 | 값 |
|------|------|-----|
| **전체 성능** | mAP50 | **0.95** |
| | mAP50-95 | **0.80** |
| **정밀도** | Precision | **0.97** |
| **재현율** | Recall | **0.91** |
| **속도** | FPS (T4 GPU) | **~60** |
| **클래스별** | Speed Limit | **1.00** |
| | Crosswalk | **0.94** |
| | Traffic Light | **0.82+** |
| | Stop | **0.90+** |

---

## 주요 특징

### 강점
- **높은 정확도**: mAP50 0.95 달성
- **실시간 처리**: 60 FPS 이상 (T4 GPU)
- **모델의 강건성**: 다양한 환경 조건(야간, 원거리, 각도)에서 안정적 탐지
- **다중 객체**: 한 이미지 내 여러 표지판 동시 탐지 가능

### 개선 가능 영역
- 극심한 날씨 조건(폭우, 안개)에서의 데이터 추가 학습
- 희귀 표지판 및 국내 표지판으로 클래스 확장
- 모바일 엣지 디바이스 최적화 (YOLOv8n)

---

## Acknowledgments

- **Dataset**: [Kaggle Road Sign Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/road-sign-detection)
- **Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Platform**: Google Colab & Google Drive

---

