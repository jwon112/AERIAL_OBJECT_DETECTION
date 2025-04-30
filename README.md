## 📁 Directory Structure

AERIAL_OBJECT_DETECTION/ \
├── Datasets/ # 각 데이터셋 (YOLO 포맷: /images, /labels) \
├── Models/ # 모델 소스코드 (YOLOH, YoloOW 등) \
├── output/ # 실험 결과 저장\
├── utility/ # 공통 함수 및 모델 래퍼\
| ├── autoanchor.py \
│ ├── dataloader_utils.py \    
│ ├── metrics.py # 성능 계산 관련 함수\
│ ├── optimizer.py \
│ ├── path_manager.py # import 경로 설정\
│ ├── trainer.py # 훈련 루프 및 출력 설정\
│ ├── yoloh_utils.py\
│ ├── yoloow_utils.py\
│ └── utils.py \
├── main.ipynb # 실험 실행 (학습/검증/테스트)\
├── Data_Split.ipynb # 데이터셋 분할 및 yaml 생성\
├── registry.py # 모델 등록 및 파이프라인 호출\
└── requirements.txt # 의존 패키지 목록\

## 🧪 주요 기능
✅ YOLO 포맷 데이터 자동 로딩

✅ 멀티 모델 통합 실험 파이프라인

✅ 객체 탐지 성능 평가 (mAP, Confusion Matrix 등)

✅ AutoAnchor 적합도 평가

✅ 학습 결과 저장 및 반복 실험 로그 기록




## ⚙️ Model Integration

새 모델을 추가하려면:

1. `/Models/`에 전체 소스코드 추가
2. `/utility/`에 `yourmodel_utils.py` 작성
3. `registry.py`에 `'yourmodel': {...}` 형태로 등록
4. `main.ipynb`에서 `model_name='yourmodel'` 지정 후 실행



