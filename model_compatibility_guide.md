# 모델별 호환성 가이드

## 🟢 **안전하게 실행 가능한 모델들**
1. **YoloOW**: 대부분의 버전과 호환 가능
2. **yolov8 (ultralytics)**: 현재 환경과 잘 호환됨
3. **YOLOH**: requirements가 유연하여 대부분 호환

## 🟡 **주의해서 실행해야 하는 모델들**
4. **FFCA-YOLO**: PyTorch 1.10 요구로 다운그레이드 필요 가능성
5. **MSNet**: 오래된 scipy 버전 요구

## 🔴 **별도 환경이 필요한 모델들**
6. **DNTR**: MMDetection 전체 설치 필요
7. **YOLC**: VisDrone 데이터셋 특수 처리 필요

## 실행 순서 권장사항:

### Phase 1: 호환성 높은 모델들 먼저 테스트
```bash
# 현재 환경에서 바로 실행 가능
python main.ipynb  # yolov8n, YoloOW_CLI만 활성화
```

### Phase 2: 중간 호환성 모델들
```bash
# requirements_unified.txt 설치 후
pip install -r requirements_unified.txt
python main.ipynb  # FFCA_YOLO_CLI, MSNet_CLI 추가
```

### Phase 3: 특수 환경 모델들
```bash
# DNTR용 별도 conda 환경
conda create -n dntr_env python=3.8
conda activate dntr_env
pip install mmcv-full mmdet
```

## 충돌 회피 전략:

1. **import 에러 발생 시**: 해당 모델만 registry에서 제외
2. **버전 충돌 발생 시**: 범위 지정으로 호환 버전 사용  
3. **런타임 에러 발생 시**: try-catch로 모델별 격리

## 모니터링 포인트:

- ⚠️ PyTorch 버전 호환성
- ⚠️ CUDA 버전 매칭
- ⚠️ protobuf 버전 (3.x vs 4.x 충돌)
- ⚠️ numpy dtype 변경사항
- ⚠️ OpenCV API 변경사항 