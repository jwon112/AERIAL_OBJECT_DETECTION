from ultralytics import YOLO
import os
from datetime import datetime

def build_yolov8_model(cfg=None, ex_dict=None):
    # cfg: yaml 경로 or 모델명(str), ex_dict: 실험 정보 dict
    device = ex_dict['Device']
    model = YOLO(cfg)  # cfg가 'yolov8n.pt' 또는 yaml 경로
    model.to(device)
    return model

def extract_metrics(results):
    """YOLOv8의 결과에서 필요한 메트릭을 추출"""
    metrics = {}
    
    # 전체 성능 지표
    metrics.update({
        'mAP@0.5': results.box.map50,
        'mAP@0.5:0.95': results.box.map,
        'Mean Precision': results.box.mp,
        'Mean Recall': results.box.mr,
        'mAP@0.75': results.box.map75
    })
    
    # 클래스별 성능 지표
    for i, cls_name in enumerate(results.names.values()):
        metrics.update({
            f'{cls_name}/Precision': results.box.p[i],
            f'{cls_name}/Recall': results.box.r[i],
            f'{cls_name}/mAP@0.5': results.box.ap50[i],
            f'{cls_name}/mAP@0.5:0.95': results.box.ap[i]
        })
    
    # 속도 지표
    if hasattr(results, 'speed'):
        # YOLOv8의 speed 딕셔너리 구조 확인
        speed_dict = results.speed
        metrics['Speed/inference (ms/img) (ms)'] = speed_dict.get('inference', 0)
        metrics['Speed/nms (ms/img) (ms)'] = speed_dict.get('postprocess', 0)  # nms는 postprocess에 포함
    
    return metrics

def train_yolov8_model(ex_dict):
    model = ex_dict['Model']
    
    # Train Time 설정 (다른 모델과 동일한 형식)
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    
    # 실험 이름 생성 - iteration 값만 사용하도록 수정
    # 원래 형식: {Train Time}_{Model Name}_{Dataset Name}_Iter_{Iteration}
    # 수정: 폴더명 중복이 발생하지 않도록 Iter 대신 iter-only 사용
    experiment_name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_iteration{ex_dict['Iteration']}"
    
    # 학습 실행
    results = model.train(
        data=ex_dict['Data Config'],
        epochs=ex_dict['Epochs'],
        batch=ex_dict['Batch Size'],
        imgsz=ex_dict['Image Size'],
        device=ex_dict['Device'],
        project=ex_dict['Output Dir'],
        name=experiment_name,
    )
    
    # PT 파일 경로 저장 (다른 모델과 동일한 형식)
    pt_path = os.path.join(ex_dict['Output Dir'], experiment_name, 'weights', 'best.pt')
    ex_dict['PT path'] = pt_path
    
    # 학습 결과 저장
    ex_dict['Train Results'] = {
        'box_loss': float(results.box_loss) if hasattr(results, 'box_loss') else 0.0,
        'cls_loss': float(results.cls_loss) if hasattr(results, 'cls_loss') else 0.0,
        'dfl_loss': float(results.dfl_loss) if hasattr(results, 'dfl_loss') else 0.0
    }
    return ex_dict

def eval_yolov8_model(ex_dict):
    model = ex_dict['Model']
    # 검증 데이터셋에 대해 평가
    results = model.val(
        data=ex_dict['Data Config'],
        batch=ex_dict['Batch Size'],
        imgsz=ex_dict['Image Size'],
        device=ex_dict['Device'],
    )
    # 평가 결과 저장 - 다른 모델들과 동일한 형식으로
    ex_dict["Val Results"] = {
        "box_loss": float(results.box.loss) if hasattr(results.box, 'loss') else 0.0,
        "obj_loss": float(results.box.conf_loss) if hasattr(results.box, 'conf_loss') else 0.0,
        "cls_loss": float(results.box.cls_loss) if hasattr(results.box, 'cls_loss') else 0.0,
        "total_loss": sum([
            float(results.box.loss) if hasattr(results.box, 'loss') else 0.0,
            float(results.box.conf_loss) if hasattr(results.box, 'conf_loss') else 0.0,
            float(results.box.cls_loss) if hasattr(results.box, 'cls_loss') else 0.0
        ])
    }
    
    # 메트릭 업데이트 (CSV 파일용)
    ex_dict.update(extract_metrics(results))
    return ex_dict

def test_yolov8_model(ex_dict):
    model = ex_dict['Model']
    # 테스트 데이터셋에 대해 평가
    results = model.val(
        data=ex_dict['Data Config'],
        split='test',  # test 데이터셋 사용
        batch=ex_dict['Batch Size'],
        imgsz=ex_dict['Image Size'],
        device=ex_dict['Device'],
    )
    # 테스트 결과 저장 - 원본 results 객체를 Test Results로 저장
    ex_dict['Test Results'] = results
    
    # CSV 파일용 메트릭 업데이트
    metrics = extract_metrics(results)
    ex_dict.update(metrics)
    
    return ex_dict
