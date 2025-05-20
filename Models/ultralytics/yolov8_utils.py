from ultralytics import YOLO

def build_yolov8_model(cfg=None, ex_dict=None):
    # cfg: yaml 경로 or 모델명(str), ex_dict: 실험 정보 dict
    device = ex_dict['Device']
    model = YOLO(cfg)  # cfg가 'yolov8n.pt' 또는 yaml 경로
    model.to(device)
    return model

def train_yolov8_model(ex_dict):
    model = ex_dict['Model']
    # ultralytics는 내부적으로 train 메서드 제공
    results = model.train(
        data=ex_dict['Data YAML Path'],
        epochs=ex_dict['Epochs'],
        batch=ex_dict['Batch Size'],
        imgsz=ex_dict['Image Size'],
        device=ex_dict['Device Index'],  # 예: 0
        project=ex_dict['Output Dir'],
        name=ex_dict['Experiment Name'],
        # 기타 하이퍼파라미터
    )
    # 결과 저장 등 추가
    return results

def eval_yolov8_model(ex_dict):
    model = ex_dict['Model']
    # 검증 데이터셋에 대해 평가
    results = model.val(
        data=ex_dict['Data YAML Path'],
        batch=ex_dict['Batch Size'],
        imgsz=ex_dict['Image Size'],
        device=ex_dict['Device Index'],
    )
    return results

def test_yolov8_model(ex_dict):
    model = ex_dict['Model']
    # 테스트 데이터셋에 대해 평가
    results = model.val(
        data=ex_dict['Data YAML Path'],
        split='test',
        batch=ex_dict['Batch Size'],
        imgsz=ex_dict['Image Size'],
        device=ex_dict['Device Index'],
    )
    return results
