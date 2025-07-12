import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import yaml
import tempfile
import argparse

# MSNet 폴더를 시스템 경로에 추가
MSNET_DIR = os.path.dirname(os.path.abspath(__file__))
MSNET_SOURCE_DIR = os.path.join(MSNET_DIR, "SourceFile")

def parse_msnet_params(save_dir):
    """MSNet 모델 파라미터 정보 파싱"""
    param_info = {}
    param_file_path = os.path.join(save_dir, 'model_params.txt')
    
    if not os.path.exists(param_file_path):
        print(f"파라미터 정보 파일이 존재하지 않습니다: {param_file_path}")
        return param_info
    
    try:
        with open(param_file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if key in ['Total Parameters', 'Trainable Parameters', 'Non-trainable Parameters']:
                    param_info[key] = int(value)
                else:
                    param_info[key] = value
                    
        print(f"📊 모델 파라미터 정보 로드: {param_info}")
                    
    except Exception as e:
        print(f"MSNet 파라미터 파일 파싱 에러: {e}")
    
    return param_info

def parse_msnet_results(results_path):
    """MSNet 결과 파일 파싱"""
    metrics = {}
    if not os.path.exists(results_path):
        print(f"결과 파일이 존재하지 않습니다: {results_path}")
        return metrics
    
    try:
        with open(results_path, 'r') as f:
            lines = f.readlines()
            
        print(f"결과 파일 읽기: {results_path}")
        print(f"파일 내용: {lines}")
        
        # get_map_coco.py에서 생성하는 results.txt 형식 파싱
        for line in lines:
            line = line.strip()
            if ':' in line:
                try:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # MSNet 결과 매핑
                    if key == 'mAP':
                        metrics['mAP'] = float(value)
                    elif key == 'AP50':
                        metrics['AP50'] = float(value)
                    elif key == 'AP75':
                        metrics['AP75'] = float(value)
                    elif key == 'APs':
                        metrics['APs'] = float(value)
                    elif key == 'APm':
                        metrics['APm'] = float(value)
                    elif key == 'APl':
                        metrics['APl'] = float(value)
                    else:
                        # 기타 메트릭도 저장
                        metrics[key] = float(value)
                        
                except ValueError as e:
                    print(f"값 파싱 오류 (라인: {line}): {e}")
                    continue
        
        # 기존 MSNet 출력 형식 파싱 (백업)
        if not metrics:
            for line in reversed(lines):
                if 'Epoch' in line and 'mAP' in line:
                    # MSNet 출력 형식 파싱
                    import re
                    # Epoch xxx/xxx: loss=x.xxx, mAP=x.xxx 형식 파싱
                    loss_match = re.search(r'loss=([\d.]+)', line)
                    map_match = re.search(r'mAP=([\d.]+)', line)
                    
                    if loss_match:
                        metrics['total_loss'] = float(loss_match.group(1))
                    if map_match:
                        metrics['mAP'] = float(map_match.group(1))
                    break
            
            # key:value 형식도 시도
            for line in lines:
                if ':' in line and ('loss' in line.lower() or 'map' in line.lower()):
                    try:
                        key, value = line.strip().split(':', 1)
                        metrics[key.strip()] = float(value.strip())
                    except ValueError:
                        pass
        
        print(f"파싱된 메트릭: {metrics}")
                    
    except Exception as e:
        print(f"MSNet 결과 파일 파싱 에러: {e}")
    
    return metrics

def build_msnet_model_cli(cfg=None, ex_dict=None):
    """MSNet 모델 CLI 빌드 함수"""
    device = ex_dict.get('Device', 'cpu')
    
    if cfg is None:
        cfg = "yolov8_l.yaml"  # MSNet은 YOLOv8-L 기본
    
    # 🔥 CRITICAL: Build 시점에서 Model Config를 ex_dict에 먼저 설정
    # registry.py에서 전달받은 cfg 파라미터를 ex_dict에 반영
    ex_dict['Model Config'] = cfg
    
    return {
        'cfg': cfg,
        'device': device,
        'ex_dict': ex_dict
    }

def create_msnet_data_config(ex_dict):
    """
    MSNet 데이터 설정 파일 생성
    
    Args:
        ex_dict (dict): 데이터 설정을 담은 딕셔너리
    
    Returns:
        str: 생성된 임시 데이터 설정 파일 경로
    """
    # 프로젝트 루트 경로 계산 (MSNet/SourceFile의 상위 3단계)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(MSNET_SOURCE_DIR)))
    
    # 데이터셋 디렉토리 설정
    ex_dict['Dataset Dir'] = os.path.join(project_root, 'Datasets', ex_dict['Dataset Name'])
    
    # 데이터 설정 파일 읽기
    data_config_path = os.path.abspath(ex_dict['Data Config'])
    print(f"Reading data config from: {data_config_path}")
    
    with open(data_config_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    # 데이터셋 디렉토리 경로 설정
    dataset_dir = ex_dict['Dataset Dir']
    
    # train.txt와 val.txt 파일 경로 설정
    train_file = os.path.join(dataset_dir, data_config['train'])
    val_file = os.path.join(dataset_dir, data_config['val'])
    
    print(f"Reading train file from: {train_file}")
    print(f"Reading val file from: {val_file}")
    
    # train.txt와 val.txt 파일 읽기
    with open(train_file, 'r') as f:
        train_lines = f.readlines()
    with open(val_file, 'r') as f:
        val_lines = f.readlines()
    
    # 임시 파일 생성
    temp_dir = tempfile.gettempdir()
    temp_train_path = os.path.join(temp_dir, 'train.txt')
    temp_val_path = os.path.join(temp_dir, 'val.txt')
    
    # 절대 경로로 변환하여 임시 파일에 저장 (짧은 경로 사용으로 공백 문제 해결)
    try:
        import win32api
        # 짧은 경로 사용
        short_dataset_dir = win32api.GetShortPathName(dataset_dir)
    except ImportError:
        # win32api가 없으면 절대 경로 그대로 사용
        short_dataset_dir = dataset_dir
    
    with open(temp_train_path, 'w') as f:
        for line in train_lines:
            # 상대 경로를 절대 경로로 변환
            parts = line.strip().split()
            if len(parts) > 0:
                rel_path = parts[0]
                # 상대 경로에서 데이터셋 경로 부분 제거 (중복 방지)
                if rel_path.startswith(f"Datasets{os.sep}{ex_dict['Dataset Name']}{os.sep}"):
                    rel_path = rel_path[len(f"Datasets{os.sep}{ex_dict['Dataset Name']}{os.sep}"):]
                abs_path = os.path.join(short_dataset_dir, rel_path)
                # 공백이 있으면 따옴표로 감싸기
                if ' ' in abs_path:
                    new_line = f'"{abs_path}"'
                else:
                    new_line = abs_path
                # YOLO 형식에서는 이미지 경로만 포함 (라벨은 별도 labels 폴더에서 읽음)
                f.write(f"{new_line}\n")
    
    with open(temp_val_path, 'w') as f:
        for line in val_lines:
            # 상대 경로를 절대 경로로 변환
            parts = line.strip().split()
            if len(parts) > 0:
                rel_path = parts[0]
                # 상대 경로에서 데이터셋 경로 부분 제거 (중복 방지)
                if rel_path.startswith(f"Datasets{os.sep}{ex_dict['Dataset Name']}{os.sep}"):
                    rel_path = rel_path[len(f"Datasets{os.sep}{ex_dict['Dataset Name']}{os.sep}"):]
                abs_path = os.path.join(short_dataset_dir, rel_path)
                # 공백이 있으면 따옴표로 감싸기
                if ' ' in abs_path:
                    new_line = f'"{abs_path}"'
                else:
                    new_line = abs_path
                # YOLO 형식에서는 이미지 경로만 포함 (라벨은 별도 labels 폴더에서 읽음)
                f.write(f"{new_line}\n")
    
    # 데이터 설정 업데이트
    data_config['nc'] = len(data_config['names'])
    data_config['path'] = dataset_dir
    data_config['train'] = temp_train_path
    data_config['val'] = temp_val_path
    
    # 임시 YAML 파일 생성
    temp_data_path = os.path.join(temp_dir, f"tmp{next(tempfile._get_candidate_names())}.yaml")
    with open(temp_data_path, 'w') as f:
        yaml.dump(data_config, f)
    
    return temp_data_path

def validate_ex_dict(ex_dict, required_keys):
    """
    ex_dict에 필수 키들이 있는지 검증하는 함수
    
    Args:
        ex_dict (dict): 검증할 딕셔너리
        required_keys (list): 필수 키 리스트
    
    Raises:
        ValueError: 필수 키가 없을 경우
    """
    missing_keys = [key for key in required_keys if key not in ex_dict]
    if missing_keys:
        raise ValueError(f"Missing required keys in ex_dict: {', '.join(missing_keys)}")

def initialize_ex_dict(ex_dict=None):
    """
    ex_dict를 초기화하고 기본값을 설정하는 공통 함수
    
    Args:
        ex_dict (dict, optional): 초기화할 딕셔너리. None이면 새로 생성
        
    Returns:
        dict: 초기화된 ex_dict
    """
    if ex_dict is None:
        ex_dict = {}
    
    # 기본값 설정
    ex_dict.setdefault('Num Workers', 4)
    ex_dict.setdefault('Early Stop', 50)
    ex_dict.setdefault('AutoAnchor', True)
    ex_dict.setdefault('Experiment Time', datetime.now().strftime("%y%m%d_%H%M%S"))
    ex_dict.setdefault('Train Time', datetime.now().strftime("%y%m%d_%H%M%S"))
    ex_dict.setdefault('Model Name', 'MSNet')
    ex_dict.setdefault('Dataset Name', 'Unknown')
    ex_dict.setdefault('Iteration', '1')
    ex_dict.setdefault('Output Dir', 'output')
    ex_dict.setdefault('Image Size', 640)
    ex_dict.setdefault('Device', 'cpu')
    ex_dict.setdefault('Batch Size', 16)
    ex_dict.setdefault('Epochs', 100)
    ex_dict.setdefault('LR', 0.001)
    ex_dict.setdefault('Optimizer', 'Adam')
    ex_dict.setdefault('Momentum', 0.937)
    ex_dict.setdefault('Weight Decay', 0.0005)
    
    # 데이터 설정 파일이 없으면 기본값 설정
    if 'Data Config' not in ex_dict:
        iteration = ex_dict.get('Iteration', '1')
        data_config = {
            'path': os.path.join('Datasets', 'NWPU_VHR10_YOLO'),
            'train': f'train_iter_{iteration}.txt',
            'val': f'val_iter_{iteration}.txt',
            'test': f'test_iter_{iteration}.txt',
            'nc': 10,
            'names': [
                'airplane', 'ship', 'storage tank', 'baseball diamond',
                'tennis court', 'basketball court', 'ground track field',
                'harbor', 'bridge', 'vehicle'
            ]
        }
        temp_config_path = os.path.join('Datasets', 'NWPU_VHR10_YOLO', f'data_iter_{iteration}.yaml')
        os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, allow_unicode=True)
        ex_dict['Data Config'] = temp_config_path
        ex_dict['Number of Classes'] = data_config['nc']
        ex_dict['Class Names'] = data_config['names']
    
    # 모델이 없으면 기본 모델 생성
    if 'Model' not in ex_dict:
        model_info = {
            'cfg': os.path.join(MSNET_SOURCE_DIR, 'model_data', 'msnet_l.yaml'),
            'weights': None
        }
        ex_dict['Model'] = model_info
    
    return ex_dict

def train_fn(ex_dict):
    """
    MSNet 모델 학습 함수
    
    Args:
        ex_dict (dict): 학습에 필요한 설정을 담은 딕셔너리
    """
    ex_dict = initialize_ex_dict(ex_dict)
    return train_msnet_model_cli(ex_dict)

def eval_fn(ex_dict):
    """
    MSNet 모델 평가 함수
    
    Args:
        ex_dict (dict): 평가에 필요한 설정을 담은 딕셔너리
    """
    ex_dict = initialize_ex_dict(ex_dict)
    return eval_msnet_model_cli(ex_dict)

def test_fn(ex_dict):
    """
    MSNet 모델 테스트 함수
    
    Args:
        ex_dict (dict): 테스트에 필요한 설정을 담은 딕셔너리
    """
    ex_dict = initialize_ex_dict(ex_dict)
    return test_msnet_model_cli(ex_dict)

def train_msnet_model_cli(ex_dict):
    """
    MSNet 모델을 CLI로 학습
    
    Args:
        ex_dict (dict): 학습에 필요한 설정을 담은 딕셔너리
    """
    # 학습 시작 시간 설정
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    
    # Model Config는 build 함수에서 이미 설정됨 (registry.py에서 전달받은 값)
    # 만약 설정되지 않았다면 기본값 설정 (백업용)
    if 'Model Config' not in ex_dict:
        ex_dict['Model Config'] = 'yolov8_m'  # 기본값 설정
        print(f"⚠️  [MSNet] Model Config가 설정되지 않아 기본값 사용: {ex_dict['Model Config']}")
    else:
        print(f"✅ [MSNet] Model Config 설정됨: {ex_dict['Model Config']}")
    
    # 데이터 설정 파일 생성
    temp_data_path = create_msnet_data_config(ex_dict)
    temp_dir = os.path.dirname(temp_data_path)
    
    # 출력 디렉토리 설정
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    output_path = os.path.join(ex_dict['Output Dir'], name)
    os.makedirs(output_path, exist_ok=True)
    
    # 학습 스크립트
    train_script = os.path.join(MSNET_SOURCE_DIR, 'train.py')
    
    # 경로 설정
    train_annotation_path = os.path.join(temp_dir, 'train.txt')
    val_annotation_path = os.path.join(temp_dir, 'val.txt')
    classes_path = os.path.join(MSNET_SOURCE_DIR, 'model_data', 'classes.txt')
    
    # input_shape는 문자열 리스트 형태로 전달 (train.py에서 eval() 사용)
    input_shape_str = f"[{ex_dict['Image Size']}, {ex_dict['Image Size']}]"
    
    # 절대 경로로 save_dir 설정 (수정된 부분)
    save_dir_abs = os.path.abspath(output_path)
    
    # MSNet 모델 크기 설정 (phi 파라미터)
    # ex_dict에서 Model Config를 확인하여 phi 값 결정
    model_config = ex_dict.get('Model Config', 'yolov8_s')  # 기본값: s
    if 'yolov8_n' in model_config:
        phi = 'n'
    elif 'yolov8_s' in model_config:
        phi = 's'
    elif 'yolov8_m' in model_config:
        phi = 'm'
    elif 'yolov8_l' in model_config:
        phi = 'l'
    elif 'yolov8_x' in model_config:
        phi = 'x'
    else:
        phi = 's'  # 기본값
    
    print(f"[MSNet] Model Config: {model_config} → phi: {phi}")

    # ex_dict에서 seed 값 읽기
    seed = ex_dict.get('Seed', 42)  # MSNet 기본값: 3407

    cmd = [
        sys.executable,
        train_script,
        f"--train_annotation_path={train_annotation_path}",
        f"--val_annotation_path={val_annotation_path}",
        f"--classes_path={classes_path}",
        f"--input_shape={input_shape_str}",
        f"--phi={phi}",  # phi 파라미터 추가
        f"--save_dir={save_dir_abs}",
        f"--cuda={'True' if ex_dict['Device'] != 'cpu' else 'False'}",
        f"--UnFreeze_Epoch={ex_dict.get('Epochs', 1)}",
        f"--Unfreeze_batch_size={ex_dict.get('Batch Size', 16)}",
        f"--Init_lr={ex_dict.get('LR', 0.001)}",
        f"--Min_lr={ex_dict.get('LR', 0.001) * 0.01}",  # LR의 0.01배로 수정 (더 안정적)
        f"--momentum={ex_dict.get('Momentum', 0.937)}",  # Momentum 추가
        f"--weight_decay={ex_dict.get('Weight Decay', 0)}",  # Weight Decay 추가
        f"--optimizer_type={ex_dict.get('Optimizer', 'adam').lower()}",  # Optimizer 추가
        f"--Freeze_Train=False",  # 동결 훈련 비활성화
        f"--num_workers={ex_dict.get('Num Workers', 0)}",
        f"--seed={seed}"  # 🎯 seed 파라미터 추가
    ]
    
    print(f"MSNet 학습 명령어: {' '.join(cmd)}")
    
    # 환경 변수 설정
    env = os.environ.copy()
    env['PYTHONPATH'] = MSNET_SOURCE_DIR
    env['PYTHONUNBUFFERED'] = '1'
    
    try:
        # MSNet SourceFile 디렉토리에서 실행 (수정된 부분)
        
        process = subprocess.Popen(cmd, cwd=MSNET_SOURCE_DIR, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True, 
                                 bufsize=0, universal_newlines=True, env=env)
        
        # 실시간 출력
        stdout_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[MSNet] {output.strip()}")
                stdout_lines.append(output.strip())
        
        return_code = process.poll()
        print(f"MSNet 학습 완료. 반환 코드: {return_code}")
        
    except Exception as e:
        print(f"MSNet 학습 중 오류: {e}")
        return_code = 1
    
    # 가중치 파일 경로 설정
    ex_dict['PT path'] = os.path.join(output_path, 'best_epoch_weights.pth')
    
    # 모델 파라미터 정보를 ex_dict에 추가
    param_info = parse_msnet_params(output_path)
    if param_info:
        for key, value in param_info.items():
            ex_dict[key] = value
        print(f"📊 모델 파라미터 정보가 ex_dict에 추가되었습니다.")
    
    # 임시 데이터 파일 삭제
    if os.path.exists(temp_data_path):
        os.unlink(temp_data_path)
    if os.path.exists(train_annotation_path):
        os.unlink(train_annotation_path)
    if os.path.exists(val_annotation_path):
        os.unlink(val_annotation_path)
    
    return ex_dict

def eval_msnet_model_cli(ex_dict):
    """
    MSNet 모델을 CLI로 평가
    
    Args:
        ex_dict (dict): 평가에 필요한 설정을 담은 딕셔너리
    """
    # 데이터 설정 파일 생성
    temp_data_path = create_msnet_data_config(ex_dict)
    
    # 출력 디렉토리 설정
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    output_path = os.path.join(ex_dict['Output Dir'], name)
    os.makedirs(output_path, exist_ok=True)
    
    # 평가 스크립트
    eval_script = os.path.join(MSNET_SOURCE_DIR, 'utils_coco', 'get_map_coco.py')
    
    # 경로 설정 (따옴표 제거)
    model_path = ex_dict["PT path"]
    
    # 체크포인트 파일 존재 확인 및 대체 파일 찾기 (수정된 부분)
    if not os.path.exists(model_path):
        print(f"체크포인트 파일을 찾을 수 없습니다: {model_path}")
        # 대체 파일들 확인
        output_path = os.path.join(ex_dict['Output Dir'], f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}")
        alternative_files = [
            os.path.join(output_path, 'last_epoch_weights.pth'),
            os.path.join(output_path, 'ep100-loss*.pth'),  # 마지막 epoch 파일
        ]
        
        found_model = False
        for alt_file in alternative_files:
            if '*' in alt_file:
                # 와일드카드 패턴 매칭
                import glob
                matching_files = glob.glob(alt_file)
                if matching_files:
                    # 가장 최근 파일 선택
                    model_path = max(matching_files, key=os.path.getctime)
                    found_model = True
                    print(f"대체 체크포인트 파일을 찾았습니다: {model_path}")
                    break
            elif os.path.exists(alt_file):
                model_path = alt_file
                found_model = True
                print(f"대체 체크포인트 파일을 찾았습니다: {model_path}")
                break
        
        if not found_model:
            print(f"사용 가능한 체크포인트 파일이 없습니다. 평가를 건너뜁니다.")
            ex_dict['Eval Results'] = {'mAP': 0.0}
            return ex_dict
    
    # 모델 경로를 절대 경로로 변환 (수정된 부분)
    model_path_abs = os.path.abspath(model_path)
    data_yaml = temp_data_path
    # 절대 경로로 변환 (수정된 부분)
    map_out_path = os.path.abspath(output_path)
    
    # MSNet 모델 크기 설정 (phi 파라미터)
    model_config = ex_dict.get('Model Config', 'yolov8_s')  # 기본값: s
    if 'yolov8_n' in model_config:
        phi = 'n'
    elif 'yolov8_s' in model_config:
        phi = 's'
    elif 'yolov8_m' in model_config:
        phi = 'm'
    elif 'yolov8_l' in model_config:
        phi = 'l'
    elif 'yolov8_x' in model_config:
        phi = 'x'
    else:
        phi = 's'  # 기본값
    
    print(f"[MSNet Eval] Model Config: {model_config} → phi: {phi}")

    # input_shape는 개별 정수로 전달 (get_map_coco.py에서 nargs='+' 사용)
    cmd = [
        sys.executable,
        eval_script,
        f"--model_path={model_path_abs}",
        f"--data_yaml={data_yaml}",
        f"--map_out_path={map_out_path}",
        "--input_shape", str(ex_dict['Image Size']), str(ex_dict['Image Size']),
        f"--phi={phi}",  # phi 파라미터 추가
        f"--confidence=0.3",  # 신뢰도 임계값을 0.3으로 설정
        f"--cuda={'True' if ex_dict['Device'] != 'cpu' else 'False'}"
    ]
    
    print(f"MSNet 평가 명령어: {' '.join(cmd)}")
    
    # 환경 변수 설정
    env = os.environ.copy()
    env['PYTHONPATH'] = MSNET_SOURCE_DIR
    env['PYTHONUNBUFFERED'] = '1'
    
    try:
        # MSNet SourceFile 디렉토리에서 실행 (수정된 부분)
        
        process = subprocess.Popen(cmd, cwd=MSNET_SOURCE_DIR, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True, 
                                 bufsize=0, universal_newlines=True, env=env)
        
        # 실시간 출력
        stdout_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[MSNet] {output.strip()}")
                stdout_lines.append(output.strip())
        
        return_code = process.poll()
        print(f"MSNet 평가 완료. 반환 코드: {return_code}")
        
    except Exception as e:
        print(f"MSNet 평가 중 오류: {e}")
        return_code = 1
    
    # 결과 파싱
    # get_map_coco.py는 map_out_path 디렉토리에 results.txt를 생성
    results_path = os.path.join(output_path, 'results.txt')
    
    if os.path.exists(results_path):
        metrics = parse_msnet_results(results_path)
    else:
        print(f"결과 파일을 찾을 수 없습니다: {results_path}")
        # 디버깅을 위해 디렉토리 내용 확인
        if os.path.exists(output_path):
            print(f"출력 디렉토리 내용: {os.listdir(output_path)}")
        metrics = {'mAP': 0.0}
    
    ex_dict['Eval Results'] = metrics if metrics else {'mAP': 0.0}
    
    # 임시 데이터 파일 삭제
    if os.path.exists(temp_data_path):
        os.unlink(temp_data_path)
    
    return ex_dict

def test_msnet_model_cli(ex_dict):
    """
    MSNet 모델을 CLI로 테스트
    
    Args:
        ex_dict (dict): 테스트에 필요한 설정을 담은 딕셔너리
    """
    # 데이터 설정 파일 생성
    temp_data_path = create_msnet_data_config(ex_dict)
    
    # 출력 디렉토리 설정
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    output_path = os.path.join(ex_dict['Output Dir'], name)
    os.makedirs(output_path, exist_ok=True)
    
    # 테스트 스크립트
    test_script = os.path.join(MSNET_SOURCE_DIR, 'utils_coco', 'get_map_coco.py')
    
    # 경로 설정 (따옴표 제거)
    model_path = ex_dict["PT path"]
    
    # 체크포인트 파일 존재 확인 및 대체 파일 찾기 (수정된 부분)
    if not os.path.exists(model_path):
        print(f"체크포인트 파일을 찾을 수 없습니다: {model_path}")
        # 대체 파일들 확인
        output_path = os.path.join(ex_dict['Output Dir'], f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}")
        alternative_files = [
            os.path.join(output_path, 'last_epoch_weights.pth'),
            os.path.join(output_path, 'ep100-loss*.pth'),  # 마지막 epoch 파일
        ]
        
        found_model = False
        for alt_file in alternative_files:
            if '*' in alt_file:
                # 와일드카드 패턴 매칭
                import glob
                matching_files = glob.glob(alt_file)
                if matching_files:
                    # 가장 최근 파일 선택
                    model_path = max(matching_files, key=os.path.getctime)
                    found_model = True
                    print(f"대체 체크포인트 파일을 찾았습니다: {model_path}")
                    break
            elif os.path.exists(alt_file):
                model_path = alt_file
                found_model = True
                print(f"대체 체크포인트 파일을 찾았습니다: {model_path}")
                break
        
        if not found_model:
            print(f"사용 가능한 체크포인트 파일이 없습니다. 테스트를 건너뜁니다.")
            ex_dict['Test Results'] = {'mAP': 0.0}
            return ex_dict
    
    # 모델 경로를 절대 경로로 변환 (수정된 부분)
    model_path_abs = os.path.abspath(model_path)
    data_yaml = temp_data_path
    # 절대 경로로 변환 (수정된 부분)
    map_out_path = os.path.abspath(output_path)
    
    # MSNet 모델 크기 설정 (phi 파라미터)
    model_config = ex_dict.get('Model Config', 'yolov8_s')  # 기본값: s
    if 'yolov8_n' in model_config:
        phi = 'n'
    elif 'yolov8_s' in model_config:
        phi = 's'
    elif 'yolov8_m' in model_config:
        phi = 'm'
    elif 'yolov8_l' in model_config:
        phi = 'l'
    elif 'yolov8_x' in model_config:
        phi = 'x'
    else:
        phi = 's'  # 기본값
    
    print(f"[MSNet Test] Model Config: {model_config} → phi: {phi}")

    # input_shape는 개별 정수로 전달 (get_map_coco.py에서 nargs='+' 사용)
    cmd = [
        sys.executable,
        test_script,
        f"--model_path={model_path_abs}",
        f"--data_yaml={data_yaml}",
        f"--map_out_path={map_out_path}",
        "--input_shape", str(ex_dict['Image Size']), str(ex_dict['Image Size']),
        f"--phi={phi}",  # phi 파라미터 추가
        f"--confidence=0.3",  # 신뢰도 임계값을 0.3으로 설정
        f"--cuda={'True' if ex_dict['Device'] != 'cpu' else 'False'}"
    ]
    
    print(f"MSNet 테스트 명령어: {' '.join(cmd)}")
    
    # 환경 변수 설정
    env = os.environ.copy()
    env['PYTHONPATH'] = MSNET_SOURCE_DIR
    env['PYTHONUNBUFFERED'] = '1'
    
    try:
        # MSNet SourceFile 디렉토리에서 실행 (수정된 부분)
        
        process = subprocess.Popen(cmd, cwd=MSNET_SOURCE_DIR, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True, 
                                 bufsize=0, universal_newlines=True, env=env)
        
        # 실시간 출력
        stdout_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[MSNet] {output.strip()}")
                stdout_lines.append(output.strip())
        
        return_code = process.poll()
        print(f"MSNet 테스트 완료. 반환 코드: {return_code}")
        
    except Exception as e:
        print(f"MSNet 테스트 중 오류: {e}")
        return_code = 1
    
    # 결과 파싱
    # get_map_coco.py는 map_out_path 디렉토리에 results.txt를 생성
    results_path = os.path.join(output_path, 'results.txt')
    
    if os.path.exists(results_path):
        metrics = parse_msnet_results(results_path)
    else:
        print(f"결과 파일을 찾을 수 없습니다: {results_path}")
        # 디버깅을 위해 디렉토리 내용 확인
        if os.path.exists(output_path):
            print(f"출력 디렉토리 내용: {os.listdir(output_path)}")
        metrics = {'mAP': 0.0}
    
    ex_dict['Test Results'] = metrics if metrics else {'mAP': 0.0}
    
    # 임시 데이터 파일 삭제
    if os.path.exists(temp_data_path):
        os.unlink(temp_data_path)
    
    return ex_dict

def main():
    parser = argparse.ArgumentParser(description='MSNet CLI for model evaluation')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--map_out_path', type=str, default='map_out', help='Path to save evaluation results')
    args = parser.parse_args()

    # config 파일 읽기
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # get_map_coco.py 실행
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'SourceFile', 'utils_coco', 'get_map_coco.py'),
        '--model_path', args.model_path,
        '--data_yaml', args.data_yaml,
        '--map_out_path', args.map_out_path,
        '--confidence', str(config.get('confidence', 0.5)),
        '--nms_iou', str(config.get('nms_iou', 0.3)),
        '--input_shape', str(config.get('input_shape', [640, 640])),
        '--phi', config.get('phi', 'l'),
        '--cuda', str(config.get('cuda', True))
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 