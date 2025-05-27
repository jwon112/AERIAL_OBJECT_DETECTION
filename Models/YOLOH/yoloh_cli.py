import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import yaml

def extract_root_and_name_from_dataconfig(data_config_path):
    # yaml 파일이면 파싱
    if data_config_path.endswith('.yaml') or data_config_path.endswith('.yml'):
        with open(data_config_path, 'r') as f:
            y = yaml.safe_load(f)
        train_path = y.get('train', '')
        val_path = y.get('val', '')
        # 공통 상위 폴더(2단계 위)를 root로 추정
        root = os.path.dirname(os.path.dirname(train_path or val_path))
        # 데이터셋 이름은 yaml 파일명(확장자 제거)
        dataset_name = os.path.splitext(os.path.basename(data_config_path))[0]
        return root, dataset_name
    else:
        # 폴더 경로라면 마지막 폴더명을 name, 상위 폴더를 root로
        root = os.path.dirname(data_config_path)
        dataset_name = os.path.basename(data_config_path)
        return root, dataset_name

def build_yoloh_model_cli(cfg=None, ex_dict=None):
    """ex_dict를 사용하여 YOLOH 모델을 구성하지만 실제로 로드하지는 않음"""
    device = ex_dict.get('Device', 'cpu')
    return {
        'cfg': cfg,
        'device': device,
        'ex_dict': ex_dict
    }

def parse_results_txt(results_path):
    metrics = {}
    if not os.path.exists(results_path):
        return metrics
    with open(results_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                try:
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    metrics[key.strip()] = value.strip()
    return metrics

def train_yoloh_model_cli(ex_dict):
    """
    YOLOH 모델을 공식 CLI 명령어로 학습
    """
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    model_info = ex_dict['Model']
    cfg = model_info['cfg']
    # 출력 디렉토리 설정
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    output_path = os.path.join(ex_dict['Output Dir'], name)
    os.makedirs(output_path, exist_ok=True)
    # train.py 경로
    YOLOH_DIR = os.path.dirname(os.path.abspath(__file__))
    train_py = os.path.join(YOLOH_DIR, 'train.py')
    # 자동 추출
    if 'Dataset Root' not in ex_dict or 'Dataset Name' not in ex_dict:
        data_config = ex_dict.get('Data Config')
        root, dataset_name = extract_root_and_name_from_dataconfig(data_config)
    # 명령줄 인수 구성
    cmd = [
        sys.executable,
        train_py,
        '--cuda' if ex_dict.get('Device', 'cpu') != 'cpu' else '',
        f"-d", dataset_name,
        f"--root", root,
        f"-v", ex_dict['Model Name'],
        f"--batch_size", str(ex_dict['Batch Size']),
        f"--train_min_size", str(ex_dict['Image Size']),
        f"--train_max_size", str(ex_dict['Image Size']),
        f"--val_min_size", str(ex_dict['Image Size']),
        f"--val_max_size", str(ex_dict['Image Size']),
        f"--schedule", ex_dict.get('Schedule', '1x'),
        f"--grad_clip_norm", str(ex_dict.get('Grad Clip', 4.0)),
        f"--save_folder", output_path
    ]
    # 불필요한 빈 문자열 제거
    cmd = [c for c in cmd if c != '']
    print(f"실행 명령어: {' '.join(cmd)}")
    process = subprocess.run(cmd, cwd=YOLOH_DIR)
    # 결과 저장 경로
    pt_path = os.path.join(output_path, 'final.pth')
    ex_dict['PT path'] = pt_path
    # 결과 파일 파싱
    results_path = os.path.join(output_path, 'results.txt')
    metrics = parse_results_txt(results_path)
    ex_dict['Train Results'] = metrics if metrics else {'total_loss': 0.0}
    return ex_dict

def eval_yoloh_model_cli(ex_dict):
    """YOLOH 모델 검증 (COCO val 등)"""
    pt_path = ex_dict.get('PT path')
    if not pt_path or not os.path.exists(pt_path):
        print(f"Warning: Model weights not found at {pt_path}")
        ex_dict["Val Results"] = {
            "map": 0.0,
            "map50": 0.0,
            "mp": 0.0,
            "mr": 0.0
        }
        return ex_dict
    YOLOH_DIR = os.path.dirname(os.path.abspath(__file__))
    eval_py = os.path.join(YOLOH_DIR, 'eval.py')
    if 'Dataset Root' not in ex_dict or 'Dataset Name' not in ex_dict:
        data_config = ex_dict.get('Data Config')
        root, dataset_name = extract_root_and_name_from_dataconfig(data_config)
    cmd = [
        sys.executable,
        eval_py,
        '--cuda' if ex_dict.get('Device', 'cpu') != 'cpu' else '',
        f"-d", dataset_name,
        f"--root", root,
        f"-v", ex_dict['Model Name'],
        f"--weight", pt_path,
        f"--min_size", str(ex_dict['Image Size']),
        f"--max_size", str(ex_dict['Image Size'])
    ]
    cmd = [c for c in cmd if c != '']
    print(f"검증 명령어: {' '.join(cmd)}")
    process = subprocess.run(cmd, cwd=YOLOH_DIR)
    # 결과 저장 경로
    output_path = os.path.dirname(pt_path)
    # 결과 파일 파싱
    results_path = os.path.join(output_path, 'results.txt')
    metrics = parse_results_txt(results_path)
    ex_dict["Val Results"] = metrics if metrics else {
        "map": 0.0,
        "map50": 0.0,
        "mp": 0.0,
        "mr": 0.0
    }
    return ex_dict

def test_yoloh_model_cli(ex_dict):
    """YOLOH 모델 테스트 (COCO test 등)"""
    pt_path = ex_dict.get('PT path')
    if not pt_path or not os.path.exists(pt_path):
        print(f"Warning: Model weights not found at {pt_path}")
        ex_dict["Test Results"] = type('DetMetricsDict', (), {
            'box': type('BoxMetrics', (), {
                'map': 0.0,
                'map50': 0.0,
                'map75': 0.0,
                'mp': 0.0,
                'mr': 0.0,
                'ap_class_index': None
            })
        })
        return ex_dict
    YOLOH_DIR = os.path.dirname(os.path.abspath(__file__))
    test_py = os.path.join(YOLOH_DIR, 'test.py')
    if 'Dataset Root' not in ex_dict or 'Dataset Name' not in ex_dict:
        data_config = ex_dict.get('Data Config')
        root, dataset_name = extract_root_and_name_from_dataconfig(data_config)
    cmd = [
        sys.executable,
        test_py,
        '--cuda' if ex_dict.get('Device', 'cpu') != 'cpu' else '',
        f"-d", dataset_name,
        f"--root",  root,
        f"-v", ex_dict['Model Name'],
        f"--weight", pt_path,
        f"--min_size", str(ex_dict['Image Size']),
        f"--max_size", str(ex_dict['Image Size'])
    ]
    cmd = [c for c in cmd if c != '']
    print(f"테스트 명령어: {' '.join(cmd)}")
    process = subprocess.run(cmd, cwd=YOLOH_DIR)
    # 결과 저장 경로
    output_path = os.path.dirname(pt_path)
    # 결과 파일 파싱
    results_path = os.path.join(output_path, 'results.txt')
    metrics = parse_results_txt(results_path)
    if metrics:
        ex_dict["Test Results"] = metrics
    else:
        ex_dict["Test Results"] = type('DetMetricsDict', (), {
            'box': type('BoxMetrics', (), {
                'map': 0.0,
                'map50': 0.0,
                'map75': 0.0,
                'mp': 0.0,
                'mr': 0.0,
                'ap_class_index': None
            })
        })
    return ex_dict 