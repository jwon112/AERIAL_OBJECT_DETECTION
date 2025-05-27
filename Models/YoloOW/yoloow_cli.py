import os
import sys
import subprocess
from datetime import datetime
import torch
from pathlib import Path

# YOLOoW 폴더를 시스템 경로에 추가
YOLOOW_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(YOLOOW_DIR)

def parse_results_txt(results_path):
    metrics = {}
    if not os.path.exists(results_path):
        return metrics
    
    try:
        with open(results_path, 'r') as f:
            lines = f.readlines()
            
        # YoloOW results.txt는 보통 마지막 줄에 최종 결과가 있음
        if lines:
            # 마지막 줄 파싱 시도
            last_line = lines[-1].strip()
            print(f"results.txt 마지막 줄: {last_line}")
            
            # 공백으로 분리된 숫자들을 파싱
            values = last_line.split()
            if len(values) >= 7:  # epoch, gpu_mem, box, obj, cls, total, labels, img_size, P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                try:
                    # 일반적인 YoloOW 결과 형식
                    metrics = {
                        'precision': float(values[-7]) if len(values) > 7 else 0.0,
                        'recall': float(values[-6]) if len(values) > 6 else 0.0,
                        'map50': float(values[-5]) if len(values) > 5 else 0.0,
                        'map': float(values[-4]) if len(values) > 4 else 0.0,
                        'box_loss': float(values[-3]) if len(values) > 3 else 0.0,
                        'obj_loss': float(values[-2]) if len(values) > 2 else 0.0,
                        'cls_loss': float(values[-1]) if len(values) > 1 else 0.0,
                    }
                    metrics['total_loss'] = metrics['box_loss'] + metrics['obj_loss'] + metrics['cls_loss']
                except (ValueError, IndexError) as e:
                    print(f"숫자 파싱 에러: {e}")
            
            # 기존 key:value 형식도 시도
            for line in lines:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    try:
                        metrics[key.strip()] = float(value.strip())
                    except ValueError:
                        metrics[key.strip()] = value.strip()
                        
    except Exception as e:
        print(f"results.txt 파싱 에러: {e}")
    
    return metrics

def build_yoloow_model_cli(cfg=None, ex_dict=None):
    """ex_dict를 사용하여 YOLOoW 모델을 구성하지만 실제로 로드하지는 않음"""
    device = ex_dict.get('Device', 'cpu')
    
    # 이 함수에서는 모델 객체를 반환하지 않고, 설정 정보만 반환
    return {
        'cfg': cfg,
        'device': device,
        'ex_dict': ex_dict
    }

def train_yoloow_model_cli(ex_dict):
    """
    YOLOoW 모델을 공식 CLI 명령어로 학습
    
    기존 명령어 형식:
    python train.py --workers 8 --device 0 --batch-size 4 --data data/sea_drones_see.yaml 
                    --img 1280 --cfg cfg/training/yoloOW.yaml --weights 'yoloow.pt' 
                    --name yoloOW --hyp data/hyp.scratch.sea.yaml --epoch 300
    """
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    model_info = ex_dict['Model']
    cfg = model_info['cfg']
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    output_path = os.path.join(ex_dict['Output Dir'], name)
    os.makedirs(output_path, exist_ok=True)
    cfg_path = os.path.join(YOLOOW_DIR, 'cfg', 'training', cfg)
    hyp_path = os.path.join(YOLOOW_DIR, 'data', 'hyp.scratch.yml')
    cmd = [
        "python",  # sys.executable 대신 python 명령어 사용
        os.path.join(YOLOOW_DIR, 'train.py'),
        f"--workers={ex_dict.get('Num Workers', 0)}",
        f"--device={ex_dict['Device'] if isinstance(ex_dict['Device'], int) else 0}",
        f"--batch-size={ex_dict['Batch Size']}",
        f"--data={ex_dict['Data Config']}",
        f"--img-size={ex_dict['Image Size']}",
        f"--cfg={cfg_path}",
        f"--weights=",  # 빈 문자열로 설정
        f"--name={name}",
        f"--epochs={ex_dict['Epochs']}",
        f"--project={ex_dict['Output Dir']}",
        f"--hyp={hyp_path}",
    ]
    
    # Adam optimizer 설정 (boolean flag로 처리)
    if ex_dict['Optimizer'] == 'AdamW':
        cmd.append("--adam")
    
    print(f"실행 명령어: {' '.join(cmd)}")
    
    # 더 자세한 디버깅을 위해 실시간 출력 확인
    print("학습 시작...")
    process = subprocess.Popen(cmd, cwd=YOLOOW_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    
    # 실시간으로 출력 확인
    stdout_lines = []
    stderr_lines = []
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(f"[STDOUT] {output.strip()}")
            stdout_lines.append(output.strip())
    
    # 에러 출력 확인
    stderr_output = process.stderr.read()
    if stderr_output:
        print(f"[STDERR] {stderr_output}")
        stderr_lines.append(stderr_output)
    
    return_code = process.poll()
    print(f"학습 프로세스 종료 코드: {return_code}")
    
    # 학습이 성공적으로 완료되었는지 확인
    if return_code == 0:
        print("학습이 성공적으로 완료되었습니다!")
    else:
        print(f"학습이 실패했습니다. 종료 코드: {return_code}")
        
    # 출력 디렉토리 확인
    output_dir = os.path.join(ex_dict['Output Dir'], name)
    print(f"출력 디렉토리: {output_dir}")
    
    if os.path.exists(output_dir):
        print(f"출력 디렉토리 내용: {os.listdir(output_dir)}")
        
        # weights 디렉토리 확인
        weights_dir = os.path.join(output_dir, 'weights')
        if os.path.exists(weights_dir):
            print(f"weights 디렉토리 내용: {os.listdir(weights_dir)}")
        else:
            print("weights 디렉토리가 존재하지 않습니다.")
            
        # 다른 파일들도 확인
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                print(f"생성된 파일: {file_path}")
    else:
        print("출력 디렉토리가 생성되지 않았습니다.")
    
    pt_path = os.path.join(ex_dict['Output Dir'], name, 'weights', 'best.pt')
    ex_dict['PT path'] = pt_path
    
    print(f"예상 best.pt 경로: {pt_path}")
    print(f"best.pt 파일 존재 여부: {os.path.exists(pt_path)}")
    
    # last.pt 파일도 확인
    last_pt_path = os.path.join(ex_dict['Output Dir'], name, 'weights', 'last.pt')
    print(f"last.pt 파일 존재 여부: {os.path.exists(last_pt_path)}")
    
    results_path = os.path.join(ex_dict['Output Dir'], name, 'results.txt')
    metrics = parse_results_txt(results_path)
    print(f"results.txt 경로: {results_path}")
    print(f"results.txt 존재 여부: {os.path.exists(results_path)}")
    print(f"파싱된 메트릭: {metrics}")
    
    ex_dict['Train Results'] = metrics if metrics else {
        'box_loss': 0.0,
        'obj_loss': 0.0,
        'cls_loss': 0.0,
        'total_loss': 0.0
    }
    return ex_dict

def eval_yoloow_model_cli(ex_dict):
    """YOLOoW 모델 검증"""
    # 학습된 모델 경로
    pt_path = ex_dict.get('PT path')
    if not pt_path or not os.path.exists(pt_path):
        print(f"Warning: Model weights not found at {pt_path}")
        # DetMetrics 구조를 모방한 결과 객체 생성
        ex_dict["Val Results"] = {
            "box_loss": 0.0,
            "obj_loss": 0.0,
            "cls_loss": 0.0,
            "total_loss": 0.0,
        }
        return ex_dict
    
    # 검증 명령어 구성
    cmd = [
        sys.executable,
        os.path.join(YOLOOW_DIR, 'test.py'),
        f"--weights={pt_path}",
        f"--data={ex_dict['Data Config']}",
        f"--batch-size={ex_dict['Batch Size']}",
        f"--img={ex_dict['Image Size']}",
        f"--task=val",
        f"--device={ex_dict['Device'] if isinstance(ex_dict['Device'], int) else 0}",
    ]
    
    print(f"검증 명령어: {' '.join(cmd)}")
    process = subprocess.run(cmd, cwd=YOLOOW_DIR)
    
    # 검증 결과 저장
    output_path = os.path.dirname(pt_path) if pt_path else ex_dict['Output Dir']
    results_path = os.path.join(output_path, 'results.txt')
    metrics = parse_results_txt(results_path)
    ex_dict["Val Results"] = metrics if metrics else {
        "box_loss": 0.0,
        "obj_loss": 0.0,
        "cls_loss": 0.0,
        "total_loss": 0.0,
    }
    
    return ex_dict

def test_yoloow_model_cli(ex_dict):
    """YOLOoW 모델 테스트"""
    # 학습된 모델 경로
    pt_path = ex_dict.get('PT path')
    if not pt_path or not os.path.exists(pt_path):
        print(f"Warning: Model weights not found at {pt_path}")
        # DetMetrics 구조를 모방한 결과 객체 생성
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
    
    # 테스트 명령어 구성
    cmd = [
        sys.executable,
        os.path.join(YOLOOW_DIR, 'test.py'),
        f"--weights={pt_path}",
        f"--data={ex_dict['Data Config']}",
        f"--batch-size={ex_dict['Batch Size']}",
        f"--img={ex_dict['Image Size']}",
        f"--task=test",
        f"--device={ex_dict['Device'] if isinstance(ex_dict['Device'], int) else 0}",
    ]
    
    print(f"테스트 명령어: {' '.join(cmd)}")
    process = subprocess.run(cmd, cwd=YOLOOW_DIR)
    
    # DetMetrics 구조를 모방한 결과 객체 생성
    output_path = os.path.dirname(pt_path) if pt_path else ex_dict['Output Dir']
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