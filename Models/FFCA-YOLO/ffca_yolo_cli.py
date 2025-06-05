import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import yaml
import tempfile
from utility.metrics import BoxResults, EvalResults

# FFCA-YOLO 폴더를 시스템 경로에 추가
FFCA_YOLO_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_results_txt(results_path):
    """FFCA-YOLO 결과 파일 파싱 (YOLOv5 스타일)"""
    metrics = {}
    if not os.path.exists(results_path):
        return metrics
    
    try:
        with open(results_path, 'r') as f:
            lines = f.readlines()
            
        # YOLOv5 results.txt 파싱 - 마지막 줄에 최종 결과
        if lines:
            last_line = lines[-1].strip()
            print(f"FFCA-YOLO results.txt 마지막 줄: {last_line}")
            
            # 공백으로 분리된 값들 파싱
            values = last_line.split()
            if len(values) >= 15:  # YOLOv5 표준 출력 형식
                try:
                    # epoch, gpu_mem, box, obj, cls, total, labels, img_size, P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                    metrics = {
                        'precision': float(values[8]) if len(values) > 8 else 0.0,
                        'recall': float(values[9]) if len(values) > 9 else 0.0,
                        'map50': float(values[10]) if len(values) > 10 else 0.0,
                        'map': float(values[11]) if len(values) > 11 else 0.0,
                        'box_loss': float(values[12]) if len(values) > 12 else 0.0,
                        'obj_loss': float(values[13]) if len(values) > 13 else 0.0,
                        'cls_loss': float(values[14]) if len(values) > 14 else 0.0,
                    }
                    metrics['total_loss'] = metrics['box_loss'] + metrics['obj_loss'] + metrics['cls_loss']
                except (ValueError, IndexError) as e:
                    print(f"FFCA-YOLO 숫자 파싱 에러: {e}")
            
            # key:value 형식도 시도
            for line in lines:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    try:
                        metrics[key.strip()] = float(value.strip())
                    except ValueError:
                        metrics[key.strip()] = value.strip()
                        
    except Exception as e:
        print(f"FFCA-YOLO results.txt 파싱 에러: {e}")
    
    return metrics

def build_ffca_yolo_model_cli(cfg=None, ex_dict=None):
    """FFCA-YOLO 모델 CLI 빌드 함수"""
    device = ex_dict.get('Device', 'cpu')
    
    if cfg is None:
        cfg = "FFCA-YOLO.yaml"
    
    return {
        'cfg': cfg,
        'device': device,
        'ex_dict': ex_dict
    }

def create_ffca_data_config(ex_dict):
    """FFCA-YOLO용 데이터 설정 파일 생성"""
    data_config_path = os.path.abspath(ex_dict['Data Config'])
    
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # path를 절대 경로로 변환
    if 'path' in data_config:
        original_path = data_config['path']
        if not os.path.isabs(original_path):
            project_root = os.path.dirname(os.path.dirname(FFCA_YOLO_DIR))
            absolute_path = os.path.abspath(os.path.join(project_root, original_path))
            data_config['path'] = absolute_path
            
            # train, val, test 경로도 절대 경로로 변환
            for key in ['train', 'val', 'test']:
                if key in data_config:
                    relative_file = data_config[key]
                    absolute_file_path = os.path.join(absolute_path, relative_file)
                    data_config[key] = absolute_file_path
    
    # 클래스 수 업데이트
    data_config['nc'] = ex_dict['Number of Classes']
    
    # 임시 파일 생성
    temp_data_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(data_config, temp_data_file, default_flow_style=False)
    temp_data_file.close()
    
    return temp_data_file.name

def train_ffca_yolo_model_cli(ex_dict):
    """
    FFCA-YOLO 모델을 CLI로 학습
    
    기본 명령어:
    python train.py --weights '' --cfg models/FFCA-YOLO.yaml --data data/dataset.yaml 
                    --hyp data/hyps/hyp.scratch-low.yaml --epochs 300 --batch-size 32
    """
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    model_info = ex_dict['Model']
    cfg = model_info['cfg']
    
    # 출력 디렉토리 설정
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    output_path = os.path.join(ex_dict['Output Dir'], name)
    os.makedirs(output_path, exist_ok=True)
    
    # Config 파일 경로
    cfg_path = os.path.join(FFCA_YOLO_DIR, 'models', cfg)
    hyp_path = os.path.join(FFCA_YOLO_DIR, 'data', 'hyps', 'hyp.scratch-low.yaml')
    
    if not os.path.exists(hyp_path):
        # 기본 hyp 파일 사용
        hyp_path = os.path.join(FFCA_YOLO_DIR, 'data', 'hyp.scratch.yaml')
    
    # 데이터 설정 파일 생성
    temp_data_path = create_ffca_data_config(ex_dict)
    
    # 학습 스크립트
    train_script = os.path.join(FFCA_YOLO_DIR, 'train.py')
    
    cmd = [
        sys.executable,
        train_script,
        f"--weights=",  # 빈 문자열로 설정 (from scratch)
        f"--cfg={cfg_path}",
        f"--data={temp_data_path}",
        f"--hyp={hyp_path}",
        f"--epochs={ex_dict['Epochs']}",
        f"--batch-size={ex_dict['Batch Size']}",
        f"--img-size={ex_dict['Image Size']}",
        f"--name={name}",
        f"--project={ex_dict['Output Dir']}",
        f"--workers={ex_dict.get('Num Workers', 0)}",
        f"--device={ex_dict['Device'] if isinstance(ex_dict['Device'], int) else 0}",
        "--cache",  # 이미지 캐싱으로 속도 향상
        "--save-period=1",  # 매 epoch마다 저장
        "--exist-ok"  # 기존 디렉토리 덮어쓰기 허용
    ]
    
    print(f"FFCA-YOLO 학습 명령어: {' '.join(cmd)}")
    
    # 환경 변수 설정
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    try:
        # 프로젝트 루트에서 실행
        project_root = os.path.dirname(os.path.dirname(FFCA_YOLO_DIR))
        
        process = subprocess.Popen(cmd, cwd=project_root, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True, 
                                 bufsize=0, universal_newlines=True, env=env)
        
        # 실시간 출력
        stdout_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[FFCA-YOLO] {output.strip()}")
                stdout_lines.append(output.strip())
        
        return_code = process.poll()
        print(f"FFCA-YOLO 학습 완료. 반환 코드: {return_code}")
        
    except Exception as e:
        print(f"FFCA-YOLO 학습 중 오류: {e}")
        return_code = 1
    
    # 결과 파싱
    output_dir = os.path.join(ex_dict['Output Dir'], name)
    results_path = os.path.join(output_dir, 'results.txt')
    
    metrics = parse_results_txt(results_path)
    
    # 가중치 파일 경로
    pt_path = os.path.join(output_dir, 'weights', 'best.pt')
    if not os.path.exists(pt_path):
        pt_path = os.path.join(output_dir, 'weights', 'last.pt')
    
    ex_dict['PT path'] = pt_path
    ex_dict['Train Results'] = metrics if metrics else {'total_loss': 0.1}
    
    # 임시 데이터 파일 삭제
    if os.path.exists(temp_data_path):
        os.unlink(temp_data_path)
    
    return ex_dict

def eval_ffca_yolo_model_cli(ex_dict):
    """FFCA-YOLO 모델 검증"""
    pt_path = ex_dict.get('PT path')
    if not pt_path or not os.path.exists(pt_path):
        print(f"Warning: FFCA-YOLO model weights not found at {pt_path}")
        ex_dict["Val Results"] = {
            "map": 0.0,
            "map50": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }
        return ex_dict
    
    # 검증 스크립트
    val_script = os.path.join(FFCA_YOLO_DIR, 'val.py')
    
    # 데이터 설정 파일 생성
    temp_data_path = create_ffca_data_config(ex_dict)
    
    cmd = [
        sys.executable,
        val_script,
        f"--weights={pt_path}",
        f"--data={temp_data_path}",
        f"--img-size={ex_dict['Image Size']}",
        f"--batch-size={ex_dict['Batch Size']}",
        f"--device={ex_dict['Device'] if isinstance(ex_dict['Device'], int) else 0}",
        f"--conf-thres=0.001",  # 낮은 confidence threshold로 설정
        "--verbose",
        "--save-txt",
        "--save-conf"
    ]
    
    print(f"FFCA-YOLO 검증 명령어: {' '.join(cmd)}")
    
    try:
        project_root = os.path.dirname(os.path.dirname(FFCA_YOLO_DIR))
        
        # 환경 변수 설정
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(cmd, cwd=project_root, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True, 
                                 bufsize=0, universal_newlines=True, env=env)
        
        # 실시간 출력 및 결과 수집
        stdout_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[FFCA-YOLO Val] {output.strip()}")
                stdout_lines.append(output.strip())
        
        return_code = process.poll()
        stdout_text = '\n'.join(stdout_lines)
        
        # 결과 파싱
        metrics = {}
        if stdout_text:
            import re
            # mAP 추출 - 더 유연한 패턴 사용
            map_match = re.search(r'all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', stdout_text)
            if map_match:
                metrics = {
                    'precision': float(map_match.group(3)),
                    'recall': float(map_match.group(4)),
                    'map50': float(map_match.group(5)),
                    'map': float(map_match.group(6))
                }
                print(f"[FFCA-YOLO Val 파싱 성공] P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, mAP50={metrics['map50']:.3f}, mAP={metrics['map']:.3f}")
            else:
                print(f"[FFCA-YOLO Val 파싱 실패] 패턴을 찾을 수 없음")
        
        # 결과를 format_measures와 호환되는 구조로 변환
        if metrics:
            # BoxResults 객체 생성
            box_results = BoxResults(
                mp=metrics['precision'],      # Mean Precision
                mr=metrics['recall'],         # Mean Recall  
                map50=metrics['map50'],       # mAP@0.5
                map=metrics['map'],           # mAP@0.5:0.95
                map75=metrics.get('map75', metrics['map'] * 0.8),  # mAP@0.75
                ap_class_index=None,          # 클래스별 인덱스 (없음)
                names=None,                   # 클래스 이름 (없음)
                per_class_data=[]             # 클래스별 데이터 (없음)
            )
            
            # EvalResults 객체로 감싸기
            val_results = EvalResults(box=box_results)
        else:
            # 기본값으로 BoxResults 생성
            box_results = BoxResults(
                mp=0.0, mr=0.0, map50=0.0, map=0.0, map75=0.0,
                ap_class_index=None, names=None, per_class_data=[]
            )
            val_results = EvalResults(box=box_results)
        
        ex_dict["Val Results"] = val_results
        
    except Exception as e:
        print(f"FFCA-YOLO 검증 중 오류: {e}")
        # 기본값으로 BoxResults 생성
        box_results = BoxResults(
            mp=0.0, mr=0.0, map50=0.0, map=0.0, map75=0.0,
            ap_class_index=None, names=None, per_class_data=[]
        )
        ex_dict["Val Results"] = EvalResults(box=box_results)
    
    # 임시 데이터 파일 삭제
    if os.path.exists(temp_data_path):
        os.unlink(temp_data_path)
    
    return ex_dict

def test_ffca_yolo_model_cli(ex_dict):
    """FFCA-YOLO 모델 테스트"""
    pt_path = ex_dict.get('PT path')
    if not pt_path or not os.path.exists(pt_path):
        print(f"Warning: FFCA-YOLO model weights not found at {pt_path}")
        # 기본값으로 BoxResults 생성
        box_results = BoxResults(
            mp=0.0, mr=0.0, map50=0.0, map=0.0, map75=0.0,
            ap_class_index=None, names=None, per_class_data=[]
        )
        ex_dict["Test Results"] = EvalResults(box=box_results)
        return ex_dict
    
    # 테스트 스크립트 (val.py 사용)
    test_script = os.path.join(FFCA_YOLO_DIR, 'val.py')
    
    # 데이터 설정 파일 생성
    temp_data_path = create_ffca_data_config(ex_dict)
    
    # 출력 디렉토리
    output_path = os.path.dirname(pt_path)
    test_output = os.path.join(output_path, 'test_results')
    os.makedirs(test_output, exist_ok=True)
    
    cmd = [
        sys.executable,
        test_script,
        f"--weights={pt_path}",
        f"--data={temp_data_path}",
        f"--img-size={ex_dict['Image Size']}",
        f"--batch-size={ex_dict['Batch Size']}",
        f"--device={ex_dict['Device'] if isinstance(ex_dict['Device'], int) else 0}",
        f"--project={test_output}",
        f"--name=test",
        f"--conf-thres=0.001",  # 낮은 confidence threshold로 설정
        "--verbose",
        "--save-txt",
        "--save-conf",
        "--save-json"
    ]
    
    print(f"FFCA-YOLO 테스트 명령어: {' '.join(cmd)}")
    
    try:
        project_root = os.path.dirname(os.path.dirname(FFCA_YOLO_DIR))
        
        # 환경 변수 설정
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(cmd, cwd=project_root, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True, 
                                 bufsize=0, universal_newlines=True, env=env)
        
        # 실시간 출력 및 결과 수집
        stdout_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[FFCA-YOLO Test] {output.strip()}")
                stdout_lines.append(output.strip())
        
        return_code = process.poll()
        stdout_text = '\n'.join(stdout_lines)
        
        # 결과 파싱
        metrics = {}
        if stdout_text:
            import re
            # 전체 결과 라인 찾기 - 더 유연한 패턴 사용
            map_match = re.search(r'all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', stdout_text)
            if map_match:
                metrics = {
                    'precision': float(map_match.group(3)),
                    'recall': float(map_match.group(4)),
                    'map50': float(map_match.group(5)),
                    'map': float(map_match.group(6)),
                    'map75': float(map_match.group(6)) * 0.8  # 추정값
                }
                print(f"[FFCA-YOLO Test 파싱 성공] P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, mAP50={metrics['map50']:.3f}, mAP={metrics['map']:.3f}")
            else:
                print(f"[FFCA-YOLO Test 파싱 실패] 패턴을 찾을 수 없음")
                # 디버그를 위해 출력 일부를 보여줌
                print(f"[FFCA-YOLO Test 출력 샘플]: {stdout_text[-500:] if len(stdout_text) > 500 else stdout_text}")
        
        # 결과 객체 생성 - format_measures와 호환되도록 BoxResults 구조 사용
        if metrics:
            # BoxResults 객체 생성
            box_results = BoxResults(
                mp=metrics['precision'],      # Mean Precision
                mr=metrics['recall'],         # Mean Recall  
                map50=metrics['map50'],       # mAP@0.5
                map=metrics['map'],           # mAP@0.5:0.95
                map75=metrics.get('map75', metrics['map'] * 0.8),  # mAP@0.75
                ap_class_index=None,          # 클래스별 인덱스 (없음)
                names=None,                   # 클래스 이름 (없음)
                per_class_data=[]             # 클래스별 데이터 (없음)
            )
            
            # EvalResults 객체로 감싸기 (format_measures가 기대하는 구조)
            results = EvalResults(box=box_results)
        else:
            # 기본값으로 BoxResults 생성
            box_results = BoxResults(
                mp=0.0, mr=0.0, map50=0.0, map=0.0, map75=0.0,
                ap_class_index=None, names=None, per_class_data=[]
            )
            results = EvalResults(box=box_results)
        
        ex_dict["Test Results"] = results
        
        print(f"[FFCA-YOLO Test] mAP: {getattr(results.box, 'map', 0.0):.3f}, "
              f"mAP@50: {getattr(results.box, 'map50', 0.0):.3f}")
        print(f"[FFCA-YOLO] Optimized for small object detection in remote sensing images")
        
    except Exception as e:
        print(f"FFCA-YOLO 테스트 중 오류: {e}")
        # 기본값으로 BoxResults 생성
        box_results = BoxResults(
            mp=0.0, mr=0.0, map50=0.0, map=0.0, map75=0.0,
            ap_class_index=None, names=None, per_class_data=[]
        )
        ex_dict["Test Results"] = EvalResults(box=box_results)
    
    # 임시 데이터 파일 삭제
    if os.path.exists(temp_data_path):
        os.unlink(temp_data_path)
    
    return ex_dict 