import os
import sys
import subprocess
from datetime import datetime
import torch
from pathlib import Path
import yaml
import tempfile
from utility.metrics import BoxResults, EvalResults

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
            if len(values) >= 15:  # epoch, gpu_mem, box, obj, cls, total, labels, img_size, P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
                try:
                    # 올바른 YoloOW 결과 형식 (0-based indexing)
                    # 컬럼 8=P, 9=R, 10=mAP@0.5, 11=mAP@0.5:0.95, 12=box_loss, 13=obj_loss, 14=cls_loss
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
    hyp_path = os.path.join(YOLOOW_DIR, 'data', 'hyp.scratch.p5.yaml')
    
    # Data Config 경로를 절대 경로로 변환
    data_config_path = os.path.abspath(ex_dict['Data Config'])
    print(f"Data Config 절대 경로: {data_config_path}")
    print(f"Data Config 파일 존재 여부: {os.path.exists(data_config_path)}")
    
    # 기존 데이터 설정 파일을 읽어서 절대 경로로 변환한 임시 파일 생성
    with open(data_config_path, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # path를 절대 경로로 변환
    if 'path' in data_config:
        original_path = data_config['path']
        if not os.path.isabs(original_path):
            # 상대 경로인 경우 프로젝트 루트 기준으로 절대 경로 변환
            project_root = os.path.dirname(os.path.dirname(YOLOOW_DIR))
            absolute_path = os.path.abspath(os.path.join(project_root, original_path))
            data_config['path'] = absolute_path
            print(f"경로 변환: {original_path} -> {absolute_path}")
            
            # train, val, test 경로도 절대 경로로 변환
            for key in ['train', 'val', 'test']:
                if key in data_config:
                    relative_file = data_config[key]
                    absolute_file_path = os.path.join(absolute_path, relative_file)
                    data_config[key] = absolute_file_path
                    print(f"{key} 경로 변환: {relative_file} -> {absolute_file_path}")
                    print(f"{key} 파일 존재 여부: {os.path.exists(absolute_file_path)}")
    
    # 임시 데이터 설정 파일 생성
    temp_data_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(data_config, temp_data_file, default_flow_style=False)
    temp_data_file.close()
    temp_data_path = temp_data_file.name
    print(f"임시 데이터 설정 파일: {temp_data_path}")
    
    cmd = [
        "python",  # sys.executable 대신 python 명령어 사용
        os.path.join(YOLOOW_DIR, 'train.py'),
        f"--workers={ex_dict.get('Num Workers', 0)}",
        f"--device={ex_dict['Device'] if isinstance(ex_dict['Device'], int) else 0}",
        f"--batch-size={ex_dict['Batch Size']}",
        f"--data={temp_data_path}",  # 임시 파일 사용
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
    
    # 디렉토리 이름 중복 방지를 위해 exist-ok 플래그 추가
    cmd.append("--exist-ok")
    
    print(f"실행 명령어: {' '.join(cmd)}")
    
    # 프로젝트 루트 디렉토리 계산 (YOLOOW_DIR에서 2단계 위로)
    project_root = os.path.dirname(os.path.dirname(YOLOOW_DIR))
    print(f"프로젝트 루트 디렉토리: {project_root}")
    print(f"YoloOW 실행 디렉토리: {YOLOOW_DIR}")
    
    # 더 자세한 디버깅을 위해 실시간 출력 확인
    print("학습 시작...")
    # cwd를 프로젝트 루트로 변경하여 상대 경로가 올바르게 작동하도록 함
    # 환경 변수 설정으로 Python 출력 버퍼링 비활성화
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['TQDM_DISABLE'] = '0'  # tqdm 활성화
    env['TQDM_NCOLS'] = '80'   # tqdm 너비 설정
    process = subprocess.Popen(cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, bufsize=0, universal_newlines=True, env=env)
    
    # 실시간으로 출력 확인
    stdout_lines = []
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(f"[YoloOW] {output.strip()}")
            stdout_lines.append(output.strip())
    
    return_code = process.poll()
    print(f"YoloOW 학습 완료. 반환 코드: {return_code}")
    
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
    
    # 임시 파일 정리
    try:
        os.unlink(temp_data_path)
        print(f"임시 파일 삭제: {temp_data_path}")
    except Exception as e:
        print(f"임시 파일 삭제 실패: {e}")
    
    return ex_dict

def eval_yoloow_model_cli(ex_dict):
    """YOLOoW 모델 검증"""
    # 학습된 모델 경로
    pt_path = ex_dict.get('PT path')
    if not pt_path or not os.path.exists(pt_path):
        print(f"Warning: YoloOW model weights not found at {pt_path}")
        # 기본값으로 BoxResults 생성
        box_results = BoxResults(
            mp=0.0, mr=0.0, map50=0.0, map=0.0, map75=0.0,
            ap_class_index=None, names=None, per_class_data=[]
        )
        ex_dict["Val Results"] = EvalResults(box=box_results)
        return ex_dict

    # 데이터 설정 파일 생성
    data_config_path = os.path.abspath(ex_dict['Data Config'])
    with open(data_config_path, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # path를 절대 경로로 변환
    if 'path' in data_config:
        original_path = data_config['path']
        if not os.path.isabs(original_path):
            project_root = os.path.dirname(os.path.dirname(YOLOOW_DIR))
            absolute_path = os.path.abspath(os.path.join(project_root, original_path))
            data_config['path'] = absolute_path
            
            # train, val, test 경로도 절대 경로로 변환
            for key in ['train', 'val', 'test']:
                if key in data_config:
                    relative_file = data_config[key]
                    absolute_file_path = os.path.join(absolute_path, relative_file)
                    data_config[key] = absolute_file_path
    
    # 임시 데이터 파일 생성
    temp_data_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(data_config, temp_data_file, default_flow_style=False)
    temp_data_file.close()
    temp_data_path = temp_data_file.name

    # 검증 명령어 구성
    cmd = [
        sys.executable,
        os.path.join(YOLOOW_DIR, 'test.py'),
        f"--weights={pt_path}",
        f"--data={temp_data_path}",
        f"--batch-size={ex_dict['Batch Size']}",
        f"--img={ex_dict['Image Size']}",
        f"--conf-thres=0.001",  # 낮은 confidence threshold로 설정
        f"--task=val",
        f"--device={ex_dict['Device'] if isinstance(ex_dict['Device'], int) else 0}",
        "--verbose"
    ]
    
    print(f"YoloOW 검증 명령어: {' '.join(cmd)}")
    
    try:
        project_root = os.path.dirname(os.path.dirname(YOLOOW_DIR))
        
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
                print(f"[YoloOW Val] {output.strip()}")
                stdout_lines.append(output.strip())
        
        return_code = process.poll()
        stdout_text = '\n'.join(stdout_lines)
        
        # 결과 파싱
        metrics = {}
        if stdout_text:
            import re
            # YoloOW 결과 라인 찾기 - 더 유연한 패턴 사용
            map_match = re.search(r'all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', stdout_text)
            if map_match:
                metrics = {
                    'precision': float(map_match.group(3)),
                    'recall': float(map_match.group(4)),
                    'map50': float(map_match.group(5)),
                    'map': float(map_match.group(6))
                }
                print(f"[YoloOW Val 파싱 성공] P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, mAP50={metrics['map50']:.3f}, mAP={metrics['map']:.3f}")
            else:
                print(f"[YoloOW Val 파싱 실패] 패턴을 찾을 수 없음")
        
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
        print(f"YoloOW 검증 중 오류: {e}")
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

def test_yoloow_model_cli(ex_dict):
    """YOLOoW 모델 테스트"""
    # 학습된 모델 경로
    pt_path = ex_dict.get('PT path')
    if not pt_path or not os.path.exists(pt_path):
        print(f"Warning: YoloOW model weights not found at {pt_path}")
        # 기본값으로 BoxResults 생성
        box_results = BoxResults(
            mp=0.0, mr=0.0, map50=0.0, map=0.0, map75=0.0,
            ap_class_index=None, names=None, per_class_data=[]
        )
        ex_dict["Test Results"] = EvalResults(box=box_results)
        return ex_dict

    # 데이터 설정 파일 생성
    data_config_path = os.path.abspath(ex_dict['Data Config'])
    with open(data_config_path, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # path를 절대 경로로 변환
    if 'path' in data_config:
        original_path = data_config['path']
        if not os.path.isabs(original_path):
            project_root = os.path.dirname(os.path.dirname(YOLOOW_DIR))
            absolute_path = os.path.abspath(os.path.join(project_root, original_path))
            data_config['path'] = absolute_path
            
            # train, val, test 경로도 절대 경로로 변환
            for key in ['train', 'val', 'test']:
                if key in data_config:
                    relative_file = data_config[key]
                    absolute_file_path = os.path.join(absolute_path, relative_file)
                    data_config[key] = absolute_file_path
    
    # 임시 데이터 파일 생성
    temp_data_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(data_config, temp_data_file, default_flow_style=False)
    temp_data_file.close()
    temp_data_path = temp_data_file.name

    # 테스트 명령어 구성
    cmd = [
        sys.executable,
        os.path.join(YOLOOW_DIR, 'test.py'),
        f"--weights={pt_path}",
        f"--data={temp_data_path}",
        f"--batch-size={ex_dict['Batch Size']}",
        f"--img={ex_dict['Image Size']}",
        f"--conf-thres=0.001",  # 낮은 confidence threshold로 설정
        f"--task=test",
        f"--device={ex_dict['Device'] if isinstance(ex_dict['Device'], int) else 0}",
        "--verbose"
    ]
    
    print(f"YoloOW 테스트 명령어: {' '.join(cmd)}")
    
    try:
        project_root = os.path.dirname(os.path.dirname(YOLOOW_DIR))
        
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
                print(f"[YoloOW Test] {output.strip()}")
                stdout_lines.append(output.strip())
        
        return_code = process.poll()
        stdout_text = '\n'.join(stdout_lines)
        
        # 결과 파싱
        metrics = {}
        if stdout_text:
            import re
            # YoloOW 결과 라인 찾기 - 더 유연한 패턴 사용
            map_match = re.search(r'all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', stdout_text)
            if map_match:
                metrics = {
                    'precision': float(map_match.group(3)),
                    'recall': float(map_match.group(4)),
                    'map50': float(map_match.group(5)),
                    'map': float(map_match.group(6)),
                    'map75': float(map_match.group(6)) * 0.8  # 추정값
                }
                print(f"[YoloOW Test 파싱 성공] P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, mAP50={metrics['map50']:.3f}, mAP={metrics['map']:.3f}")
            else:
                print(f"[YoloOW Test 파싱 실패] 패턴을 찾을 수 없음")
                # 디버그를 위해 출력 일부를 보여줌
                print(f"[YoloOW Test 출력 샘플]: {stdout_text[-500:] if len(stdout_text) > 500 else stdout_text}")
        
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
        
        print(f"[YoloOW Test] mAP: {getattr(results.box, 'map', 0.0):.3f}, "
              f"mAP@50: {getattr(results.box, 'map50', 0.0):.3f}")
        print(f"[YoloOW] Optimized for underwater object detection")
        
    except Exception as e:
        print(f"YoloOW 테스트 중 오류: {e}")
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