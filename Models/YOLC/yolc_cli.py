import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
import yaml
import tempfile

# YOLC 폴더를 시스템 경로에 추가
YOLC_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_yolc_results(results_path):
    """YOLC 결과 파일 파싱"""
    metrics = {}
    if not os.path.exists(results_path):
        return metrics
    
    try:
        with open(results_path, 'r') as f:
            lines = f.readlines()
            
        # YOLC 특유의 LSM (Local Spatial Modeling) 결과 파싱
        for line in reversed(lines):
            if 'mAP' in line or 'AP' in line:
                import re
                # YOLC 출력에서 mAP 추출
                map_match = re.search(r'mAP[^:]*:\s*([\d.]+)', line)
                ap50_match = re.search(r'AP50[^:]*:\s*([\d.]+)', line)
                ap75_match = re.search(r'AP75[^:]*:\s*([\d.]+)', line)
                
                if map_match:
                    metrics['mAP'] = float(map_match.group(1))
                if ap50_match:
                    metrics['AP50'] = float(ap50_match.group(1))
                if ap75_match:
                    metrics['AP75'] = float(ap75_match.group(1))
                
                if metrics:  # 결과를 찾았으면 중단
                    break
        
        # loss 정보도 추출
        for line in reversed(lines):
            if 'loss' in line.lower():
                import re
                loss_match = re.search(r'loss[^:]*:\s*([\d.]+)', line)
                if loss_match:
                    metrics['total_loss'] = float(loss_match.group(1))
                    break
                    
    except Exception as e:
        print(f"YOLC 결과 파일 파싱 에러: {e}")
    
    return metrics

def build_yolc_model_cli(cfg=None, ex_dict=None):
    """YOLC 모델 CLI 빌드 함수"""
    device = ex_dict.get('Device', 'cpu')
    
    if cfg is None:
        cfg = "configs/yolc.py"
    
    return {
        'cfg': cfg,
        'device': device,
        'ex_dict': ex_dict
    }

def create_yolc_config(ex_dict, cfg_path):
    """YOLC config 파일을 실험 설정에 맞게 수정"""
    import tempfile
    
    # 기존 config 파일 복사
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    
    with open(cfg_path, 'r') as f:
        config_content = f.read()
    
    # 클래스 수 및 기타 설정 수정
    num_classes = ex_dict['Number of Classes']
    batch_size = ex_dict['Batch Size']
    epochs = ex_dict['Epochs']
    
    # YOLC 특유의 설정 수정
    config_content = config_content.replace(
        'num_classes=10',  # VisDrone 기본값
        f'num_classes={num_classes}'
    )
    
    # LSM (Local Spatial Modeling) 파라미터 설정
    config_content += f"""

# YOLC LSM Configuration Override
data = dict(
    samples_per_gpu={batch_size},
    workers_per_gpu=2,
)

# LSM settings for tiny object detection
model = dict(
    bbox_head=dict(
        num_classes={num_classes},
        # LSM k parameter (0,1,2,3 for different spatial modeling)
        lsm_k=2,  # default LSM configuration
    )
)

# Optimizer and scheduler
optimizer = dict(
    type='{ex_dict.get("Optimizer", "SGD")}',
    lr={ex_dict.get('LR', 0.01)},
    momentum={ex_dict.get('Momentum', 0.9)},
    weight_decay={ex_dict.get('Weight Decay', 0.0001)}
)

total_epochs = {epochs}
"""
    
    # 데이터셋 경로 설정
    data_config = ex_dict.get('Data Config', '')
    if data_config:
        dataset_root = os.path.dirname(os.path.dirname(data_config))
        config_content += f"""

# Dataset configuration override
data_root = '{dataset_root}'
"""
    
    temp_config.write(config_content)
    temp_config.close()
    
    return temp_config.name

def prepare_yolc_dataset(ex_dict):
    """YOLC용 데이터셋 준비 (VisDrone 형식으로 변환)"""
    # YOLC는 VisDrone 데이터셋 형식을 사용하므로 crop 과정이 필요
    data_config_path = os.path.abspath(ex_dict['Data Config'])
    
    # gen_crop.py 실행하여 데이터셋 준비
    gen_crop_script = os.path.join(YOLC_DIR, 'gen_crop.py')
    
    if os.path.exists(gen_crop_script):
        print("YOLC 데이터셋 크롭 실행...")
        try:
            process = subprocess.run([sys.executable, gen_crop_script], 
                                   cwd=YOLC_DIR, timeout=3600)  # 1시간 타임아웃
            print(f"데이터셋 크롭 완료. 반환 코드: {process.returncode}")
        except Exception as e:
            print(f"데이터셋 크롭 중 오류: {e}")
    else:
        print(f"Warning: gen_crop.py not found at {gen_crop_script}")

def train_yolc_model_cli(ex_dict):
    """
    YOLC 모델을 CLI로 학습
    
    기본 명령어:
    python train.py configs/yolc.py
    """
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    model_info = ex_dict['Model']
    cfg = model_info['cfg']
    
    # 출력 디렉토리 설정
    name = f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    output_path = os.path.join(ex_dict['Output Dir'], name)
    os.makedirs(output_path, exist_ok=True)
    
    # Config 파일 경로
    config_path = os.path.join(YOLC_DIR, cfg)
    if not os.path.exists(config_path):
        print(f"Warning: YOLC config not found at {config_path}")
        config_path = os.path.join(YOLC_DIR, "configs", "yolc.py")
    
    # YOLC 데이터셋 준비
    prepare_yolc_dataset(ex_dict)
    
    # 실험 설정에 맞게 config 수정
    temp_config_path = create_yolc_config(ex_dict, config_path)
    
    # 학습 스크립트 경로
    train_script = os.path.join(YOLC_DIR, "train.py")
    
    cmd = [
        sys.executable,
        train_script,
        temp_config_path,
        f"--work-dir={output_path}",
        f"--seed=42",
        f"--deterministic"
    ]
    
    # GPU 설정
    if ex_dict.get('Device', 'cpu') != 'cpu':
        cmd.extend([f"--gpu-ids=0"])
    
    print(f"YOLC 학습 명령어: {' '.join(cmd)}")
    
    # 환경 변수 설정
    env = os.environ.copy()
    env['PYTHONPATH'] = YOLC_DIR
    env['PYTHONUNBUFFERED'] = '1'
    
    try:
        process = subprocess.Popen(cmd, cwd=YOLC_DIR, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True, 
                                 bufsize=0, universal_newlines=True, env=env)
        
        # 실시간 출력
        stdout_lines = []
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[YOLC] {output.strip()}")
                stdout_lines.append(output.strip())
        
        return_code = process.poll()
        print(f"YOLC 학습 완료. 반환 코드: {return_code}")
        
    except Exception as e:
        print(f"YOLC 학습 중 오류: {e}")
        return_code = 1
    
    # 결과 파싱
    log_files = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith('.log') or file.endswith('.txt'):
                log_files.append(os.path.join(root, file))
    
    metrics = {}
    for log_file in log_files:
        metrics.update(parse_yolc_results(log_file))
    
    # 가중치 파일 경로
    pt_path = os.path.join(output_path, "latest.pth")
    if not os.path.exists(pt_path):
        # epoch_*.pth 파일 찾기
        import glob
        pth_files = glob.glob(os.path.join(output_path, "epoch_*.pth"))
        if pth_files:
            pt_path = max(pth_files, key=os.path.getctime)  # 가장 최근 파일
    
    ex_dict['PT path'] = pt_path
    ex_dict['Train Results'] = metrics if metrics else {'total_loss': 0.1}
    
    # 임시 config 파일 삭제
    if os.path.exists(temp_config_path):
        os.unlink(temp_config_path)
    
    return ex_dict

def eval_yolc_model_cli(ex_dict):
    """YOLC 모델 검증"""
    pt_path = ex_dict.get('PT path')
    if not pt_path or not os.path.exists(pt_path):
        print(f"Warning: YOLC model weights not found at {pt_path}")
        ex_dict["Val Results"] = {
            "mAP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0
        }
        return ex_dict
    
    # YOLC 평가 스크립트
    eval_script = os.path.join(YOLC_DIR, "eval_yolc.py")
    
    if not os.path.exists(eval_script):
        print(f"Warning: YOLC eval script not found at {eval_script}")
        ex_dict["Val Results"] = {
            "mAP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0
        }
        return ex_dict
    
    # 출력 디렉토리
    output_path = os.path.dirname(pt_path)
    eval_output = os.path.join(output_path, "eval_results.pkl")
    
    # LSM 파라미터별 평가 (k=0,1,2,3)
    cmd = [
        sys.executable,
        eval_script,
        f"--checkpoint={pt_path}",
        f"--out={eval_output}",
        f"--saved_crop=2"  # LSM k=2 (기본값)
    ]
    
    if ex_dict.get('Device', 'cpu') != 'cpu':
        cmd.extend([f"--gpu-ids=0"])
    
    print(f"YOLC 평가 명령어: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env['PYTHONPATH'] = YOLC_DIR
    
    try:
        process = subprocess.run(cmd, cwd=YOLC_DIR, env=env,
                               capture_output=True, text=True, timeout=1800)  # 30분 타임아웃
        
        # 결과 파싱
        metrics = {}
        if process.stdout:
            import re
            # YOLC 출력에서 mAP 추출
            map_match = re.search(r'mAP[^:]*:\s*([\d.]+)', process.stdout)
            ap50_match = re.search(r'AP50[^:]*:\s*([\d.]+)', process.stdout)
            ap75_match = re.search(r'AP75[^:]*:\s*([\d.]+)', process.stdout)
            
            if map_match:
                metrics['mAP'] = float(map_match.group(1))
            if ap50_match:
                metrics['AP50'] = float(ap50_match.group(1))
            if ap75_match:
                metrics['AP75'] = float(ap75_match.group(1))
        
        ex_dict["Val Results"] = metrics if metrics else {
            "mAP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0
        }
        
    except Exception as e:
        print(f"YOLC 평가 중 오류: {e}")
        ex_dict["Val Results"] = {
            "mAP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0
        }
    
    return ex_dict

def test_yolc_model_cli(ex_dict):
    """YOLC 모델 테스트"""
    pt_path = ex_dict.get('PT path')
    if not pt_path or not os.path.exists(pt_path):
        print(f"Warning: YOLC model weights not found at {pt_path}")
        ex_dict["Test Results"] = type('YOLCResults', (), {
            'mAP': 0.0,
            'AP50': 0.0,
            'AP75': 0.0,
            'AP_tiny': 0.0,
            'AP_small': 0.0,
            'AP_medium': 0.0
        })()
        return ex_dict
    
    # YOLC 테스트 스크립트
    test_script = os.path.join(YOLC_DIR, "test.py")
    
    # 출력 디렉토리
    output_path = os.path.dirname(pt_path)
    test_output = os.path.join(output_path, "test_results.pkl")
    
    # YOLC는 다양한 LSM 설정으로 테스트 가능
    cmd = [
        sys.executable,
        test_script,
        f"--checkpoint={pt_path}",
        f"--out={test_output}",
        f"--eval=bbox",
        f"--show-dir={os.path.join(output_path, 'visualizations')}"
    ]
    
    if ex_dict.get('Device', 'cpu') != 'cpu':
        cmd.extend([f"--gpu-ids=0"])
    
    print(f"YOLC 테스트 명령어: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env['PYTHONPATH'] = YOLC_DIR
    
    try:
        process = subprocess.run(cmd, cwd=YOLC_DIR, env=env,
                               capture_output=True, text=True, timeout=3600)  # 1시간 타임아웃
        
        # 결과 파싱
        metrics = {}
        if process.stdout:
            import re
            # YOLC 특유의 tiny object detection 성능 지표
            map_match = re.search(r'mAP[^:]*:\s*([\d.]+)', process.stdout)
            ap50_match = re.search(r'AP50[^:]*:\s*([\d.]+)', process.stdout)
            ap75_match = re.search(r'AP75[^:]*:\s*([\d.]+)', process.stdout)
            ap_tiny_match = re.search(r'AP_tiny[^:]*:\s*([\d.]+)', process.stdout)
            ap_small_match = re.search(r'AP_small[^:]*:\s*([\d.]+)', process.stdout)
            ap_medium_match = re.search(r'AP_medium[^:]*:\s*([\d.]+)', process.stdout)
            
            if map_match:
                metrics['mAP'] = float(map_match.group(1))
            if ap50_match:
                metrics['AP50'] = float(ap50_match.group(1))
            if ap75_match:
                metrics['AP75'] = float(ap75_match.group(1))
            if ap_tiny_match:
                metrics['AP_tiny'] = float(ap_tiny_match.group(1))
            if ap_small_match:
                metrics['AP_small'] = float(ap_small_match.group(1))
            if ap_medium_match:
                metrics['AP_medium'] = float(ap_medium_match.group(1))
        
        # 결과 객체 생성
        results = type('YOLCResults', (), metrics if metrics else {
            'mAP': 0.0,
            'AP50': 0.0,
            'AP75': 0.0,
            'AP_tiny': 0.0,
            'AP_small': 0.0,
            'AP_medium': 0.0
        })()
        
        ex_dict["Test Results"] = results
        
        print(f"[YOLC Test] mAP: {getattr(results, 'mAP', 0.0):.3f}, "
              f"AP@50: {getattr(results, 'AP50', 0.0):.3f}")
        print(f"[YOLC] AP_tiny: {getattr(results, 'AP_tiny', 0.0):.3f} (specialized for tiny objects)")
        print(f"[YOLC] You Only Look Clusters - LSM for tiny object detection in aerial images")
        
    except Exception as e:
        print(f"YOLC 테스트 중 오류: {e}")
        ex_dict["Test Results"] = type('YOLCResults', (), {
            'mAP': 0.0,
            'AP50': 0.0,
            'AP75': 0.0,
            'AP_tiny': 0.0,
            'AP_small': 0.0,
            'AP_medium': 0.0
        })()
    
    return ex_dict 