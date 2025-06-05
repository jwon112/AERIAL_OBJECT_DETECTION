from Models.YoloOW.yoloow_utils import build_yoloow_model, train_yoloow_model, eval_yoloow_model, test_yoloow_model
from Models.YoloOW.yoloow_cli import build_yoloow_model_cli, train_yoloow_model_cli, eval_yoloow_model_cli, test_yoloow_model_cli
from Models.YOLOH.yoloh_utils import build_yoloh_model, train_yoloh_model, eval_yoloh_model, test_yoloh_model
from functools import partial
from Models.YOLOH.config.yoloh_config import yoloh_config
from Models.ultralytics.yolov8_utils import (
    build_yolov8_model, train_yolov8_model, eval_yolov8_model, test_yolov8_model
)
from Models.YOLOH.yoloh_cli import build_yoloh_model_cli, train_yoloh_model_cli, eval_yoloh_model_cli, test_yoloh_model_cli

# 새로운 모델들의 CLI 인터페이스 import
from Models.DNTR.dntr_cli import build_dntr_model_cli, train_dntr_model_cli, eval_dntr_model_cli, test_dntr_model_cli
import sys
import os
import torch
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'Models', 'FFCA-YOLO'))
from ffca_yolo_cli import build_ffca_yolo_model_cli, train_ffca_yolo_model_cli, eval_ffca_yolo_model_cli, test_ffca_yolo_model_cli
from Models.MSNet.msnet_cli import build_msnet_model_cli, train_msnet_model_cli, eval_msnet_model_cli, test_msnet_model_cli
from Models.YOLC.yolc_cli import build_yolc_model_cli, train_yolc_model_cli, eval_yolc_model_cli, test_yolc_model_cli

# 통합 인터페이스 모델들 import (각자 자체 train/eval/test 함수 사용)
def try_import_unified_models():
    """통합 모델들을 안전하게 import"""
    unified_models = {}
    
    # YOLOv8 통합 모델
    try:
        yolov8_path = os.path.join(os.path.dirname(__file__), 'Models', 'ultralytics', 'yolov8_unified.py')
        if os.path.exists(yolov8_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("yolov8_unified", yolov8_path)
            yolov8_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(yolov8_module)
            
            unified_models['yolov8_unified'] = {
                'build': yolov8_module.build_yolov8_unified_model,
                'train': getattr(yolov8_module, 'train_yolov8_unified', None),
                'eval': getattr(yolov8_module, 'eval_yolov8_unified', None),
                'test': getattr(yolov8_module, 'test_yolov8_unified', None)
            }
            print("✅ YOLOv8 통합 모델 로드 성공")
    except Exception as e:
        print(f"⚠️ YOLOv8 통합 모델 로드 실패: {e}")
    
    # YoloOW 통합 모델
    try:
        yoloow_path = os.path.join(os.path.dirname(__file__), 'Models', 'YoloOW', 'yoloow_unified.py')
        if os.path.exists(yoloow_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("yoloow_unified", yoloow_path)
            yoloow_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(yoloow_module)
            
            unified_models['yoloow_unified'] = {
                'build': yoloow_module.build_yoloow_unified_model,
                'train': getattr(yoloow_module, 'train_yoloow_unified', None),
                'eval': getattr(yoloow_module, 'eval_yoloow_unified', None),
                'test': getattr(yoloow_module, 'test_yoloow_unified', None)
            }
            print("✅ YoloOW 통합 모델 로드 성공")
    except Exception as e:
        print(f"⚠️ YoloOW 통합 모델 로드 실패: {e}")
    
    # FFCA-YOLO 통합 모델
    try:
        ffca_path = os.path.join(os.path.dirname(__file__), 'Models', 'FFCA-YOLO', 'ffca_yolo_unified.py')
        if os.path.exists(ffca_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("ffca_yolo_unified", ffca_path)
            ffca_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ffca_module)
            
            unified_models['ffca_yolo_unified'] = {
                'build': ffca_module.build_ffca_yolo_unified_model,
                'train': getattr(ffca_module, 'train_ffca_yolo_unified', None),
                'eval': getattr(ffca_module, 'eval_ffca_yolo_unified', None),
                'test': getattr(ffca_module, 'test_ffca_yolo_unified', None)
            }
            print("✅ FFCA-YOLO 통합 모델 로드 성공")
    except Exception as e:
        print(f"⚠️ FFCA-YOLO 통합 모델 로드 실패: {e}")
    
    return unified_models

# 통합 모델들 로드 시도
try:
    unified_model_registry = try_import_unified_models()
    print(f"✅ {len(unified_model_registry)}개 통합 인터페이스 모델이 로드되었습니다.")
except Exception as e:
    unified_model_registry = {}
    print(f"⚠️ 통합 인터페이스 모델 로드 실패: {e}")

# 기본 모델 레지스트리
model_registry = {
    'YoloOW': {
        'build': partial(build_yoloow_model, cfg='yoloOW.yaml'), 
        'train': train_yoloow_model,
        'eval': eval_yoloow_model,
        'test': test_yoloow_model
    },
    'YoloOW_CLI': {
        'build': partial(build_yoloow_model_cli, cfg='yoloOW.yaml'), 
        'train': train_yoloow_model_cli,
        'eval': eval_yoloow_model_cli,
        'test': test_yoloow_model_cli
    },
    'yoloh18': {
        'build': partial(build_yoloh_model, cfg=yoloh_config['yoloh18']),
        'train': train_yoloh_model,
        'eval': eval_yoloh_model,
        'test' : test_yoloh_model
    },
    'yoloh50' : {
        'build': partial(build_yoloh_model, cfg=yoloh_config['yoloh50']),
        'train': train_yoloh_model,
        'eval': eval_yoloh_model,
        'test' : test_yoloh_model
    },
    'yoloh101' : {
        'build': partial(build_yoloh_model, cfg=yoloh_config['yoloh101']),
        'train': train_yoloh_model,
        'eval': eval_yoloh_model,
        'test' : test_yoloh_model
    },
    'yolov8n': {
        'build': partial(build_yolov8_model, cfg='Models/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml'),
        'train': train_yolov8_model,
        'eval': eval_yolov8_model,
        'test': test_yolov8_model
    },
    'YOLOH_CLI': {
        'build': partial(build_yoloh_model_cli, cfg=None),
        'train': train_yoloh_model_cli,
        'eval': eval_yoloh_model_cli,
        'test': test_yoloh_model_cli
    },
    # 새로운 모델들의 CLI 인터페이스
    'DNTR_CLI': {
        'build': partial(build_dntr_model_cli, cfg='configs/aitod-dntr/aitod_DNTR_mask.py'),
        'train': train_dntr_model_cli,
        'eval': eval_dntr_model_cli,
        'test': test_dntr_model_cli
    },
    'FFCA_YOLO_CLI': {
        'build': partial(build_ffca_yolo_model_cli, cfg='FFCA-YOLO.yaml'),
        'train': train_ffca_yolo_model_cli,
        'eval': eval_ffca_yolo_model_cli,
        'test': test_ffca_yolo_model_cli
    },
    'MSNet_CLI': {
        'build': partial(build_msnet_model_cli, cfg='yolov8_l.yaml'),
        'train': train_msnet_model_cli,
        'eval': eval_msnet_model_cli,
        'test': test_msnet_model_cli
    },
    'YOLC_CLI': {
        'build': partial(build_yolc_model_cli, cfg='configs/yolc.py'),
        'train': train_yolc_model_cli,
        'eval': eval_yolc_model_cli,
        'test': test_yolc_model_cli
    },
}

# 통합 모델들을 기본 레지스트리에 추가
model_registry.update(unified_model_registry)
print(f"📝 총 {len(model_registry)}개 모델이 registry에 등록되었습니다.")

def get_model(model_name, ex_dict):
    if model_name not in model_registry:
        raise ValueError(f"Unsupported model: {model_name}")
    return model_registry[model_name]['build'](ex_dict=ex_dict)

def get_pipeline(model_name):
    if model_name not in model_registry:
        raise ValueError(f"Unsupported model: {model_name}")
    
    train_fn = model_registry[model_name]['train']
    eval_fn = model_registry[model_name]['eval'] 
    test_fn = model_registry[model_name]['test']
    
    # None인 함수들 체크 (통합 모델에서 아직 구현 안된 경우)
    if train_fn is None:
        print(f"⚠️ {model_name}: train 함수가 구현되지 않았습니다.")
    if eval_fn is None:
        print(f"⚠️ {model_name}: eval 함수가 구현되지 않았습니다.")
    if test_fn is None:
        print(f"⚠️ {model_name}: test 함수가 구현되지 않았습니다.")
    
    return train_fn, eval_fn, test_fn

def get_available_models():
    """사용 가능한 모델 목록 반환 (상태 포함)"""
    available = {}
    for model_name in model_registry.keys():
        if 'unified' in model_name:
            available[model_name] = "🔧 통합 인터페이스"
        elif 'CLI' in model_name:
            available[model_name] = "🔄 CLI 인터페이스"
        else:
            available[model_name] = "⚙️ 네이티브 인터페이스"
    return available

# 실험 로깅 시스템 추가
import logging
from pathlib import Path

class ExperimentLogger:
    """실험별 디버그 로깅을 위한 클래스"""
    
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        self.log_file = self.log_dir / f"debug_{experiment_id}.log"
        
        # 로거 설정
        self.logger = logging.getLogger(f"experiment_{experiment_id}")
        self.logger.setLevel(logging.DEBUG)
        
        # 기존 핸들러 제거 (중복 방지)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 파일 핸들러 추가
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 포맷터 설정
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # 시작 로그
        self.logger.info(f"=== 실험 시작: {experiment_id} ===")
        print(f"📋 DEBUG 로그 파일: {self.log_file}")
    
    def get_log_file_path(self):
        """로그 파일 경로 반환"""
        return str(self.log_file)
    
    def debug(self, message):
        """디버그 메시지 로깅"""
        self.logger.debug(message)
    
    def info(self, message):
        """정보 메시지 로깅"""
        self.logger.info(message)
    
    def warning(self, message):
        """경고 메시지 로깅"""
        self.logger.warning(message)
    
    def error(self, message):
        """에러 메시지 로깅"""
        self.logger.error(message)
    
    def log_model_start(self, model_name, dataset_name, iteration):
        """모델 실험 시작 로깅"""
        self.info(f"모델 실험 시작 - {model_name} | 데이터셋: {dataset_name} | 반복: {iteration}")
    
    def log_model_complete(self, model_name, train_time, results):
        """모델 실험 완료 로깅"""
        self.info(f"모델 실험 완료 - {model_name} | 학습 시간: {train_time:.2f}초")
        if results:
            self.info(f"결과: {results}")
    
    def log_error(self, model_name, error_msg):
        """모델 실험 에러 로깅"""
        self.error(f"모델 에러 - {model_name}: {error_msg}")

def initialize_experiment_logging(experiment_id):
    """
    실험별 디버그 로깅 초기화
    
    Args:
        experiment_id (str): 실험 ID (예: '250602_114530')
    
    Returns:
        ExperimentLogger: 실험 로거 객체
    """
    try:
        logger = ExperimentLogger(experiment_id)
        return logger
    except Exception as e:
        print(f"⚠️ 실험 로깅 초기화 실패: {e}")
        return None
