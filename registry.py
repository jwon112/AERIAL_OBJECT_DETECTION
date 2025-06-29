"""
Model Registry
모델 등록 및 관리를 위한 통합 레지스트리

이 모듈은 다음과 같은 기능을 제공합니다:
1. 모델 등록 및 관리
2. 인터페이스 타입별 분류 (Native, CLI, Unified)
3. 모델 빌드 및 파이프라인 제공
4. 실험 로깅 시스템
"""

import os
import sys
import logging
from pathlib import Path
from functools import partial
from typing import Dict, Any, Optional, Tuple, Callable

# ============================================================================
# 모델 Import 섹션
# ============================================================================

# Native 인터페이스 모델들 (기존 방식)
try:
    from Models.YoloOW.yoloow_utils import build_yoloow_model, train_yoloow_model, eval_yoloow_model, test_yoloow_model
    from Models.YOLOH.yoloh_utils import build_yoloh_model, train_yoloh_model, eval_yoloh_model, test_yoloh_model
    from Models.ultralytics.yolov8_utils import build_yolov8_model, train_yolov8_model, eval_yolov8_model, test_yolov8_model
    from Models.YOLOH.config.yoloh_config import yoloh_config
    NATIVE_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Native 모델들 로드 실패: {e}")
    NATIVE_MODELS_AVAILABLE = False

# CLI 인터페이스 모델들
try:
    from Models.YoloOW.yoloow_cli import build_yoloow_model_cli, train_yoloow_model_cli, eval_yoloow_model_cli, test_yoloow_model_cli
    from Models.YOLOH.yoloh_cli import build_yoloh_model_cli, train_yoloh_model_cli, eval_yoloh_model_cli, test_yoloh_model_cli
    from Models.DNTR.dntr_cli import build_dntr_model_cli, train_dntr_model_cli, eval_dntr_model_cli, test_dntr_model_cli
    from Models.MSNet.msnet_cli import build_msnet_model_cli, train_msnet_model_cli, eval_msnet_model_cli, test_msnet_model_cli
    from Models.YOLC.yolc_cli import build_yolc_model_cli, train_yolc_model_cli, eval_yolc_model_cli, test_yolc_model_cli
    
    # FFCA-YOLO CLI
    sys.path.append(os.path.join(os.path.dirname(__file__), 'Models', 'FFCA-YOLO'))
    from ffca_yolo_cli import build_ffca_yolo_model_cli, train_ffca_yolo_model_cli, eval_ffca_yolo_model_cli, test_ffca_yolo_model_cli
    
    CLI_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ CLI 모델들 로드 실패: {e}")
    CLI_MODELS_AVAILABLE = False

# Unified 인터페이스 모델들
try:
    from Models.YOLC.yolc_unified import build_yolc_unified_model, train_yolc_unified, eval_yolc_unified, test_yolc_unified
    YOLC_UNIFIED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ YOLC unified 로드 실패: {e}")
    YOLC_UNIFIED_AVAILABLE = False

try:
    from Models.DNTR.dntr_unified import build_dntr_unified_model, train_dntr_unified, eval_dntr_unified, test_dntr_unified
    DNTR_UNIFIED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ DNTR unified 로드 실패: {e}")
    DNTR_UNIFIED_AVAILABLE = False

# ============================================================================
# 모델 등록 함수들
# ============================================================================

def register_native_models() -> Dict[str, Dict[str, Any]]:
    """Native 인터페이스 모델들 등록"""
    models = {}
    
    if not NATIVE_MODELS_AVAILABLE:
        return models
    
    # YoloOW Native
    models['YoloOW'] = {
        'build': partial(build_yoloow_model, cfg='yoloOW.yaml'),
        'train': train_yoloow_model,
        'eval': eval_yoloow_model,
        'test': test_yoloow_model,
        'type': 'native',
        'description': 'YoloOW Native Interface'
    }
    
    # YOLOH Native
    models['yoloh18'] = {
        'build': partial(build_yoloh_model, cfg=yoloh_config['yoloh18']),
        'train': train_yoloh_model,
        'eval': eval_yoloh_model,
        'test': test_yoloh_model,
        'type': 'native',
        'description': 'YOLOH-18 Native Interface'
    }
    
    models['yoloh50'] = {
        'build': partial(build_yoloh_model, cfg=yoloh_config['yoloh50']),
        'train': train_yoloh_model,
        'eval': eval_yoloh_model,
        'test': test_yoloh_model,
        'type': 'native',
        'description': 'YOLOH-50 Native Interface'
    }
    
    models['yoloh101'] = {
        'build': partial(build_yoloh_model, cfg=yoloh_config['yoloh101']),
        'train': train_yoloh_model,
        'eval': eval_yoloh_model,
        'test': test_yoloh_model,
        'type': 'native',
        'description': 'YOLOH-101 Native Interface'
    }
    
    # YOLOv8 Native
    models['yolov8n'] = {
        'build': partial(build_yolov8_model, cfg='Models/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml'),
        'train': train_yolov8_model,
        'eval': eval_yolov8_model,
        'test': test_yolov8_model,
        'type': 'native',
        'description': 'YOLOv8n Native Interface'
    }
    
    return models

def register_cli_models() -> Dict[str, Dict[str, Any]]:
    """CLI 인터페이스 모델들 등록"""
    models = {}
    
    if not CLI_MODELS_AVAILABLE:
        return models
    
    # YoloOW CLI
    models['YoloOW_CLI'] = {
        'build': partial(build_yoloow_model_cli, cfg='yoloOW.yaml'),
        'train': train_yoloow_model_cli,
        'eval': eval_yoloow_model_cli,
        'test': test_yoloow_model_cli,
        'type': 'cli',
        'description': 'YoloOW CLI Interface'
    }
    
    # YOLOH CLI
    models['YOLOH_CLI'] = {
        'build': partial(build_yoloh_model_cli, cfg=None),
        'train': train_yoloh_model_cli,
        'eval': eval_yoloh_model_cli,
        'test': test_yoloh_model_cli,
        'type': 'cli',
        'description': 'YOLOH CLI Interface'
    }
    
    # DNTR CLI
    models['DNTR_CLI'] = {
        'build': partial(build_dntr_model_cli, cfg='configs/aitod-dntr/aitod_DNTR_mask.py'),
        'train': train_dntr_model_cli,
        'eval': eval_dntr_model_cli,
        'test': test_dntr_model_cli,
        'type': 'cli',
        'description': 'DNTR CLI Interface'
    }
    
    # FFCA-YOLO CLI
    models['FFCA_YOLO_CLI'] = {
        'build': partial(build_ffca_yolo_model_cli, cfg='FFCA-YOLO.yaml'),
        'train': train_ffca_yolo_model_cli,
        'eval': eval_ffca_yolo_model_cli,
        'test': test_ffca_yolo_model_cli,
        'type': 'cli',
        'description': 'FFCA-YOLO CLI Interface'
    }
    
    # MSNet CLI
    models['MSNet_CLI'] = {
        'build': partial(build_msnet_model_cli, cfg='yolov8_l.yaml'),
        'train': train_msnet_model_cli,
        'eval': eval_msnet_model_cli,
        'test': test_msnet_model_cli,
        'type': 'cli',
        'description': 'MSNet CLI Interface'
    }
    
    # YOLC CLI
    models['YOLC_CLI'] = {
        'build': partial(build_yolc_model_cli, cfg='configs/yolc.py'),
        'train': train_yolc_model_cli,
        'eval': eval_yolc_model_cli,
        'test': test_yolc_model_cli,
        'type': 'cli',
        'description': 'YOLC CLI Interface'
    }
    
    return models

def register_unified_models() -> Dict[str, Dict[str, Any]]:
    """Unified 인터페이스 모델들 등록"""
    models = {}
    
    # YOLC Unified
    if YOLC_UNIFIED_AVAILABLE:
        models['YOLC_UNIFIED'] = {
            'build': build_yolc_unified_model,
            'train': train_yolc_unified,
            'eval': eval_yolc_unified,
            'test': test_yolc_unified,
            'type': 'unified',
            'description': 'YOLC Unified Interface (Utility 기반)'
        }
    
    # DNTR Unified
    if DNTR_UNIFIED_AVAILABLE:
        models['DNTR_UNIFIED'] = {
            'build': build_dntr_unified_model,
            'train': train_dntr_unified,
            'eval': eval_dntr_unified,
            'test': test_dntr_unified,
            'type': 'unified',
            'description': 'DNTR Unified Interface (Utility 기반)'
        }
    
    # 기존 unified 모델들 (동적 로드)
    unified_models = _load_existing_unified_models()
    models.update(unified_models)
    
    return models

def _load_existing_unified_models() -> Dict[str, Dict[str, Any]]:
    """기존 unified 모델들을 동적으로 로드"""
    models = {}
    
    # YOLOv8 Unified
    try:
        yolov8_path = os.path.join(os.path.dirname(__file__), 'Models', 'ultralytics', 'yolov8_unified.py')
        if os.path.exists(yolov8_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("yolov8_unified", yolov8_path)
            yolov8_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(yolov8_module)
            
            models['yolov8_unified'] = {
                'build': yolov8_module.build_yolov8_unified_model,
                'train': getattr(yolov8_module, 'train_yolov8_unified', None),
                'eval': getattr(yolov8_module, 'eval_yolov8_unified', None),
                'test': getattr(yolov8_module, 'test_yolov8_unified', None),
                'type': 'unified',
                'description': 'YOLOv8 Unified Interface'
            }
            print("✅ YOLOv8 Unified 로드 성공")
    except Exception as e:
        print(f"⚠️ YOLOv8 Unified 로드 실패: {e}")
    
    # YoloOW Unified
    try:
        yoloow_path = os.path.join(os.path.dirname(__file__), 'Models', 'YoloOW', 'yoloow_unified.py')
        if os.path.exists(yoloow_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("yoloow_unified", yoloow_path)
            yoloow_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(yoloow_module)
            
            models['yoloow_unified'] = {
                'build': yoloow_module.build_yoloow_unified_model,
                'train': getattr(yoloow_module, 'train_yoloow_unified', None),
                'eval': getattr(yoloow_module, 'eval_yoloow_unified', None),
                'test': getattr(yoloow_module, 'test_yoloow_unified', None),
                'type': 'unified',
                'description': 'YoloOW Unified Interface'
            }
            print("✅ YoloOW Unified 로드 성공")
    except Exception as e:
        print(f"⚠️ YoloOW Unified 로드 실패: {e}")
    
    # FFCA-YOLO Unified
    try:
        ffca_path = os.path.join(os.path.dirname(__file__), 'Models', 'FFCA-YOLO', 'ffca_yolo_unified.py')
        if os.path.exists(ffca_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("ffca_yolo_unified", ffca_path)
            ffca_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ffca_module)
            
            models['ffca_yolo_unified'] = {
                'build': ffca_module.build_ffca_yolo_unified_model,
                'train': getattr(ffca_module, 'train_ffca_yolo_unified', None),
                'eval': getattr(ffca_module, 'eval_ffca_yolo_unified', None),
                'test': getattr(ffca_module, 'test_ffca_yolo_unified', None),
                'type': 'unified',
                'description': 'FFCA-YOLO Unified Interface'
            }
            print("✅ FFCA-YOLO Unified 로드 성공")
    except Exception as e:
        print(f"⚠️ FFCA-YOLO Unified 로드 실패: {e}")
    
    return models

# ============================================================================
# 메인 레지스트리 구성
# ============================================================================

def build_model_registry() -> Dict[str, Dict[str, Any]]:
    """전체 모델 레지스트리 구성"""
    registry = {}
    
    # 각 인터페이스 타입별로 모델 등록
    registry.update(register_native_models())
    registry.update(register_cli_models())
    registry.update(register_unified_models())
    
    return registry

# 전역 모델 레지스트리
MODEL_REGISTRY = build_model_registry()

# ============================================================================
# 레지스트리 접근 함수들
# ============================================================================

def get_model(model_name: str, ex_dict: Optional[Dict[str, Any]] = None):
    """모델 인스턴스 반환"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"지원하지 않는 모델: {model_name}")
    
    build_func = MODEL_REGISTRY[model_name]['build']
    if ex_dict is not None:
        return build_func(ex_dict=ex_dict)
    else:
        return build_func()

def get_pipeline(model_name: str) -> Tuple[Callable, Callable, Callable]:
    """모델 파이프라인 반환 (train, eval, test 함수)"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"지원하지 않는 모델: {model_name}")
    
    model_info = MODEL_REGISTRY[model_name]
    train_fn = model_info['train']
    eval_fn = model_info['eval']
    test_fn = model_info['test']
    
    # None인 함수들 체크
    if train_fn is None:
        print(f"⚠️ {model_name}: train 함수가 구현되지 않았습니다.")
    if eval_fn is None:
        print(f"⚠️ {model_name}: eval 함수가 구현되지 않았습니다.")
    if test_fn is None:
        print(f"⚠️ {model_name}: test 함수가 구현되지 않았습니다.")
    
    return train_fn, eval_fn, test_fn

def get_available_models() -> Dict[str, str]:
    """사용 가능한 모델 목록 반환 (인터페이스 타입별 분류)"""
    available = {}
    
    for model_name, model_info in MODEL_REGISTRY.items():
        model_type = model_info['type']
        description = model_info['description']
        
        if model_type == 'native':
            available[model_name] = f"⚙️ {description}"
        elif model_type == 'cli':
            available[model_name] = f"🔄 {description}"
        elif model_type == 'unified':
            available[model_name] = f"🔧 {description}"
        else:
            available[model_name] = f"❓ {description}"
    
    return available

def get_models_by_type(model_type: str) -> Dict[str, Dict[str, Any]]:
    """특정 타입의 모델들만 반환"""
    return {name: info for name, info in MODEL_REGISTRY.items() 
            if info['type'] == model_type}

def get_model_info(model_name: str) -> Dict[str, Any]:
    """특정 모델의 정보 반환"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"지원하지 않는 모델: {model_name}")
    
    return MODEL_REGISTRY[model_name].copy()

# ============================================================================
# 실험 로깅 시스템
# ============================================================================

class ExperimentLogger:
    """실험별 디버그 로깅을 위한 클래스"""
    
    def __init__(self, experiment_id: str):
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
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
        
        print(f"📋 DEBUG 로그 파일: {self.log_file}")
    
    def get_log_file_path(self) -> str:
        """로그 파일 경로 반환"""
        return str(self.log_file)
    
    def debug(self, message: str):
        """DEBUG 레벨 로그"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """INFO 레벨 로그"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """WARNING 레벨 로그"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """ERROR 레벨 로그"""
        self.logger.error(message)
    
    def log_model_start(self, model_name: str, dataset_name: str, iteration: int):
        """모델 시작 로그"""
        self.info(f"🚀 모델 시작: {model_name} | 데이터셋: {dataset_name} | 반복: {iteration}")
    
    def log_model_complete(self, model_name: str, train_time: float, results: Dict[str, Any]):
        """모델 완료 로그"""
        self.info(f"✅ 모델 완료: {model_name} | 학습시간: {train_time:.2f}s")
        if results:
            self.info(f"📊 결과: {results}")
    
    def log_error(self, model_name: str, error_msg: str):
        """에러 로그"""
        self.error(f"❌ 모델 에러: {model_name} | 에러: {error_msg}")

def initialize_experiment_logging(experiment_id: str) -> ExperimentLogger:
    """실험 로깅 초기화"""
    return ExperimentLogger(experiment_id)

# ============================================================================
# 초기화 및 상태 출력
# ============================================================================

def print_registry_status():
    """레지스트리 상태 출력"""
    print("\n" + "="*60)
    print("📋 MODEL REGISTRY STATUS")
    print("="*60)
    
    # 인터페이스 타입별 통계
    native_count = len(get_models_by_type('native'))
    cli_count = len(get_models_by_type('cli'))
    unified_count = len(get_models_by_type('unified'))
    
    print(f"⚙️  Native Interface: {native_count}개")
    print(f"🔄 CLI Interface: {cli_count}개")
    print(f"🔧 Unified Interface: {unified_count}개")
    print(f"📝 총 {len(MODEL_REGISTRY)}개 모델이 등록되었습니다.")
    
    # 사용 가능한 모델 목록
    print("\n📋 사용 가능한 모델들:")
    available_models = get_available_models()
    for model_name, description in available_models.items():
        print(f"  {model_name}: {description}")
    
    print("="*60)

# 레지스트리 상태 출력
print_registry_status()
