"""
YOLC Unified Model Implementation
YOLC 모델을 unified interface를 상속받아 구현

이 모듈은 다음과 같은 역할을 합니다:
1. YOLC의 코어 모듈을 그대로 사용
2. UnifiedModelInterface를 상속받아 표준화된 인터페이스 제공
3. utility 관리 모듈과의 연동
4. YOLC 특유의 LSM (Local Spatial Modeling) 기능 유지
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import yaml
import tempfile
import subprocess

# Utility imports
from utility.debug_logger import debug_log
from utility.checkpoint import save_checkpoint, load_checkpoint
from utility.logger import ExperimentLogger
from utility.predict import predict_batch

# YOLC 디렉토리를 시스템 경로에 추가
YOLC_DIR = os.path.dirname(os.path.abspath(__file__))
YOLC_MODELS_DIR = os.path.join(YOLC_DIR, "models")
if YOLC_MODELS_DIR not in sys.path:
    sys.path.insert(0, YOLC_MODELS_DIR)

# Unified interface import
from utility.unified_model_interface import UnifiedModelInterface

# YOLC 관련 import (코어 모듈)
try:
    from detectors.yolc import YOLC
    from backbones.resnet import ResNet
    from necks.fpn import FPN
    from dense_heads.yolc_head import YOLCHead
    from VisDrone_Dataset import VisDroneDataset
    from mmengine.config import Config  # 최신 MMCV에서는 mmengine.config에서 import
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.apis import train_detector
    YOLC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: YOLC modules not available: {e}")
    YOLC_AVAILABLE = False

class YOLCUnified(UnifiedModelInterface):
    """
    YOLC 모델의 통합 인터페이스 구현
    """
    
    def __init__(self, config: dict, device: str = 'cpu'):
        """
        YOLC 통합 모델 초기화
        
        Args:
            config: 모델 설정 딕셔너리
            device: 학습/추론 디바이스
        """
        # LSM (Local Spatial Modeling) 설정을 먼저 초기화
        self.lsm_k = config.get('lsm_k', 2)
        
        super().__init__(config, device)
        
        # YOLC 특유의 설정
        self.config_path = config.get('config_path', os.path.join(YOLC_DIR, "configs", "yolc.py"))
        self.mmdet_model = None
        self.mmdet_cfg = None
        
        debug_log("YOLC Unified model initialized")
    
    def _initialize_model(self):
        """YOLC 모델 초기화 (원본만 지원)"""
        if not YOLC_AVAILABLE:
            raise ImportError(
                "YOLC 원본 코어 모듈 또는 MMCV/MMDetection이 설치되어 있지 않습니다. "
                "pip install mmcv-full mmdet 및 YOLC 코어 소스가 필요합니다."
            )
        self._initialize_mmdet_model()
    
    def _initialize_mmdet_model(self):
        """MMCV 기반 YOLC 모델 초기화"""
        try:
            # MMCV 설정 로드
            self.mmdet_cfg = Config.fromfile(self.config_path)
            
            # 설정 업데이트
            self._update_config()
            
            # 모델 빌드
            self.mmdet_model = build_detector(
                self.mmdet_cfg.model,
                train_cfg=self.mmdet_cfg.get('train_cfg'),
                test_cfg=self.mmdet_cfg.get('test_cfg')
            )
            
            # 모델을 디바이스로 이동
            self.mmdet_model.to(self.device)
            
            # YOLC 특유의 LSM 설정
            if hasattr(self.mmdet_model, 'bbox_head'):
                self.mmdet_model.bbox_head.lsm_k = self.lsm_k
            
            debug_log("YOLC MMCV model architecture initialized")
        except Exception as e:
            debug_log(f"MMCV model initialization failed: {e}")
    
    def _update_config(self):
        """설정 업데이트 (MMCV 모델용)"""
        if self.mmdet_cfg is None:
            return
            
        # 클래스 수 설정
        num_classes = self.config.get('num_classes', 10)
        self.mmdet_cfg.model.bbox_head.num_classes = num_classes
        
        # 배치 크기 설정
        batch_size = self.config.get('batch_size', 16)
        self.mmdet_cfg.data.samples_per_gpu = batch_size
        
        # 학습률 설정
        lr = self.config.get('lr', 0.01)
        self.mmdet_cfg.optimizer.lr = lr
        
        # 에포크 수 설정
        epochs = self.config.get('epochs', 12)
        self.mmdet_cfg.total_epochs = epochs
        
        # 옵티마이저 설정
        optimizer_type = self.config.get('optimizer', 'SGD')
        self.mmdet_cfg.optimizer.type = optimizer_type
        
        if optimizer_type == 'Adam':
            self.mmdet_cfg.optimizer.weight_decay = self.config.get('weight_decay', 1e-4)
        else:  # SGD
            self.mmdet_cfg.optimizer.momentum = self.config.get('momentum', 0.9)
            self.mmdet_cfg.optimizer.weight_decay = self.config.get('weight_decay', 1e-4)
    
    def forward(self, x):
        """순전파"""
        if self.mmdet_model is not None:
            return self.mmdet_model(x)
        else:
            return self.model(x)
    
    def create_dataloaders(self, data_config: dict):
        """
        YOLC용 데이터로더 생성
        
        Args:
            data_config: 데이터 설정
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        # 기존 dataloader_utils 사용
        from utility.dataloader_utils import build_all_loaders
        
        # ex_dict 형식으로 변환
        ex_dict = {
            'Data Config': data_config.get('data_root', ''),
            'Batch Size': self.config.get('batch_size', 16),
            'Image Size': self.config.get('input_size', 640),
            'Num Workers': self.config.get('num_workers', 2)
        }
        
        train_loader, val_loader, test_loader = build_all_loaders(ex_dict)
        
        return train_loader, val_loader, test_loader
    
    def get_loss_function(self):
        """손실 함수 반환"""
        if self.mmdet_model is not None and hasattr(self.mmdet_model, 'bbox_head'):
            return self.mmdet_model.bbox_head.loss
        else:
            # 간단한 YOLO 스타일 손실 함수
            from utility.loss import YOLOLoss
            return YOLOLoss(
                num_classes=self.config.get('num_classes', 10),
                anchors=self.config.get('anchors', None)
            )
    
    def get_optimizer(self):
        """옵티마이저 반환"""
        optimizer_type = self.config.get('optimizer', 'SGD')
        lr = self.config.get('lr', 0.01)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_type == 'Adam':
            optimizer = optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:  # SGD
            momentum = self.config.get('momentum', 0.9)
            optimizer = optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        
        return optimizer
    
    def get_scheduler(self, optimizer):
        """스케줄러 반환"""
        from torch.optim.lr_scheduler import MultiStepLR
        
        # YOLC 기본 스케줄러: MultiStepLR
        milestones = self.config.get('lr_milestones', [8, 11])
        gamma = self.config.get('lr_gamma', 0.1)
        
        scheduler = MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
        
        return scheduler
    
    def compute_loss(self, outputs, targets):
        """YOLC 손실 계산"""
        if self.mmdet_model is not None and hasattr(self.mmdet_model, 'bbox_head'):
            # YOLC의 내장 손실 함수 사용
            return self.mmdet_model.bbox_head.loss(outputs, targets)
        else:
            # 간단한 손실 계산
            criterion = self.get_loss_function()
            return criterion(outputs, targets)
    
    def process_predictions(self, outputs):
        """YOLC 예측 결과 처리"""
        if self.mmdet_model is not None and hasattr(self.mmdet_model, 'bbox_head'):
            # YOLC의 예측 결과를 표준 형식으로 변환
            return self.mmdet_model.bbox_head.get_bboxes(outputs)
        else:
            # 간단한 예측 처리
            return outputs
    
    def apply_nms(self, predictions, iou_threshold):
        """YOLC NMS 적용"""
        if self.mmdet_model is not None and hasattr(self.mmdet_model, 'bbox_head'):
            # YOLC의 NMS 사용
            return self.mmdet_model.bbox_head.get_bboxes(predictions, iou_threshold)
        else:
            # 간단한 NMS
            return predictions
    
    def get_model_info(self):
        """YOLC 모델 정보 반환"""
        info = super().get_model_info()
        info.update({
            'architecture': 'YOLC with LSM (Local Spatial Modeling)',
            'lsm_k': self.lsm_k,
            'config_path': self.config_path,
            'mmcv_available': YOLC_AVAILABLE
        })
        return info
    
    def save_model(self, path: str, include_optimizer: bool = True):
        """YOLC 모델 저장"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'lsm_k': self.lsm_k
        }
        
        if self.mmdet_model is not None:
            checkpoint['mmdet_model_state_dict'] = self.mmdet_model.state_dict()
            checkpoint['mmdet_cfg'] = self.mmdet_cfg
        
        if include_optimizer and hasattr(self, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, path)
        debug_log(f"YOLC model saved to {path}")
    
    def load_model(self, path: str, load_optimizer: bool = True):
        """YOLC 모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 모델 상태 로드
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'mmdet_model_state_dict' in checkpoint and self.mmdet_model is not None:
            self.mmdet_model.load_state_dict(checkpoint['mmdet_model_state_dict'])
        
        # 설정 및 상태 업데이트
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        if 'mmdet_cfg' in checkpoint:
            self.mmdet_cfg = checkpoint['mmdet_cfg']
        
        self.is_trained = checkpoint.get('is_trained', False)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.lsm_k = checkpoint.get('lsm_k', 2)
        
        # 옵티마이저 상태 로드
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer = self.get_optimizer()
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        debug_log(f"YOLC model loaded from {path}")

# YOLC 모델 생성 함수들
def build_yolc_unified_model(ex_dict=None):
    """YOLC 통합 모델 빌드"""
    if ex_dict is None:
        ex_dict = {}
    
    # 기본 설정
    config = {
        'num_classes': ex_dict.get('Number of Classes', 10),
        'batch_size': ex_dict.get('Batch Size', 16),
        'lr': ex_dict.get('LR', 0.01),
        'optimizer': ex_dict.get('Optimizer', 'SGD'),
        'momentum': ex_dict.get('Momentum', 0.9),
        'weight_decay': ex_dict.get('Weight Decay', 1e-4),
        'epochs': ex_dict.get('Epochs', 12),
        'lsm_k': 2,  # YOLC LSM 설정
        'config_path': os.path.join(YOLC_DIR, "configs", "yolc.py"),
        'input_size': ex_dict.get('Image Size', 640)
    }
    
    device = ex_dict.get('Device', 'cpu')
    
    return YOLCUnified(config, device)

def train_yolc_unified(ex_dict):
    """YOLC 통합 모델 학습"""
    model = build_yolc_unified_model(ex_dict)
    
    # 데이터로더 생성
    data_config = {
        'data_root': ex_dict.get('Data Config', ''),
        'num_classes': ex_dict.get('Number of Classes', 10)
    }
    
    train_loader, val_loader, test_loader = model.create_dataloaders(data_config)
    
    # 학습 실행
    results = model.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=ex_dict.get('Epochs', 12)
    )
    
    return ex_dict

def eval_yolc_unified(ex_dict):
    """YOLC 통합 모델 평가"""
    model = build_yolc_unified_model(ex_dict)
    
    # 데이터로더 생성
    data_config = {
        'data_root': ex_dict.get('Data Config', ''),
        'num_classes': ex_dict.get('Number of Classes', 10)
    }
    
    _, val_loader, _ = model.create_dataloaders(data_config)
    
    # 평가 실행
    results = model.evaluate(
        val_loader=val_loader,
        class_names=ex_dict.get('Class Names', [])
    )
    
    return ex_dict

def test_yolc_unified(ex_dict):
    """YOLC 통합 모델 테스트"""
    model = build_yolc_unified_model(ex_dict)
    
    # 데이터로더 생성
    data_config = {
        'data_root': ex_dict.get('Data Config', ''),
        'num_classes': ex_dict.get('Number of Classes', 10)
    }
    
    _, _, test_loader = model.create_dataloaders(data_config)
    
    # 추론 실행
    results = model.predict(
        test_loader=test_loader,
        class_names=ex_dict.get('Class Names', [])
    )
    
    return ex_dict 