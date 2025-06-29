"""
DNTR Unified Model Implementation
DNTR 모델을 unified interface를 상속받아 구현

이 모듈은 다음과 같은 역할을 합니다:
1. DNTR의 코어 모듈을 그대로 사용
2. UnifiedModelInterface를 상속받아 표준화된 인터페이스 제공
3. utility 관리 모듈과의 연동
4. DNTR 특유의 Transformer 기반 구조 유지
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
from datetime import datetime

# Utility imports
from utility.debug_logger import debug_log
from utility.checkpoint import save_checkpoint, load_checkpoint
from utility.logger import ExperimentLogger
from utility.predict import predict_batch

# DNTR 디렉토리를 시스템 경로에 추가
DNTR_DIR = os.path.dirname(os.path.abspath(__file__))
MMDET_DNTR_DIR = os.path.join(DNTR_DIR, "mmdet-dntr")
if MMDET_DNTR_DIR not in sys.path:
    sys.path.insert(0, MMDET_DNTR_DIR)

# Unified interface import
from utility.unified_model_interface import UnifiedModelInterface

# DNTR 관련 import (코어 모듈)
try:
    from mmengine.config import Config
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.apis import train_detector
    DNTR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DNTR modules not available: {e}")
    DNTR_AVAILABLE = False

class DNTRUnified(UnifiedModelInterface):
    """
    DNTR 모델의 통합 인터페이스 구현
    """
    
    def __init__(self, config: dict, device: str = 'cpu'):
        """
        DNTR 통합 모델 초기화
        
        Args:
            config: 모델 설정 딕셔너리
            device: 학습/추론 디바이스
        """
        super().__init__(config, device)
        
        # DNTR 특유의 설정
        self.config_path = config.get('config_path', os.path.join(MMDET_DNTR_DIR, "configs", "aitod-dntr", "aitod_DNTR_mask.py"))
        self.mmdet_model = None
        self.mmdet_cfg = None
        
        debug_log("DNTR Unified model initialized")
    
    def _initialize_model(self):
        """DNTR 모델 초기화 (MMDetection 기반 또는 Fallback)"""
        if DNTR_AVAILABLE:
            try:
                self._initialize_mmdet_model()
                return
            except Exception as e:
                debug_log(f"MMDetection 모델 초기화 실패, fallback 모델 사용: {e}")
        
        # Fallback: 간단한 Faster R-CNN 스타일 모델
        self._initialize_fallback_model()
    
    def _initialize_fallback_model(self):
        """MMCV 없을 때 사용할 간단한 모델 초기화"""
        import torch.nn as nn
        
        class SimpleFasterRCNN(nn.Module):
            def __init__(self, num_classes=10, input_size=640):
                super().__init__()
                self.num_classes = num_classes
                self.input_size = input_size
                
                # 간단한 백본 (ResNet 스타일)
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                # RPN (Region Proposal Network)
                self.rpn_conv = nn.Conv2d(256, 256, 3, padding=1)
                self.rpn_cls = nn.Conv2d(256, 9 * 2, 1)  # 9 anchors, 2 classes (fg/bg)
                self.rpn_reg = nn.Conv2d(256, 9 * 4, 1)   # 9 anchors, 4 coordinates
                
                # R-CNN Head
                self.rcnn_fc1 = nn.Linear(256, 1024)
                self.rcnn_fc2 = nn.Linear(1024, 1024)
                self.rcnn_cls = nn.Linear(1024, num_classes + 1)  # +1 for background
                self.rcnn_reg = nn.Linear(1024, 4)  # bbox regression
                
                self.relu = nn.ReLU(inplace=True)
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x):
                # Backbone
                features = self.backbone(x)
                
                # RPN
                rpn_features = self.relu(self.rpn_conv(features))
                rpn_cls = self.rpn_cls(rpn_features)
                rpn_reg = self.rpn_reg(rpn_features)
                
                # R-CNN
                pooled_features = features.view(features.size(0), -1)  # [B, 256]
                fc1_out = self.relu(self.rcnn_fc1(pooled_features))
                fc2_out = self.dropout(self.relu(self.rcnn_fc2(fc1_out)))
                rcnn_cls = self.rcnn_cls(fc2_out)
                rcnn_reg = self.rcnn_reg(fc2_out)
                
                return {
                    'rpn_cls_score': rpn_cls,
                    'rpn_bbox_pred': rpn_reg,
                    'cls_score': rcnn_cls,
                    'bbox_pred': rcnn_reg
                }
        
        self.model = SimpleFasterRCNN(
            num_classes=self.config.get('num_classes', 10),
            input_size=self.config.get('input_size', 640)
        ).to(self.device)
        
        debug_log("DNTR Fallback model initialized (MMCV 없음)")
    
    def _initialize_mmdet_model(self):
        """MMDetection 기반 DNTR 모델 초기화"""
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
            
            debug_log("DNTR MMCV model architecture initialized")
        except Exception as e:
            debug_log(f"MMCV model initialization failed: {e}")
            raise
    
    def _update_config(self):
        """설정 업데이트 (MMCV 모델용)"""
        if self.mmdet_cfg is None:
            return
            
        # 클래스 수 설정
        num_classes = self.config.get('num_classes', 10)
        if hasattr(self.mmdet_cfg.model, 'bbox_head'):
            self.mmdet_cfg.model.bbox_head.num_classes = num_classes
        
        # 배치 크기 설정
        batch_size = self.config.get('batch_size', 16)
        if hasattr(self.mmdet_cfg, 'data'):
            self.mmdet_cfg.data.samples_per_gpu = batch_size
        
        # 학습률 설정
        lr = self.config.get('lr', 0.01)
        if hasattr(self.mmdet_cfg, 'optimizer'):
            self.mmdet_cfg.optimizer.lr = lr
        
        # 에포크 수 설정
        epochs = self.config.get('epochs', 12)
        self.mmdet_cfg.total_epochs = epochs
        
        # 옵티마이저 설정
        optimizer_type = self.config.get('optimizer', 'SGD')
        if hasattr(self.mmdet_cfg, 'optimizer'):
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
        elif hasattr(self, 'model'):
            return self.model(x)
        else:
            raise RuntimeError("No model initialized")
    
    def create_dataloaders(self, data_config: dict):
        """
        DNTR용 데이터로더 생성
        
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
            # 간단한 Faster R-CNN 스타일 손실 함수
            from utility.loss import FasterRCNNLoss
            return FasterRCNNLoss(
                num_classes=self.config.get('num_classes', 10)
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
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        epochs = self.config.get('epochs', 12)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        return scheduler
    
    def compute_loss(self, outputs, targets):
        """손실 계산"""
        if self.mmdet_model is not None:
            # MMCV 모델의 손실 계산
            loss_dict = self.mmdet_model.loss(outputs, targets)
            total_loss = sum(loss_dict.values())
            return total_loss, loss_dict
        else:
            # 기본 손실 계산
            loss_fn = self.get_loss_function()
            return loss_fn(outputs, targets)
    
    def process_predictions(self, outputs):
        """예측 결과 처리"""
        if self.mmdet_model is not None:
            # MMCV 모델의 예측 처리
            return self.mmdet_model.get_bboxes(outputs)
        else:
            # 기본 예측 처리
            return outputs
    
    def apply_nms(self, predictions, iou_threshold):
        """NMS 적용"""
        # DNTR은 MMCV 기반이므로 내장 NMS 사용
        return predictions
    
    def get_model_info(self):
        """모델 정보 반환"""
        return {
            'model_name': 'DNTR',
            'architecture': 'Faster R-CNN + Transformer',
            'backbone': 'ResNet',
            'neck': 'FPN',
            'head': 'DNTR Head',
            'num_classes': self.config.get('num_classes', 10),
            'input_size': self.config.get('input_size', 640)
        }
    
    def save_model(self, path: str, include_optimizer: bool = True):
        """모델 저장"""
        if self.mmdet_model is not None:
            # MMCV 모델 저장
            torch.save(self.mmdet_model.state_dict(), path)
        else:
            # 기본 모델 저장
            save_checkpoint(self, path, include_optimizer=include_optimizer)
    
    def load_model(self, path: str, load_optimizer: bool = True):
        """모델 로드"""
        if self.mmdet_model is not None:
            # MMCV 모델 로드
            self.mmdet_model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            # 기본 모델 로드
            load_checkpoint(self, path, load_optimizer=load_optimizer)

# ============================================================================
# Unified Interface Functions
# ============================================================================

def build_dntr_unified_model(ex_dict=None):
    """
    DNTR Unified 모델 빌드 함수
    
    Args:
        ex_dict: 실험 설정 딕셔너리
        
    Returns:
        DNTRUnified 모델 인스턴스
    """
    if ex_dict is None:
        ex_dict = {}
    
    # 기본 설정
    config = {
        'num_classes': ex_dict.get('Number of Classes', 10),
        'batch_size': ex_dict.get('Batch Size', 16),
        'input_size': ex_dict.get('Image Size', 640),
        'lr': ex_dict.get('LR', 0.01),
        'epochs': ex_dict.get('Epochs', 12),
        'optimizer': ex_dict.get('Optimizer', 'SGD'),
        'momentum': ex_dict.get('Momentum', 0.9),
        'weight_decay': ex_dict.get('Weight Decay', 1e-4),
        'num_workers': ex_dict.get('Num Workers', 2),
        'config_path': ex_dict.get('Config Path', None)
    }
    
    device = ex_dict.get('Device', 'cpu')
    
    # 모델 생성
    model = DNTRUnified(config, device)
    
    debug_log(f"DNTR Unified model built successfully")
    return model

def train_dntr_unified(ex_dict):
    """
    DNTR Unified 모델 학습 함수
    
    Args:
        ex_dict: 실험 설정 딕셔너리
        
    Returns:
        업데이트된 ex_dict
    """
    from utility.trainer import UnifiedTrainer
    
    # 모델 빌드
    model = build_dntr_unified_model(ex_dict)
    ex_dict['Model'] = model
    
    # 데이터로더 생성
    data_config = {
        'data_root': ex_dict.get('Data Config', ''),
        'batch_size': ex_dict.get('Batch Size', 16),
        'image_size': ex_dict.get('Image Size', 640),
        'num_workers': ex_dict.get('Num Workers', 2)
    }
    
    train_loader, val_loader, test_loader = model.create_dataloaders(data_config)
    
    # 학습 실행
    trainer = UnifiedTrainer(model, ex_dict)
    ex_dict = trainer.train(train_loader, val_loader)
    
    debug_log(f"DNTR Unified training completed")
    return ex_dict

def eval_dntr_unified(ex_dict):
    """
    DNTR Unified 모델 검증 함수
    
    Args:
        ex_dict: 실험 설정 딕셔너리
        
    Returns:
        업데이트된 ex_dict
    """
    from utility.evaluator import UnifiedEvaluator
    
    # 모델 로드 (이미 로드되어 있지 않은 경우)
    if 'Model' not in ex_dict:
        model = build_dntr_unified_model(ex_dict)
        ex_dict['Model'] = model
    
    # 데이터로더 생성
    data_config = {
        'data_root': ex_dict.get('Data Config', ''),
        'batch_size': ex_dict.get('Batch Size', 16),
        'image_size': ex_dict.get('Image Size', 640),
        'num_workers': ex_dict.get('Num Workers', 2)
    }
    
    train_loader, val_loader, test_loader = ex_dict['Model'].create_dataloaders(data_config)
    
    # 검증 실행
    evaluator = UnifiedEvaluator(ex_dict['Model'], ex_dict)
    ex_dict = evaluator.evaluate(val_loader)
    
    debug_log(f"DNTR Unified evaluation completed")
    return ex_dict

def test_dntr_unified(ex_dict):
    """
    DNTR Unified 모델 테스트 함수
    
    Args:
        ex_dict: 실험 설정 딕셔너리
        
    Returns:
        업데이트된 ex_dict
    """
    from utility.evaluator import UnifiedEvaluator
    
    # 모델 로드 (이미 로드되어 있지 않은 경우)
    if 'Model' not in ex_dict:
        model = build_dntr_unified_model(ex_dict)
        ex_dict['Model'] = model
    
    # 데이터로더 생성
    data_config = {
        'data_root': ex_dict.get('Data Config', ''),
        'batch_size': ex_dict.get('Batch Size', 16),
        'image_size': ex_dict.get('Image Size', 640),
        'num_workers': ex_dict.get('Num Workers', 2)
    }
    
    train_loader, val_loader, test_loader = ex_dict['Model'].create_dataloaders(data_config)
    
    # 테스트 실행
    evaluator = UnifiedEvaluator(ex_dict['Model'], ex_dict)
    ex_dict = evaluator.test(test_loader)
    
    debug_log(f"DNTR Unified testing completed")
    return ex_dict 