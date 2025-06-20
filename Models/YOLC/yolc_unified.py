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
from utility.eval import calculate_metrics
from utility.predict import predict_batch

# YOLC 디렉토리를 시스템 경로에 추가
YOLC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, YOLC_DIR)

# Unified interface import
from utility.unified_model_interface import UnifiedModelInterface

# YOLC 관련 import
try:
    from models.detectors.yolc import YOLC
    from models.backbones.resnet import ResNet
    from models.necks.fpn import FPN
    from models.dense_heads.yolc_head import YOLCHead
    from VisDrone_Dataset import VisDroneDataset
    from mmcv import Config
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
        if not YOLC_AVAILABLE:
            raise ImportError("YOLC modules are not available. Please check the installation.")
        
        super().__init__(config, device)
        
        # YOLC 특유의 설정
        self.config_path = config.get('config_path', os.path.join(YOLC_DIR, "configs", "yolc.py"))
        self.mmdet_model = None
        self.mmdet_cfg = None
        
        debug_log("YOLC Unified model initialized")
    
    def _initialize_model(self):
        """YOLC 모델 초기화"""
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
            self.mmdet_model.bbox_head.lsm_k = self.config.get('lsm_k', 2)
        
        debug_log("YOLC model architecture initialized")
    
    def _update_config(self):
        """설정 업데이트"""
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
        return self.mmdet_model(x)
    
    def create_dataloaders(self, data_config: dict):
        """
        YOLC용 데이터로더 생성
        
        Args:
            data_config: 데이터 설정
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        # 데이터셋 준비 (VisDrone 형식으로 변환)
        self._prepare_dataset(data_config)
        
        # MMCV 데이터셋 빌드
        train_dataset = build_dataset(self.mmdet_cfg.data.train)
        val_dataset = build_dataset(self.mmdet_cfg.data.val)
        test_dataset = build_dataset(self.mmdet_cfg.data.test)
        
        # DataLoader 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=True,
            num_workers=self.config.get('num_workers', 2),
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            collate_fn=self._collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('batch_size', 16),
            shuffle=False,
            num_workers=self.config.get('num_workers', 2),
            collate_fn=self._collate_fn
        )
        
        return train_loader, val_loader, test_loader
    
    def _prepare_dataset(self, data_config: dict):
        """데이터셋 준비 (VisDrone 형식으로 변환)"""
        # gen_crop.py 실행하여 데이터셋 준비
        gen_crop_script = os.path.join(YOLC_DIR, 'gen_crop.py')
        
        if os.path.exists(gen_crop_script):
            debug_log("YOLC 데이터셋 크롭 실행...")
            try:
                process = subprocess.run([sys.executable, gen_crop_script], 
                                       cwd=YOLC_DIR, timeout=3600)
                debug_log(f"데이터셋 크롭 완료. 반환 코드: {process.returncode}")
            except Exception as e:
                debug_log(f"데이터셋 크롭 중 오류: {e}")
        else:
            debug_log(f"Warning: gen_crop.py not found at {gen_crop_script}")
    
    def _collate_fn(self, batch):
        """YOLC 데이터를 utility 형식으로 변환하는 collate 함수"""
        images = []
        targets = []
        
        for item in batch:
            if isinstance(item, dict):
                # MMCV 형식의 데이터
                img = item['img'].data
                gt_bboxes = item['gt_bboxes'].data
                gt_labels = item['gt_labels'].data
                
                # utility 형식으로 변환: (class_id, x_center, y_center, width, height)
                if len(gt_bboxes) > 0:
                    bboxes = gt_bboxes[0]  # 첫 번째 이미지의 박스들
                    labels = gt_labels[0]  # 첫 번째 이미지의 라벨들
                    
                    # (x1, y1, x2, y2) -> (x_center, y_center, width, height)
                    x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2
                    y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2
                    width = bboxes[:, 2] - bboxes[:, 0]
                    height = bboxes[:, 3] - bboxes[:, 1]
                    
                    # (class_id, x_center, y_center, width, height) 형식으로 변환
                    target = torch.stack([
                        labels.float(),
                        x_center,
                        y_center,
                        width,
                        height
                    ], dim=1)
                else:
                    target = torch.empty((0, 5))
                
                images.append(img)
                targets.append(target)
            else:
                # 기존 형식
                images.append(item[0])
                targets.append(item[1])
        
        return torch.stack(images), targets
    
    def get_loss_function(self):
        """YOLC 손실 함수 반환"""
        # YOLC 모델의 내장 손실 함수 사용
        return self.mmdet_model.bbox_head.loss
    
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
        # YOLC의 내장 손실 함수 사용
        return self.mmdet_model.bbox_head.loss(outputs, targets)
    
    def process_predictions(self, outputs):
        """YOLC 예측 결과 처리"""
        # YOLC의 예측 결과를 표준 형식으로 변환
        if hasattr(self.mmdet_model, 'bbox_head'):
            return self.mmdet_model.bbox_head.get_bboxes(outputs)
        else:
            return outputs
    
    def apply_nms(self, predictions, iou_threshold):
        """YOLC NMS 적용"""
        # YOLC의 NMS 사용
        if hasattr(self.mmdet_model, 'bbox_head'):
            return self.mmdet_model.bbox_head.get_bboxes(predictions, iou_threshold)
        else:
            return predictions
    
    def get_model_info(self):
        """YOLC 모델 정보 반환"""
        info = super().get_model_info()
        info.update({
            'architecture': 'YOLC with LSM (Local Spatial Modeling)',
            'lsm_k': self.config.get('lsm_k', 2),
            'config_path': self.config_path
        })
        return info
    
    def save_model(self, path: str, include_optimizer: bool = True):
        """YOLC 모델 저장"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'mmdet_model_state_dict': self.mmdet_model.state_dict(),
            'config': self.config,
            'mmdet_cfg': self.mmdet_cfg,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric
        }
        
        if include_optimizer and hasattr(self, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, path)
        debug_log(f"YOLC model saved to {path}")
    
    def load_model(self, path: str, load_optimizer: bool = True):
        """YOLC 모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # 모델 상태 로드
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'mmdet_model_state_dict' in checkpoint:
            self.mmdet_model.load_state_dict(checkpoint['mmdet_model_state_dict'])
        
        # 설정 및 상태 업데이트
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        if 'mmdet_cfg' in checkpoint:
            self.mmdet_cfg = checkpoint['mmdet_cfg']
        
        self.is_trained = checkpoint.get('is_trained', False)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
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
        'config_path': os.path.join(YOLC_DIR, "configs", "yolc.py")
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