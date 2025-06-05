#!/usr/bin/env python3
"""
통합 모델 인터페이스 - 모든 YOLO 계열 모델들의 공통 래퍼
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

class UnifiedDetectionModel(nn.Module, ABC):
    """모든 객체 탐지 모델들의 공통 인터페이스 - nn.Module 상속"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()  # nn.Module 초기화
        self.config = config
        self.device = config.get('device', 'cpu')
        self.num_classes = config.get('num_classes', 80)
        self.input_size = config.get('input_size', 640)
        self.model = None
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        """모델 아키텍처 구성"""
        pass
    
    @abstractmethod
    def load_weights(self, weights_path: str) -> None:
        """사전 훈련된 가중치 로드"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파 (추론)"""
        pass
    
    @abstractmethod
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """손실 함수 계산 - (loss, loss_items) 튜플 반환"""
        pass
    
    @abstractmethod
    def postprocess(self, predictions: torch.Tensor) -> List[Dict]:
        """후처리 (NMS, 좌표 변환 등)"""
        pass
    
    # 🔧 PyTorch nn.Module과의 호환성을 위한 기본 구현들
    def to(self, device):
        """디바이스 이동"""
        self.device = device if isinstance(device, str) else str(device)
        super().to(device)  # nn.Module의 to 호출
        if self.model:
            self.model.to(device)
        return self
    
    def eval(self):
        """평가 모드 설정"""
        super().eval()  # nn.Module의 eval 호출
        if self.model:
            self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        """학습 모드 설정"""
        super().train(mode)  # nn.Module의 train 호출
        if self.model:
            self.model.train(mode)
        return self

class UnifiedTrainer:
    """모든 모델에 대한 공통 학습 파이프라인"""
    
    def __init__(self, model: UnifiedDetectionModel, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.optimizer = None
        self.scheduler = None
        
    def setup_training(self):
        """최적화기, 스케줄러 설정"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),  # 이제 nn.Module이므로 parameters() 사용 가능
            lr=self.config.get('lr', 0.001),
            weight_decay=self.config.get('weight_decay', 0.0001)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.get('epochs', 100)
        )
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(self.model.device)
            targets = targets.to(self.model.device)
            
            # 순전파 - 이제 model(images) 형태로 호출 가능
            predictions = self.model(images)
            
            # 손실 계산
            loss = self.model.compute_loss(predictions, targets)
            
            # 역전파
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return {'avg_loss': total_loss / num_batches}
    
    def validate(self, val_loader) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.model.device)
                predictions = self.model(images)  # 이제 model(images) 형태로 호출 가능
                
                # 후처리
                processed_preds = self.model.postprocess(predictions)
                all_predictions.extend(processed_preds)
                all_targets.extend(targets)
        
        # mAP 계산 (COCO 스타일)
        metrics = self._compute_metrics(all_predictions, all_targets)
        return metrics
    
    def _compute_metrics(self, predictions, targets) -> Dict[str, float]:
        """메트릭 계산 (mAP, precision, recall 등)"""
        # 실제 구현에서는 pycocotools 또는 자체 구현 사용
        return {
            'mAP@0.5': 0.0,  # 실제 계산 로직 필요
            'mAP@0.5:0.95': 0.0,
            'precision': 0.0,
            'recall': 0.0
        } 