"""
Unified Model Interface
모든 모델이 상속받을 공통 인터페이스

이 모듈은 다음과 같은 기능을 제공합니다:
1. 표준화된 모델 인터페이스 정의
2. 공통 메서드 구현
3. utility 관리 모듈과의 연동
4. 모델별 특수성 처리 가이드라인
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import json
import yaml

from utility.debug_logger import debug_log
from utility.train import create_training_loop
from utility.eval import create_evaluator
from utility.predict import create_predictor
from utility.logger import create_experiment_logger

class UnifiedModelInterface(ABC, nn.Module):
    """
    모든 모델이 상속받을 통합 인터페이스
    
    이 클래스는 다음과 같은 표준 인터페이스를 제공합니다:
    1. 모델 초기화 및 설정
    2. 데이터로더 생성
    3. 손실 함수 정의
    4. 옵티마이저 및 스케줄러 설정
    5. 학습, 평가, 추론 메서드
    6. 모델 저장 및 로드
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        """
        통합 모델 인터페이스 초기화
        
        Args:
            config: 모델 설정 딕셔너리
            device: 학습/추론 디바이스
        """
        super().__init__()
        
        self.config = config
        self.device = device
        self.model_name = self.__class__.__name__
        
        # 모델 상태
        self.is_trained = False
        self.current_epoch = 0
        self.best_metric = 0.0
        
        # utility 모듈들
        self.logger = None
        self.trainer = None
        self.evaluator = None
        self.predictor = None
        
        # 모델별 초기화
        self._initialize_model()
        
        debug_log(f"{self.model_name} unified interface initialized")
    
    @abstractmethod
    def _initialize_model(self):
        """
        모델별 초기화 메서드 (하위 클래스에서 구현)
        
        이 메서드에서는 다음을 수행해야 합니다:
        1. 모델 아키텍처 정의
        2. 가중치 초기화
        3. 모델을 디바이스로 이동
        """
        pass
    
    @abstractmethod
    def create_dataloaders(self, data_config: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        """
        데이터로더 생성 (하위 클래스에서 구현)
        
        Args:
            data_config: 데이터 설정
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        pass
    
    @abstractmethod
    def get_loss_function(self) -> nn.Module:
        """
        손실 함수 반환 (하위 클래스에서 구현)
        
        Returns:
            손실 함수
        """
        pass
    
    @abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        """
        옵티마이저 반환 (하위 클래스에서 구현)
        
        Returns:
            옵티마이저
        """
        pass
    
    @abstractmethod
    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        """
        스케줄러 반환 (하위 클래스에서 구현)
        
        Args:
            optimizer: 옵티마이저
            
        Returns:
            스케줄러
        """
        pass
    
    def setup_logging(self, experiment_name: str, **kwargs):
        """로깅 설정"""
        self.logger = create_experiment_logger(
            experiment_name=experiment_name,
            config=self.config,
            **kwargs
        )
        
        # 모델 정보 로깅
        model_info = self.get_model_info()
        self.logger.log_model_info(model_info)
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,
            'device': self.device,
            'is_trained': self.is_trained,
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric
        }
    
    def train(self, train_loader: Any, val_loader: Optional[Any] = None,
              num_epochs: int = 100, save_freq: int = 1, 
              early_stopping_patience: Optional[int] = None) -> Dict[str, Any]:
        """
        모델 학습
        
        Args:
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            num_epochs: 학습 에포크 수
            save_freq: 체크포인트 저장 빈도
            early_stopping_patience: 조기 종료 인내심
            
        Returns:
            학습 결과
        """
        debug_log(f"Starting training for {self.model_name}")
        
        # 로깅 설정 (아직 설정되지 않은 경우)
        if self.logger is None:
            self.setup_logging(f"{self.model_name}_training")
        
        # 학습 구성 요소 준비
        criterion = self.get_loss_function()
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        
        # 학습 루프 생성
        self.trainer = create_training_loop(
            model=self,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            config=self.config
        )
        
        # 학습 실행
        results = self.trainer.train(
            num_epochs=num_epochs,
            save_freq=save_freq,
            early_stopping_patience=early_stopping_patience
        )
        
        # 학습 완료 상태 업데이트
        self.is_trained = True
        self.current_epoch = results['best_epoch']
        self.best_metric = results['best_metric']
        
        # 최종 결과 로깅
        if self.logger:
            self.logger.log_experiment_end(results)
        
        debug_log(f"Training completed for {self.model_name}")
        return results
    
    def evaluate(self, val_loader: Any, iou_thresholds: Optional[List[float]] = None,
                class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        모델 평가
        
        Args:
            val_loader: 검증 데이터로더
            iou_thresholds: IoU 임계값 리스트
            class_names: 클래스 이름 리스트
            
        Returns:
            평가 결과
        """
        debug_log(f"Starting evaluation for {self.model_name}")
        
        # 평가기 생성
        self.evaluator = create_evaluator(
            model=self,
            val_loader=val_loader,
            device=self.device,
            config=self.config
        )
        
        # 평가 실행
        results = self.evaluator.evaluate(
            iou_thresholds=iou_thresholds,
            class_names=class_names
        )
        
        # 평가 결과 로깅
        if self.logger:
            self.logger.log_validation_metrics(results['metrics'], self.current_epoch)
        
        debug_log(f"Evaluation completed for {self.model_name}")
        return results
    
    def predict(self, test_loader: Any, save_results: bool = True,
               class_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        모델 추론
        
        Args:
            test_loader: 테스트 데이터로더
            save_results: 결과 저장 여부
            class_names: 클래스 이름 리스트
            
        Returns:
            추론 결과 리스트
        """
        debug_log(f"Starting prediction for {self.model_name}")
        
        # 추론기 생성
        self.predictor = create_predictor(
            model=self,
            device=self.device,
            config=self.config
        )
        
        # 추론 실행
        results = self.predictor.predict_dataset(
            test_loader=test_loader,
            save_results=save_results,
            class_names=class_names
        )
        
        debug_log(f"Prediction completed for {self.model_name}")
        return results
    
    def predict_single_image(self, image_path: str, save_result: bool = True,
                           class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        단일 이미지 추론
        
        Args:
            image_path: 이미지 경로
            save_result: 결과 저장 여부
            class_names: 클래스 이름 리스트
            
        Returns:
            추론 결과
        """
        if self.predictor is None:
            self.predictor = create_predictor(
                model=self,
                device=self.device,
                config=self.config
            )
        
        return self.predictor.predict_image(
            image_path=image_path,
            save_result=save_result,
            class_names=class_names
        )
    
    def save_model(self, path: str, include_optimizer: bool = True):
        """
        모델 저장
        
        Args:
            path: 저장 경로
            include_optimizer: 옵티마이저 상태 포함 여부
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric
        }
        
        if include_optimizer and hasattr(self, 'optimizer'):
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, path)
        debug_log(f"Model saved to {path}")
    
    def load_model(self, path: str, load_optimizer: bool = True):
        """
        모델 로드
        
        Args:
            path: 모델 파일 경로
            load_optimizer: 옵티마이저 상태 로드 여부
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # 모델 상태 로드
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # 설정 및 상태 업데이트
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
        
        self.is_trained = checkpoint.get('is_trained', False)
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.best_metric = checkpoint.get('best_metric', 0.0)
        
        # 옵티마이저 상태 로드
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer = self.get_optimizer()
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        debug_log(f"Model loaded from {path}")
    
    def get_config(self) -> Dict[str, Any]:
        """현재 설정 반환"""
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]):
        """설정 업데이트"""
        self.config.update(new_config)
        debug_log(f"Config updated for {self.model_name}")
    
    def to_device(self, device: str):
        """모델을 지정된 디바이스로 이동"""
        self.device = device
        super().to(device)
        debug_log(f"Model moved to {device}")
    
    def freeze_layers(self, layer_names: List[str]):
        """
        특정 레이어 동결
        
        Args:
            layer_names: 동결할 레이어 이름 리스트
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                debug_log(f"Frozen layer: {name}")
    
    def unfreeze_layers(self, layer_names: List[str]):
        """
        특정 레이어 동결 해제
        
        Args:
            layer_names: 동결 해제할 레이어 이름 리스트
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                debug_log(f"Unfrozen layer: {name}")
    
    def get_trainable_parameters(self) -> List[str]:
        """학습 가능한 파라미터 이름 리스트 반환"""
        return [name for name, param in self.named_parameters() if param.requires_grad]
    
    def get_frozen_parameters(self) -> List[str]:
        """동결된 파라미터 이름 리스트 반환"""
        return [name for name, param in self.named_parameters() if not param.requires_grad]
    
    def count_parameters(self) -> Dict[str, int]:
        """파라미터 개수 계산"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }
    
    def summary(self):
        """모델 요약 정보 출력"""
        print(f"\n{'='*50}")
        print(f"MODEL SUMMARY: {self.model_name}")
        print(f"{'='*50}")
        
        # 기본 정보
        model_info = self.get_model_info()
        for key, value in model_info.items():
            print(f"{key}: {value}")
        
        # 파라미터 정보
        param_counts = self.count_parameters()
        print(f"\nParameters:")
        for key, value in param_counts.items():
            print(f"  {key}: {value:,}")
        
        # 레이어 정보
        print(f"\nLayers:")
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # 리프 노드만
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    print(f"  {name}: {param_count:,} parameters")
        
        print(f"{'='*50}\n")

# 모델 팩토리 함수
def create_unified_model(model_class: type, config: Dict[str, Any], 
                        device: str = 'cpu') -> UnifiedModelInterface:
    """
    통합 모델 생성 팩토리 함수
    
    Args:
        model_class: 모델 클래스 (UnifiedModelInterface를 상속받아야 함)
        config: 모델 설정
        device: 디바이스
        
    Returns:
        초기화된 모델 인스턴스
    """
    if not issubclass(model_class, UnifiedModelInterface):
        raise ValueError(f"Model class must inherit from UnifiedModelInterface")
    
    return model_class(config, device) 