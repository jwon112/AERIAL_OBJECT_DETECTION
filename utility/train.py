"""
Training Loop Module
범용적인 학습 루프를 제공하는 모듈

이 모듈은 다음과 같은 기능을 제공합니다:
1. 표준화된 학습 루프
2. 손실 계산 및 역전파
3. 옵티마이저 및 스케줄러 관리
4. 체크포인트 저장
5. 진행 상황 모니터링
"""

import time
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from utility.debug_logger import debug_log
from utility.metrics import calculate_batch_metrics
from utility.checkpoint import save_checkpoint, load_checkpoint

class TrainingLoop:
    """
    범용적인 학습 루프 클래스
    """
    
    def __init__(self, model, train_loader, val_loader=None, 
                 criterion=None, optimizer=None, scheduler=None,
                 device='cpu', config=None):
        """
        학습 루프 초기화
        
        Args:
            model: 학습할 모델
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더 (선택사항)
            criterion: 손실 함수
            optimizer: 옵티마이저
            scheduler: 학습률 스케줄러
            device: 학습 디바이스
            config: 학습 설정
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        
        # 모델을 디바이스로 이동
        self.model.to(device)
        
        # 학습 상태 초기화
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # 체크포인트 경로
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        debug_log("Training loop initialized")
    
    def train_epoch(self):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        batch_metrics = []
        
        # 진행률 표시
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # 배치 데이터 준비
            if isinstance(batch, (list, tuple)):
                images, targets = batch
            elif isinstance(batch, dict):
                images = batch['images']
                targets = batch['targets']
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")
            
            # 디바이스로 이동
            images = images.to(self.device)
            if isinstance(targets, list):
                targets = [t.to(self.device) for t in targets]
            else:
                targets = targets.to(self.device)
            
            # 순전파
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # 손실 계산
            if self.criterion is not None:
                loss = self.criterion(outputs, targets)
            else:
                # 모델에 내장된 손실 함수 사용
                if hasattr(self.model, 'compute_loss'):
                    loss = self.model.compute_loss(outputs, targets)
                else:
                    raise ValueError("No criterion provided and model has no compute_loss method")
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑 (설정된 경우)
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            # 옵티마이저 스텝
            self.optimizer.step()
            
            # 메트릭 계산
            batch_metric = calculate_batch_metrics(outputs, targets, self.config.get('metric_type', 'mAP'))
            batch_metrics.append(batch_metric)
            
            total_loss += loss.item()
            
            # 진행률 업데이트
            avg_loss = total_loss / (batch_idx + 1)
            avg_metric = np.mean(batch_metrics)
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Metric': f'{avg_metric:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = total_loss / len(self.train_loader)
        epoch_metric = np.mean(batch_metrics)
        
        return epoch_loss, epoch_metric
    
    def validate_epoch(self):
        """한 에포크 검증"""
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        batch_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # 배치 데이터 준비
                if isinstance(batch, (list, tuple)):
                    images, targets = batch
                elif isinstance(batch, dict):
                    images = batch['images']
                    targets = batch['targets']
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")
                
                # 디바이스로 이동
                images = images.to(self.device)
                if isinstance(targets, list):
                    targets = [t.to(self.device) for t in targets]
                else:
                    targets = targets.to(self.device)
                
                # 순전파
                outputs = self.model(images)
                
                # 손실 계산
                if self.criterion is not None:
                    loss = self.criterion(outputs, targets)
                else:
                    if hasattr(self.model, 'compute_loss'):
                        loss = self.model.compute_loss(outputs, targets)
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                
                # 메트릭 계산
                batch_metric = calculate_batch_metrics(outputs, targets, self.config.get('metric_type', 'mAP'))
                batch_metrics.append(batch_metric)
                
                total_loss += loss.item()
        
        epoch_loss = total_loss / len(self.val_loader)
        epoch_metric = np.mean(batch_metrics)
        
        return epoch_loss, epoch_metric
    
    def train(self, num_epochs, save_freq=1, early_stopping_patience=None):
        """
        전체 학습 과정 실행
        
        Args:
            num_epochs: 학습할 에포크 수
            save_freq: 체크포인트 저장 빈도
            early_stopping_patience: 조기 종료 인내심 (None이면 비활성화)
        """
        debug_log(f"Starting training for {num_epochs} epochs")
        
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 학습
            train_loss, train_metric = self.train_epoch()
            
            # 검증
            val_loss, val_metric = self.validate_epoch()
            
            # 스케줄러 업데이트
            if self.scheduler is not None:
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step()
                elif hasattr(self.scheduler, 'step_epoch'):
                    self.scheduler.step_epoch(epoch)
            
            # 결과 저장
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metric)
            self.val_metrics.append(val_metric)
            
            # 로그 출력
            debug_log(f"Epoch {epoch + 1}/{num_epochs}: "
                     f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}, "
                     f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}")
            
            # 체크포인트 저장
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")
            
            # 최고 성능 모델 저장
            if val_metric > self.best_metric:
                self.best_metric = val_metric
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint("best_model.pth")
                debug_log(f"New best model saved with metric: {val_metric:.4f}")
            else:
                patience_counter += 1
            
            # 조기 종료 체크
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                debug_log(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # 최종 결과 저장
        self.save_training_results()
        
        debug_log(f"Training completed. Best metric: {self.best_metric:.4f} at epoch {best_epoch + 1}")
        
        return {
            'best_metric': self.best_metric,
            'best_epoch': best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
    
    def save_checkpoint(self, filename):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        debug_log(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, filename):
        """체크포인트 로드"""
        checkpoint_path = self.checkpoint_dir / filename
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_metrics = checkpoint.get('train_metrics', [])
        self.val_metrics = checkpoint.get('val_metrics', [])
        
        debug_log(f"Checkpoint loaded: {checkpoint_path}")
        debug_log(f"Resuming from epoch {self.current_epoch + 1}")
    
    def save_training_results(self):
        """학습 결과 저장"""
        results = {
            'config': self.config,
            'best_metric': self.best_metric,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'training_time': datetime.now().isoformat()
        }
        
        results_path = self.checkpoint_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        debug_log(f"Training results saved: {results_path}")

def create_training_loop(model, train_loader, val_loader=None, 
                        criterion=None, optimizer=None, scheduler=None,
                        device='cpu', config=None):
    """학습 루프 생성 헬퍼 함수"""
    return TrainingLoop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    ) 