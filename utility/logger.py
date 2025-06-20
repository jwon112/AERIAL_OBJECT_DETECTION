"""
Logger Module
로깅, TensorBoard, WandB 등을 관리하는 모듈

이 모듈은 다음과 같은 기능을 제공합니다:
1. 텍스트 로그 관리
2. TensorBoard 로깅
3. WandB 로깅
4. 메트릭 추적
5. 실험 관리
"""

import os
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml
from typing import Dict, Any, Optional, Union
import warnings

# TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard not available. Install with: pip install tensorboard")

# WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("WandB not available. Install with: pip install wandb")

from utility.debug_logger import debug_log

class ExperimentLogger:
    """
    실험 로깅을 위한 통합 로거 클래스
    """
    
    def __init__(self, experiment_name: str, log_dir: str = "./logs", 
                 config: Optional[Dict[str, Any]] = None,
                 enable_tensorboard: bool = True,
                 enable_wandb: bool = False,
                 wandb_project: Optional[str] = None,
                 wandb_entity: Optional[str] = None):
        """
        실험 로거 초기화
        
        Args:
            experiment_name: 실험 이름
            log_dir: 로그 저장 디렉토리
            config: 실험 설정
            enable_tensorboard: TensorBoard 활성화 여부
            enable_wandb: WandB 활성화 여부
            wandb_project: WandB 프로젝트 이름
            wandb_entity: WandB 엔티티 이름
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.config = config or {}
        
        # 타임스탬프 생성
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{self.experiment_name}_{self.timestamp}"
        
        # 로그 디렉토리 생성
        self.experiment_log_dir = self.log_dir / self.experiment_id
        self.experiment_log_dir.mkdir(parents=True, exist_ok=True)
        
        # 텍스트 로거 설정
        self._setup_text_logger()
        
        # TensorBoard 설정
        self.tensorboard_writer = None
        if enable_tensorboard and TENSORBOARD_AVAILABLE:
            self._setup_tensorboard()
        
        # WandB 설정
        self.wandb_run = None
        if enable_wandb and WANDB_AVAILABLE:
            self._setup_wandb(wandb_project, wandb_entity)
        
        # 메트릭 추적
        self.metrics_history = {}
        
        # 실험 시작 로깅
        self.log_experiment_start()
        
        debug_log(f"Experiment logger initialized: {self.experiment_id}")
    
    def _setup_text_logger(self):
        """텍스트 로거 설정"""
        # 로거 생성
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.INFO)
        
        # 기존 핸들러 제거
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 파일 핸들러
        log_file = self.experiment_log_dir / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _setup_tensorboard(self):
        """TensorBoard 설정"""
        if not TENSORBOARD_AVAILABLE:
            return
        
        tensorboard_dir = self.experiment_log_dir / "tensorboard"
        self.tensorboard_writer = SummaryWriter(str(tensorboard_dir))
        
        # 설정 로깅
        if self.config:
            self.tensorboard_writer.add_text("Config", json.dumps(self.config, indent=2), 0)
    
    def _setup_wandb(self, project: Optional[str], entity: Optional[str]):
        """WandB 설정"""
        if not WANDB_AVAILABLE:
            return
        
        try:
            # WandB 초기화
            wandb.init(
                project=project or "aerial-object-detection",
                entity=entity,
                name=self.experiment_id,
                config=self.config,
                dir=str(self.experiment_log_dir)
            )
            self.wandb_run = wandb.run
            
            self.logger.info(f"WandB initialized: {self.wandb_run.url}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize WandB: {e}")
            self.wandb_run = None
    
    def log_experiment_start(self):
        """실험 시작 로깅"""
        self.logger.info("="*60)
        self.logger.info(f"EXPERIMENT STARTED: {self.experiment_id}")
        self.logger.info("="*60)
        
        # 설정 로깅
        if self.config:
            self.logger.info("Configuration:")
            for key, value in self.config.items():
                self.logger.info(f"  {key}: {value}")
        
        # 시스템 정보 로깅
        self._log_system_info()
    
    def _log_system_info(self):
        """시스템 정보 로깅"""
        import torch
        import platform
        
        system_info = {
            "Platform": platform.platform(),
            "Python Version": platform.python_version(),
            "PyTorch Version": torch.__version__,
            "CUDA Available": torch.cuda.is_available(),
            "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "N/A",
            "GPU Count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        if torch.cuda.is_available():
            system_info["GPU Name"] = torch.cuda.get_device_name(0)
        
        self.logger.info("System Information:")
        for key, value in system_info.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: int, 
                   prefix: str = ""):
        """
        메트릭 로깅
        
        Args:
            metrics: 메트릭 딕셔너리
            step: 현재 스텝
            prefix: 메트릭 이름 접두사
        """
        # 텍스트 로깅
        metric_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                               for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {prefix} {metric_str}")
        
        # 메트릭 히스토리 저장
        for key, value in metrics.items():
            full_key = f"{prefix}_{key}" if prefix else key
            if full_key not in self.metrics_history:
                self.metrics_history[full_key] = []
            self.metrics_history[full_key].append((step, value))
        
        # TensorBoard 로깅
        if self.tensorboard_writer:
            for key, value in metrics.items():
                full_key = f"{prefix}/{key}" if prefix else key
                self.tensorboard_writer.add_scalar(full_key, value, step)
        
        # WandB 로깅
        if self.wandb_run:
            wandb_metrics = {f"{prefix}_{key}" if prefix else key: value 
                           for key, value in metrics.items()}
            self.wandb_run.log(wandb_metrics, step=step)
    
    def log_loss(self, loss: float, step: int, loss_type: str = "total"):
        """손실 로깅"""
        self.log_metrics({f"{loss_type}_loss": loss}, step, "train")
    
    def log_validation_metrics(self, metrics: Dict[str, float], step: int):
        """검증 메트릭 로깅"""
        self.log_metrics(metrics, step, "val")
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  val_metrics: Optional[Dict[str, float]] = None):
        """에포크 로깅"""
        epoch_metrics = {"epoch": epoch, "train_loss": train_loss}
        
        if val_loss is not None:
            epoch_metrics["val_loss"] = val_loss
        
        if val_metrics:
            epoch_metrics.update(val_metrics)
        
        self.log_metrics(epoch_metrics, epoch, "epoch")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """모델 정보 로깅"""
        self.logger.info("Model Information:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")
        
        # TensorBoard에 모델 정보 추가
        if self.tensorboard_writer:
            self.tensorboard_writer.add_text("Model Info", 
                                           json.dumps(model_info, indent=2), 0)
    
    def log_image(self, image, step: int, tag: str = "sample"):
        """이미지 로깅"""
        # TensorBoard 이미지 로깅
        if self.tensorboard_writer:
            self.tensorboard_writer.add_image(tag, image, step)
        
        # WandB 이미지 로깅
        if self.wandb_run:
            self.wandb_run.log({tag: wandb.Image(image)}, step=step)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """하이퍼파라미터 로깅"""
        self.logger.info("Hyperparameters:")
        for key, value in hyperparams.items():
            self.logger.info(f"  {key}: {value}")
        
        # 설정 파일로 저장
        config_file = self.experiment_log_dir / "hyperparameters.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(hyperparams, f, default_flow_style=False)
    
    def log_checkpoint(self, checkpoint_path: str, metrics: Optional[Dict[str, float]] = None):
        """체크포인트 로깅"""
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if metrics:
            self.logger.info("Checkpoint metrics:")
            for key, value in metrics.items():
                self.logger.info(f"  {key}: {value:.4f}")
        
        # WandB에 체크포인트 아티팩트 추가
        if self.wandb_run:
            artifact = wandb.Artifact(
                name=f"model-{self.experiment_id}",
                type="model",
                description=f"Model checkpoint for {self.experiment_id}"
            )
            artifact.add_file(checkpoint_path)
            self.wandb_run.log_artifact(artifact)
    
    def log_error(self, error_msg: str, error_type: str = "ERROR"):
        """에러 로깅"""
        self.logger.error(f"{error_type}: {error_msg}")
        
        # WandB에 에러 로깅
        if self.wandb_run:
            self.wandb_run.log({"error": error_msg})
    
    def log_experiment_end(self, final_metrics: Optional[Dict[str, float]] = None):
        """실험 종료 로깅"""
        self.logger.info("="*60)
        self.logger.info(f"EXPERIMENT ENDED: {self.experiment_id}")
        
        if final_metrics:
            self.logger.info("Final Metrics:")
            for key, value in final_metrics.items():
                self.logger.info(f"  {key}: {value:.4f}")
        
        self.logger.info("="*60)
        
        # 메트릭 히스토리 저장
        self._save_metrics_history()
        
        # 로거 정리
        self.close()
    
    def _save_metrics_history(self):
        """메트릭 히스토리 저장"""
        metrics_file = self.experiment_log_dir / "metrics_history.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def get_metrics_history(self, metric_name: str) -> list:
        """특정 메트릭의 히스토리 반환"""
        return self.metrics_history.get(metric_name, [])
    
    def plot_metrics(self, metric_names: list, save_path: Optional[str] = None):
        """메트릭 플롯 생성"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            for metric_name in metric_names:
                if metric_name in self.metrics_history:
                    steps, values = zip(*self.metrics_history[metric_name])
                    plt.plot(steps, values, label=metric_name)
            
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.title(f'Metrics for {self.experiment_id}')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path)
            else:
                plot_path = self.experiment_log_dir / "metrics_plot.png"
                plt.savefig(plot_path)
            
            plt.close()
            
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
    
    def close(self):
        """로거 정리"""
        # TensorBoard 정리
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        # WandB 정리
        if self.wandb_run:
            self.wandb_run.finish()
        
        # 로거 핸들러 정리
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

def create_experiment_logger(experiment_name: str, **kwargs) -> ExperimentLogger:
    """실험 로거 생성 헬퍼 함수"""
    return ExperimentLogger(experiment_name, **kwargs) 