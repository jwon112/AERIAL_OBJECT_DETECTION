"""
Visualization Module
객체 탐지 결과 시각화를 위한 모듈

이 모듈은 다음과 같은 기능을 제공합니다:
1. 바운딩 박스 시각화
2. 예측 결과 플롯
3. 성능 메트릭 시각화
4. 학습 곡선 플롯
5. 결과 이미지 저장
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
from typing import List, Dict, Any, Optional, Tuple
import seaborn as sns
from datetime import datetime

from utility.debug_logger import debug_log

class DetectionVisualizer:
    """
    객체 탐지 결과 시각화 클래스
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        시각화기 초기화
        
        Args:
            class_names: 클래스 이름 리스트
        """
        self.class_names = class_names or []
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        debug_log("DetectionVisualizer initialized")
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]], 
                       confidence_threshold: float = 0.5) -> np.ndarray:
        """
        이미지에 탐지 결과 그리기
        
        Args:
            image: 원본 이미지 (BGR 형식)
            detections: 탐지 결과 리스트 [{'bbox': [x1, y1, x2, y2], 'class_id': int, 'confidence': float}]
            confidence_threshold: 신뢰도 임계값
            
        Returns:
            시각화된 이미지
        """
        vis_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_id = detection['class_id']
            confidence = detection['confidence']
            
            if confidence < confidence_threshold:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors[class_id % len(self.colors)]
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            if class_id < len(self.class_names):
                label = f"{self.class_names[class_id]} {confidence:.2f}"
                cv2.putText(vis_image, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_image
    
    def plot_metrics(self, metrics: Dict[str, float], save_path: Optional[str] = None):
        """
        메트릭 시각화
        
        Args:
            metrics: 메트릭 딕셔너리
            save_path: 저장 경로 (선택사항)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        names, values = zip(*metrics.items())
        bars = ax.bar(range(len(names)), values, color='skyblue')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Value')
        ax.set_title('Detection Metrics')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_curves(self, train_losses: List[float], 
                           val_losses: Optional[List[float]] = None,
                           train_metrics: Optional[Dict[str, List[float]]] = None,
                           val_metrics: Optional[Dict[str, List[float]]] = None,
                           save_path: Optional[str] = None) -> None:
        """
        학습 곡선 시각화
        
        Args:
            train_losses: 학습 손실 리스트
            val_losses: 검증 손실 리스트 (선택사항)
            train_metrics: 학습 메트릭 딕셔너리 (선택사항)
            val_metrics: 검증 메트릭 딕셔너리 (선택사항)
            save_path: 저장 경로 (선택사항)
        """
        epochs = range(1, len(train_losses) + 1)
        
        # 서브플롯 개수 결정
        num_plots = 1
        if val_losses is not None:
            num_plots += 1
        if train_metrics is not None:
            num_plots += len(train_metrics)
        
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
        if num_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 손실 곡선
        ax = axes[plot_idx]
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses is not None:
            ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
        
        # 메트릭 곡선
        if train_metrics is not None:
            for metric_name, train_values in train_metrics.items():
                if plot_idx < len(axes):
                    ax = axes[plot_idx]
                    ax.plot(epochs, train_values, 'b-', label=f'Train {metric_name}', linewidth=2)
                    
                    if val_metrics and metric_name in val_metrics:
                        ax.plot(epochs, val_metrics[metric_name], 'r-', 
                               label=f'Val {metric_name}', linewidth=2)
                    
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel(metric_name)
                    ax.set_title(f'{metric_name} over epochs')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plot_idx += 1
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            debug_log(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def create_confusion_matrix_plot(self, confusion_matrix: np.ndarray,
                                   class_names: Optional[List[str]] = None,
                                   save_path: Optional[str] = None) -> None:
        """
        혼동 행렬 시각화
        
        Args:
            confusion_matrix: 혼동 행렬 (numpy array)
            class_names: 클래스 이름 리스트 (선택사항)
            save_path: 저장 경로 (선택사항)
        """
        plt.figure(figsize=(10, 8))
        
        # 히트맵 그리기
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            debug_log(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def save_detection_image(self, image: np.ndarray, detections: List[Dict[str, Any]],
                           output_path: str, confidence_threshold: float = 0.5) -> None:
        """
        탐지 결과 이미지 저장
        
        Args:
            image: 원본 이미지
            detections: 탐지 결과
            output_path: 출력 경로
            confidence_threshold: 신뢰도 임계값
        """
        # 출력 디렉토리 생성
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 시각화된 이미지 생성
        vis_image = self.draw_detections(image, detections, confidence_threshold)
        
        # 이미지 저장
        cv2.imwrite(output_path, vis_image)
        debug_log(f"Detection image saved to {output_path}")

def create_visualizer(class_names: Optional[List[str]] = None) -> DetectionVisualizer:
    """
    시각화기 생성 함수
    
    Args:
        class_names: 클래스 이름 리스트
        
    Returns:
        DetectionVisualizer 인스턴스
    """
    return DetectionVisualizer(class_names)

# 독립적인 함수로 export
def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]], 
                   class_names: Optional[List[str]] = None,
                   confidence_threshold: float = 0.5) -> np.ndarray:
    """이미지에 탐지 결과 그리기 (독립 함수)"""
    visualizer = DetectionVisualizer(class_names)
    return visualizer.draw_detections(image, detections, confidence_threshold) 