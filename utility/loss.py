"""
Loss Functions Utility
YOLO 스타일 손실 함수들
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class YOLOLoss(nn.Module):
    """YOLO 스타일 손실 함수"""
    
    def __init__(self, num_classes=10, anchors=None, lambda_coord=5.0, lambda_noobj=0.5):
        """
        YOLO 손실 함수 초기화
        
        Args:
            num_classes: 클래스 수
            anchors: 앵커 박스들
            lambda_coord: 좌표 손실 가중치
            lambda_noobj: 객체 없음 손실 가중치
        """
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
        # 기본 앵커 박스 (YOLO v5 스타일)
        if anchors is None:
            self.anchors = torch.tensor([
                [10, 13], [16, 30], [33, 23],
                [30, 61], [62, 45], [59, 119],
                [116, 90], [156, 198], [373, 326]
            ])
        else:
            self.anchors = torch.tensor(anchors)
    
    def forward(self, predictions, targets):
        """
        손실 계산
        
        Args:
            predictions: 모델 예측 [batch_size, num_anchors, 5+num_classes]
            targets: 타겟 [batch_size, num_objects, 5] (class_id, x, y, w, h)
            
        Returns:
            총 손실
        """
        batch_size = predictions.size(0)
        num_anchors = predictions.size(1)
        
        # 예측 분해
        pred_xy = predictions[..., :2]  # 중심점
        pred_wh = predictions[..., 2:4]  # 너비, 높이
        pred_conf = predictions[..., 4]  # 신뢰도
        pred_cls = predictions[..., 5:]  # 클래스 확률
        
        # 손실 초기화
        loss_xy = 0
        loss_wh = 0
        loss_conf = 0
        loss_cls = 0
        
        for b in range(batch_size):
            target = targets[b]
            if len(target) == 0:
                # 객체가 없는 경우 신뢰도 손실만 계산
                loss_conf += self.lambda_noobj * torch.sum(pred_conf[b] ** 2)
                continue
            
            # 각 타겟에 대해 가장 가까운 앵커 찾기
            for obj in target:
                class_id = int(obj[0])
                target_x, target_y = obj[1], obj[2]
                target_w, target_h = obj[3], obj[4]
                
                # 앵커와의 IoU 계산하여 가장 가까운 앵커 선택
                best_iou = 0
                best_anchor_idx = 0
                
                for a in range(num_anchors):
                    anchor_w, anchor_h = self.anchors[a]
                    iou = self._calculate_iou(
                        target_x, target_y, target_w, target_h,
                        0, 0, anchor_w, anchor_h
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor_idx = a
                
                # 좌표 손실
                pred_x, pred_y = pred_xy[b, best_anchor_idx]
                pred_w, pred_h = pred_wh[b, best_anchor_idx]
                
                loss_xy += self.lambda_coord * (
                    (pred_x - target_x) ** 2 + (pred_y - target_y) ** 2
                )
                
                loss_wh += self.lambda_coord * (
                    (torch.sqrt(pred_w) - torch.sqrt(target_w)) ** 2 +
                    (torch.sqrt(pred_h) - torch.sqrt(target_h)) ** 2
                )
                
                # 신뢰도 손실
                loss_conf += (pred_conf[b, best_anchor_idx] - 1.0) ** 2
                
                # 클래스 손실
                pred_class = pred_cls[b, best_anchor_idx]
                target_class = torch.zeros(self.num_classes)
                target_class[class_id] = 1.0
                
                loss_cls += F.cross_entropy(pred_class.unsqueeze(0), 
                                          target_class.unsqueeze(0))
                
                # 객체가 없는 앵커들에 대한 신뢰도 손실
                for a in range(num_anchors):
                    if a != best_anchor_idx:
                        loss_conf += self.lambda_noobj * (pred_conf[b, a] - 0.0) ** 2
        
        total_loss = loss_xy + loss_wh + loss_conf + loss_cls
        
        return total_loss
    
    def _calculate_iou(self, x1, y1, w1, h1, x2, y2, w2, h2):
        """IoU 계산"""
        # 박스 좌표 계산
        x1_min, x1_max = x1 - w1/2, x1 + w1/2
        y1_min, y1_max = y1 - h1/2, y1 + h1/2
        x2_min, x2_max = x2 - w2/2, x2 + w2/2
        y2_min, y2_max = y2 - h2/2, y2 + h2/2
        
        # 교집합 계산
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max <= x_min or y_max <= y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # 합집합 계산
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union

class SimpleDetectionLoss(nn.Module):
    """간단한 객체 탐지 손실 함수"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        """
        간단한 손실 계산
        
        Args:
            predictions: 모델 예측
            targets: 타겟
            
        Returns:
            총 손실
        """
        if isinstance(predictions, (list, tuple)):
            # 여러 스케일의 예측이 있는 경우
            total_loss = 0
            for pred in predictions:
                total_loss += self._compute_loss(pred, targets)
            return total_loss
        else:
            return self._compute_loss(predictions, targets)
    
    def _compute_loss(self, predictions, targets):
        """단일 예측에 대한 손실 계산"""
        # 간단한 MSE 손실
        if targets is None or len(targets) == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # 예측을 타겟과 같은 형태로 변환
        if len(predictions.shape) == 3:
            # [batch, num_detections, features] -> [batch, features]
            predictions = predictions.mean(dim=1)
        
        # 타겟을 예측과 같은 형태로 변환
        if isinstance(targets, list):
            targets = torch.stack(targets)
        
        return self.mse_loss(predictions, targets)

class FasterRCNNLoss(nn.Module):
    """Faster R-CNN 스타일 손실 함수 (DNTR용)"""
    
    def __init__(self, num_classes=10, lambda_rpn_cls=1.0, lambda_rpn_reg=1.0, 
                 lambda_rcnn_cls=1.0, lambda_rcnn_reg=1.0):
        """
        Faster R-CNN 손실 함수 초기화
        
        Args:
            num_classes: 클래스 수
            lambda_rpn_cls: RPN 분류 손실 가중치
            lambda_rpn_reg: RPN 회귀 손실 가중치
            lambda_rcnn_cls: R-CNN 분류 손실 가중치
            lambda_rcnn_reg: R-CNN 회귀 손실 가중치
        """
        super().__init__()
        self.num_classes = num_classes
        self.lambda_rpn_cls = lambda_rpn_cls
        self.lambda_rpn_reg = lambda_rpn_reg
        self.lambda_rcnn_cls = lambda_rcnn_cls
        self.lambda_rcnn_reg = lambda_rcnn_reg
        
        # 손실 함수들
        self.rpn_cls_loss = nn.CrossEntropyLoss()
        self.rpn_reg_loss = nn.SmoothL1Loss()
        self.rcnn_cls_loss = nn.CrossEntropyLoss()
        self.rcnn_reg_loss = nn.SmoothL1Loss()
    
    def forward(self, predictions, targets):
        """
        손실 계산
        
        Args:
            predictions: 모델 예측 (RPN + R-CNN 출력)
            targets: 타겟 정보
            
        Returns:
            총 손실
        """
        if isinstance(predictions, dict):
            # MMCV 스타일 출력
            return self._compute_mmcv_loss(predictions, targets)
        else:
            # 간단한 출력
            return self._compute_simple_loss(predictions, targets)
    
    def _compute_mmcv_loss(self, predictions, targets):
        """MMCV 스타일 출력에 대한 손실 계산"""
        total_loss = 0
        
        # RPN 손실
        if 'rpn_cls_score' in predictions and 'rpn_bbox_pred' in predictions:
            rpn_cls_loss = self.rpn_cls_loss(
                predictions['rpn_cls_score'], 
                targets.get('rpn_labels', torch.zeros(1, dtype=torch.long))
            )
            rpn_reg_loss = self.rpn_reg_loss(
                predictions['rpn_bbox_pred'],
                targets.get('rpn_bbox_targets', torch.zeros_like(predictions['rpn_bbox_pred']))
            )
            total_loss += self.lambda_rpn_cls * rpn_cls_loss + self.lambda_rpn_reg * rpn_reg_loss
        
        # R-CNN 손실
        if 'cls_score' in predictions and 'bbox_pred' in predictions:
            rcnn_cls_loss = self.rcnn_cls_loss(
                predictions['cls_score'],
                targets.get('labels', torch.zeros(1, dtype=torch.long))
            )
            rcnn_reg_loss = self.rcnn_reg_loss(
                predictions['bbox_pred'],
                targets.get('bbox_targets', torch.zeros_like(predictions['bbox_pred']))
            )
            total_loss += self.lambda_rcnn_cls * rcnn_cls_loss + self.lambda_rcnn_reg * rcnn_reg_loss
        
        return total_loss
    
    def _compute_simple_loss(self, predictions, targets):
        """간단한 출력에 대한 손실 계산"""
        if targets is None or len(targets) == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # 간단한 MSE 손실
        if isinstance(predictions, (list, tuple)):
            total_loss = 0
            for pred in predictions:
                total_loss += torch.mean((pred - targets) ** 2)
            return total_loss
        else:
            return torch.mean((predictions - targets) ** 2) 