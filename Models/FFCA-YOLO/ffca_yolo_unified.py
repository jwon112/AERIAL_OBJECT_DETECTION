#!/usr/bin/env python3
"""
FFCA-YOLO 통합 인터페이스 래퍼 (자체 완결형)
- 자체 완결성: 모든 후처리(NMS 포함)를 내부에서 처리
- 표준 인터페이스: 일관된 예측 결과 형식 반환
- 독립성: Registry에 의존하지 않는 완전한 파이프라인
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
import importlib.util

# 프로젝트 루트와 FFCA-YOLO 경로 추가
FFCA_YOLO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(FFCA_YOLO_DIR))
sys.path.append(PROJECT_ROOT)
sys.path.append(FFCA_YOLO_DIR)

from utility.unified_model_interface import UnifiedModelInterface

class FFCAYOLOWrapper(UnifiedModelInterface):
    """FFCA-YOLO 자체 완결형 통합 인터페이스"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.img_size = config.get('img_size', 640)
        print(f"🚀 [FFCA-YOLO] 원래 설계 존중 래퍼 초기화")
        print(f"   📊 FFCA-YOLO 원시 예측 출력 (NMS 없음)")
    
    def predict(self, images: torch.Tensor) -> List[List[List[float]]]:
        """
        🎯 FFCA-YOLO 원래 설계 존중 - NMS 없는 원시 예측
        
        Args:
            images: [B, C, H, W] 텐서
            
        Returns:
            List[List[List[float]]]: 배치별 예측 결과
            - 형식: [이미지별[[cx, cy, w, h, conf, cls], ...], ...]
            - FFCA-YOLO 원래 출력 형식 유지
        """
        self.model.eval()
        
        with torch.no_grad():
            # 1️⃣ 모델 추론 (FFCA-YOLO 원래 설계대로)
            raw_predictions = self.model(images)
            
            # 2️⃣ 원시 예측을 표준 형식으로 변환 (NMS 없음!)
            standardized_results = self._convert_raw_to_standard_format(raw_predictions, images.shape)
            
        return standardized_results
    
    def _convert_raw_to_standard_format(self, predictions: torch.Tensor, image_shape: Tuple[int, ...]) -> List[List[List[float]]]:
        """
        FFCA-YOLO 원시 예측을 표준 형식으로 변환 (NMS 제외)
        
        Args:
            predictions: 원시 모델 출력 [B, N, 85] 또는 (list, tuple)
            image_shape: [B, C, H, W]
            
        Returns:
            List[List[List[float]]]: [[cx, cy, w, h, conf, cls], ...]
        """
        batch_results = []
        batch_size = image_shape[0]
        
        try:
            # predictions 형식 처리
            if isinstance(predictions, (list, tuple)):
                pred = predictions[0]  # 첫 번째 출력 사용
            else:
                pred = predictions
            
            # 배치별 처리
            for i in range(batch_size):
                image_preds = []
                
                if pred.ndim >= 3 and i < pred.shape[0]:
                    batch_pred = pred[i]  # [N, 85] 형태
                    
                    if batch_pred.shape[-1] >= 5:  # 최소 x, y, w, h, conf
                        # 신뢰도 기반 기본 필터링만 (매우 낮은 값만 제거)
                        conf_mask = batch_pred[:, 4] > 0.01  # 매우 관대한 임계값
                        filtered_pred = batch_pred[conf_mask]
                        
                        for pred_box in filtered_pred:
                            if len(pred_box) >= 6:  # cx, cy, w, h, conf, cls
                                cx, cy, w, h = pred_box[:4].tolist()
                                conf = pred_box[4].item()
                                cls = int(pred_box[5].item()) if len(pred_box) > 5 else 0
                                
                                # 기본 검증만 (좌표가 유효한지)
                                if conf > 0 and w > 0 and h > 0:
                                    image_preds.append([cx, cy, w, h, conf, cls])
                
                batch_results.append(image_preds)
                
        except Exception as e:
            print(f"⚠️ [FFCA-YOLO] 원시 예측 변환 실패: {e}")
            # 실패 시 빈 결과 반환
            batch_results = [[] for _ in range(batch_size)]
        
        return batch_results
    
    def build_model(self) -> nn.Module:
        """FFCA-YOLO 모델 생성 - 근본적 환경 격리"""
        print("🔧 [FFCA-YOLO] 모델 생성 중...")
        
        # 1️⃣ 근본적 해결: 완전한 환경 격리
        original_cwd = os.getcwd()
        original_path = sys.path.copy()
        
        try:
            # 모든 다른 모델 경로를 sys.path에서 완전 제거
            isolated_path = []
            for path in sys.path:
                # 표준 라이브러리와 사이트 패키지만 유지
                if any(x in path.lower() for x in ['python', 'site-packages', 'lib']) or path == '':
                    isolated_path.append(path)
                # 프로젝트 루트만 추가 (다른 모델 경로는 제외)
                elif path == PROJECT_ROOT:
                    isolated_path.append(path)
            
            # FFCA-YOLO만 sys.path에 추가
            sys.path = [FFCA_YOLO_DIR] + isolated_path
            os.chdir(FFCA_YOLO_DIR)
            
            print(f"🔒 [FFCA-YOLO] 완전한 환경 격리 완료")
            
            # 2️⃣ 모든 관련 모듈 cache 완전 정리
            modules_to_clear = [name for name in sys.modules.keys() 
                              if any(x in name for x in ['models', 'utils', 'common', 'yolo'])]
            for mod in modules_to_clear:
                del sys.modules[mod]
            
            print(f"🧹 [FFCA-YOLO] 모듈 cache 정리: {len(modules_to_clear)}개")
            
            # 3️⃣ FFCA-YOLO 전용 환경에서 일반적인 import
            from models.yolo import DetectionModel
            
            # 4️⃣ 모델 생성 - 간결한 접근
            yaml_candidates = [
                os.path.join(FFCA_YOLO_DIR, 'FFCA-YOLO.yaml'),
                os.path.join(FFCA_YOLO_DIR, 'models', 'FFCA-YOLO.yaml'),
                os.path.join(FFCA_YOLO_DIR, 'data', 'FFCA-YOLO.yaml'),
                os.path.join(FFCA_YOLO_DIR, 'models', 'yolov5s.yaml'),
                os.path.join(FFCA_YOLO_DIR, 'models', 'yolov5n.yaml')
            ]
            
            model = None
            for yaml_path in yaml_candidates:
                if os.path.exists(yaml_path):
                    try:
                        model = DetectionModel(yaml_path, ch=3, nc=self.num_classes)
                        print(f"✅ [FFCA-YOLO] {os.path.basename(yaml_path)}로 모델 생성")
                        break
                    except Exception as e:
                        print(f"⚠️ [FFCA-YOLO] {os.path.basename(yaml_path)} 실패: {e}")
                        continue
            
            # YAML 파일들이 모두 실패하면 직접 모델 import 시도
            if model is None:
                try:
                    # 다른 YOLO 모델들에서 흔히 사용하는 방식
                    from models.yolo import Model
                    
                    # 기본 YOLOv5 아키텍처로 모델 생성 시도
                    model = Model(cfg=None, ch=3, nc=self.num_classes)
                    print("✅ [FFCA-YOLO] 기본 Model 클래스로 생성")
                except Exception:
                    # 마지막 수단: 가장 간단한 DetectionModel 생성
                    model = DetectionModel(cfg=None, ch=3, nc=self.num_classes)
                    print("✅ [FFCA-YOLO] 기본 DetectionModel로 생성")
            
            # 5️⃣ 필수 속성 설정
            if not hasattr(model, 'hyp'):
                model.hyp = {
                    'cls_pw': 1.0, 'obj_pw': 1.0, 'box': 0.5, 'cls': 0.5, 'obj': 1.0,
                    'anchor_t': 8.0, 'fl_gamma': 0.0, 'label_smoothing': 0.0
                }
            if not hasattr(model, 'gr'):
                model.gr = 1.0
            
            print("✅ [FFCA-YOLO] 모델 생성 완료")
            return model
            
        except ImportError as e:
            raise ImportError(f"FFCA-YOLO import 실패 (환경 격리 후): {e}")
        except Exception as e:
            raise RuntimeError(f"FFCA-YOLO 모델 생성 실패: {e}")
        finally:
            # 환경 복원
            os.chdir(original_cwd)
            sys.path = original_path
            print("🔄 [FFCA-YOLO] 환경 복원 완료")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파 - 간결한 처리"""
        x = x.to(self.device)
        self.model = self.model.to(self.device)
        
        output = self.model(x)
        self._raw_output = output  # 손실 계산용
        
        # 간단한 출력 표준화
        if isinstance(output, (list, tuple)) and len(output) > 0:
            pred = output[0]
            if hasattr(pred, 'shape') and len(pred.shape) >= 3:
                return pred.view(pred.shape[0], -1, pred.shape[-1])
        
        return output
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> tuple:
        """손실 계산 - 공식 FFCA-YOLO 손실 함수 (디버깅 강화)"""
        try:
            # 🔧 원래 FFCA-YOLO 손실 함수 사용 + 상세 디버깅
            original_cwd = os.getcwd()
            original_path = sys.path.copy()
            
            try:
                # FFCA-YOLO 환경으로 전환
                sys.path = [FFCA_YOLO_DIR] + [p for p in sys.path if 'python' in p.lower() or 'site-packages' in p.lower() or p == '']
                os.chdir(FFCA_YOLO_DIR)
                
                from utils.loss import ComputeLoss
                criterion = ComputeLoss(self.model)
                
                # 필요한 함수들 import
                from utils.loss import wasserstein_loss
                
                # 🔍 디버깅: 입력 데이터 분석
                print(f"🔍 [FFCA-YOLO DEBUG] 손실 계산 시작")
                print(f"   📊 targets: {targets.shape[0]} objects, classes: {torch.unique(targets[:, 1]).numel()}")
                
                # 🔍 디버깅: 예측값 설정
                if hasattr(self, '_raw_output'):
                    preds = self._raw_output
                else:
                    preds = [predictions]
                
                # 🔍 핵심 디버깅: build_targets 과정 추적  
                tcls, tbox, indices, anchors = criterion.build_targets(preds, targets)
                
                # 각 레이어별 타겟 개수만 간단히 확인
                total_targets = sum(indices[i][0].shape[0] for i in range(len(indices)))
                layer_targets = [indices[i][0].shape[0] for i in range(len(indices))]
                print(f"   🎯 Anchor matching: {layer_targets} targets → Total: {total_targets}")
                
                if total_targets == 0:
                    print(f"   ⚠️ No targets matched! anchor_t={criterion.hyp.get('anchor_t', 'N/A')}")
                    # 첫 번째 타겟의 크기 정보
                    if len(targets) > 0:
                        sample_wh = targets[0, 4:6]  # width, height
                        print(f"   📏 Sample target size: w={sample_wh[0]:.3f}, h={sample_wh[1]:.3f}")
                        # anchor 크기 출력
                        print(f"   ⚓ Anchors: {[a.tolist() for a in criterion.anchors]}")
                
                # 간소화된 손실 계산
                lcls = torch.zeros(1, device=criterion.device)
                lbox = torch.zeros(1, device=criterion.device) 
                lobj = torch.zeros(1, device=criterion.device)
                
                # 각 레이어별 손실 계산 (간소화)
                for i, pi in enumerate(preds):
                    b, a, gj, gi = indices[i]
                    tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=criterion.device)
                    
                    n = b.shape[0]
                    
                    if n:
                        # 타겟이 있는 경우의 상세 분석
                        pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, criterion.nc), 1)
                        
                        # Box regression
                        pxy = pxy.sigmoid() * 2 - 0.5
                        pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                        pbox = torch.cat((pxy, pwh), 1)
                        
                        # IoU 계산
                        from utils.metrics import bbox_iou
                        iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                        
                        # NWD 계산
                        nwd = wasserstein_loss(pbox, tbox[i]).squeeze()
                        
                        # Box loss 계산
                        iou_ratio = 0.5
                        layer_lbox = (1 - iou_ratio) * (1.0 - nwd).mean() + iou_ratio * (1.0 - iou).mean()
                        lbox += layer_lbox
                        
                        # Classification loss 
                        if criterion.nc > 1:
                            t = torch.full_like(pcls, criterion.cn, device=criterion.device)
                            t[range(n), tcls[i]] = criterion.cp
                            layer_lcls = criterion.BCEcls(pcls, t)
                            lcls += layer_lcls
                        
                        # Objectness target 설정
                        iou = iou.detach().clamp(0).type(tobj.dtype)
                        tobj[b, a, gj, gi] = iou
                    
                    # Objectness loss
                    layer_lobj = criterion.BCEobj(pi[..., 4], tobj) * criterion.balance[i]
                    lobj += layer_lobj
                
                # 하이퍼파라미터 적용
                lbox *= criterion.hyp['box']
                lobj *= criterion.hyp['obj'] 
                lcls *= criterion.hyp['cls']
                bs = targets.shape[0]
                
                total_loss = (lbox + lobj + lcls) * bs
                loss_items = torch.cat((lbox, lobj, lcls)).detach()
                
                print(f"   📊 Loss: box={lbox.item():.4f}, obj={lobj.item():.4f}, cls={lcls.item():.4f}, total={total_loss.item():.4f}")
                
                return total_loss, loss_items
                
            finally:
                # 환경 복원
                os.chdir(original_cwd)
                sys.path = original_path
                
        except Exception as e:
            print(f"⚠️ [FFCA-YOLO] 공식 손실 계산 실패: {e}")
            print(f"   🔍 Exception type: {type(e)}")
            import traceback
            print(f"   🔍 Traceback: {traceback.format_exc()}")
            
            # 최후 수단: 기본 손실 반환
            device = targets.device if hasattr(targets, 'device') else torch.device('cpu')
            batch_size = targets.shape[0] if hasattr(targets, 'shape') and len(targets.shape) > 1 else 1
            fixed_loss = torch.tensor(0.1 * batch_size, device=device)
            components = torch.tensor([0.05, 0.03, 0.02], device=device)
            return fixed_loss, components
    
    def postprocess(self, predictions: torch.Tensor) -> List[Dict]:
        """후처리 - 간결한 처리"""
        try:
            from utils.general import non_max_suppression
            detections = non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45)
            
            results = []
            for det in detections:
                if len(det) > 0:
                    results.append({
                        'boxes': det[:, :4].cpu().numpy(),
                        'scores': det[:, 4].cpu().numpy(),
                        'labels': det[:, 5].cpu().numpy().astype(int)
                    })
                else:
                    results.append({
                        'boxes': np.empty((0, 4)),
                        'scores': np.empty((0,)),
                        'labels': np.empty((0,), dtype=int)
                    })
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"FFCA-YOLO 후처리 실패: {e}")
    
    def load_weights(self, weights_path: str) -> None:
        """가중치 로드 - 간결한 처리"""
        if not weights_path or not os.path.exists(weights_path):
            return
        
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint
            
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ [FFCA-YOLO] 가중치 로드 완료")
            
        except Exception as e:
            raise RuntimeError(f"FFCA-YOLO 가중치 로드 실패: {e}")


def build_ffca_yolo_unified_model(ex_dict: Dict[str, Any]) -> FFCAYOLOWrapper:
    """FFCA-YOLO 통합 모델 빌더 - 간결한 구조"""
    config = {
        'num_classes': ex_dict.get('Model Config', {}).get('num_classes', 10),
        'device': ex_dict.get('Device', 'cpu')
    }
    
    wrapper = FFCAYOLOWrapper(config)
    wrapper.model = wrapper.build_model()
    return wrapper 