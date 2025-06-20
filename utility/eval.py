"""
Evaluation Module
모델 평가를 위한 범용적인 평가 루프를 제공하는 모듈

이 모듈은 다음과 같은 기능을 제공합니다:
1. 표준화된 평가 루프
2. 다양한 메트릭 계산 (mAP, AP50, AP75, Precision, Recall 등)
3. 클래스별 성능 분석
4. 평가 결과 저장 및 시각화
"""

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import warnings

from utility.debug_logger import debug_log
from utility.metrics import calculate_map, calculate_ap, calculate_precision_recall
from utility.utils import save_evaluation_results

class Evaluator:
    """
    범용적인 모델 평가 클래스
    """
    
    def __init__(self, model, val_loader, device='cpu', config=None):
        """
        평가기 초기화
        
        Args:
            model: 평가할 모델
            val_loader: 검증 데이터로더
            device: 평가 디바이스
            config: 평가 설정
        """
        self.model = model
        self.val_loader = val_loader
        self.device = device
        self.config = config or {}
        
        # 모델을 디바이스로 이동
        self.model.to(device)
        
        # 평가 결과 저장
        self.results_dir = Path(self.config.get('results_dir', './evaluation_results'))
        self.results_dir.mkdir(exist_ok=True)
        
        debug_log("Evaluator initialized")
    
    def evaluate(self, iou_thresholds=None, class_names=None):
        """
        모델 평가 실행
        
        Args:
            iou_thresholds: IoU 임계값 리스트 (기본값: [0.5, 0.75])
            class_names: 클래스 이름 리스트
            
        Returns:
            평가 결과 딕셔너리
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.75]
        
        debug_log(f"Starting evaluation with IoU thresholds: {iou_thresholds}")
        
        self.model.eval()
        
        # 예측 및 실제값 수집
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
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
                
                # 모델 예측
                outputs = self.model(images)
                
                # 예측 결과 처리
                predictions = self._process_predictions(outputs, images.shape[0])
                all_predictions.extend(predictions)
                all_targets.extend(self._process_targets(targets))
        
        # 메트릭 계산
        results = self._calculate_metrics(all_predictions, all_targets, iou_thresholds, class_names)
        
        # 결과 저장
        self._save_results(results)
        
        debug_log("Evaluation completed")
        return results
    
    def _process_predictions(self, outputs, batch_size):
        """모델 출력을 예측 형식으로 변환"""
        predictions = []
        
        # 모델별 출력 형식 처리
        if hasattr(self.model, 'process_predictions'):
            # 모델에 내장된 처리 함수가 있는 경우
            predictions = self.model.process_predictions(outputs)
        else:
            # 기본 처리 (YOLO 스타일)
            if isinstance(outputs, (list, tuple)):
                # YOLO v5/v8 스타일 출력
                for i in range(batch_size):
                    batch_preds = []
                    for output in outputs:
                        if output is not None and len(output) > i:
                            pred = output[i]  # (num_detections, 6) [x1, y1, x2, y2, conf, cls]
                            if len(pred) > 0:
                                batch_preds.append(pred.cpu().numpy())
                    
                    if batch_preds:
                        predictions.append(np.concatenate(batch_preds, axis=0))
                    else:
                        predictions.append(np.empty((0, 6)))
            else:
                # 단일 텐서 출력
                for i in range(batch_size):
                    if outputs is not None and len(outputs) > i:
                        pred = outputs[i]
                        if len(pred) > 0:
                            predictions.append(pred.cpu().numpy())
                        else:
                            predictions.append(np.empty((0, 6)))
                    else:
                        predictions.append(np.empty((0, 6)))
        
        return predictions
    
    def _process_targets(self, targets):
        """타겟을 평가 형식으로 변환"""
        processed_targets = []
        
        if isinstance(targets, list):
            for target in targets:
                if target is not None and len(target) > 0:
                    # (class_id, x_center, y_center, width, height) -> (x1, y1, x2, y2, class_id)
                    if target.shape[1] == 5:
                        x1 = target[:, 1] - target[:, 3] / 2
                        y1 = target[:, 2] - target[:, 4] / 2
                        x2 = target[:, 1] + target[:, 3] / 2
                        y2 = target[:, 2] + target[:, 4] / 2
                        class_ids = target[:, 0]
                        
                        processed_target = torch.stack([x1, y1, x2, y2, class_ids], dim=1)
                        processed_targets.append(processed_target.cpu().numpy())
                    else:
                        processed_targets.append(target.cpu().numpy())
                else:
                    processed_targets.append(np.empty((0, 5)))
        else:
            # 단일 텐서
            if targets is not None and len(targets) > 0:
                if targets.shape[1] == 5:
                    x1 = targets[:, 1] - targets[:, 3] / 2
                    y1 = targets[:, 2] - targets[:, 4] / 2
                    x2 = targets[:, 1] + targets[:, 3] / 2
                    y2 = targets[:, 2] + targets[:, 4] / 2
                    class_ids = targets[:, 0]
                    
                    processed_target = torch.stack([x1, y1, x2, y2, class_ids], dim=1)
                    processed_targets.append(processed_target.cpu().numpy())
                else:
                    processed_targets.append(targets.cpu().numpy())
            else:
                processed_targets.append(np.empty((0, 5)))
        
        return processed_targets
    
    def _calculate_metrics(self, predictions, targets, iou_thresholds, class_names):
        """메트릭 계산"""
        results = {
            'iou_thresholds': iou_thresholds,
            'class_names': class_names,
            'metrics': {},
            'class_metrics': {},
            'detailed_results': {}
        }
        
        # 각 IoU 임계값에 대해 메트릭 계산
        for iou_thresh in iou_thresholds:
            debug_log(f"Calculating metrics for IoU threshold: {iou_thresh}")
            
            # 전체 mAP 계산
            map_score = calculate_map(predictions, targets, iou_thresh, class_names)
            results['metrics'][f'mAP@{iou_thresh}'] = map_score
            
            # 클래스별 AP 계산
            class_aps = calculate_ap(predictions, targets, iou_thresh, class_names)
            results['class_metrics'][f'AP@{iou_thresh}'] = class_aps
            
            # Precision, Recall 계산
            precision, recall = calculate_precision_recall(predictions, targets, iou_thresh, class_names)
            results['metrics'][f'Precision@{iou_thresh}'] = precision
            results['metrics'][f'Recall@{iou_thresh}'] = recall
        
        # 평균 메트릭 계산
        if len(iou_thresholds) > 1:
            avg_map = np.mean([results['metrics'][f'mAP@{iou}'] for iou in iou_thresholds])
            results['metrics']['mAP'] = avg_map
            
            avg_precision = np.mean([results['metrics'][f'Precision@{iou}'] for iou in iou_thresholds])
            avg_recall = np.mean([results['metrics'][f'Recall@{iou}'] for iou in iou_thresholds])
            results['metrics']['Precision'] = avg_precision
            results['metrics']['Recall'] = avg_recall
        
        # F1 Score 계산
        if 'Precision' in results['metrics'] and 'Recall' in results['metrics']:
            precision = results['metrics']['Precision']
            recall = results['metrics']['Recall']
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0
            results['metrics']['F1_Score'] = f1_score
        
        # 상세 결과 저장
        results['detailed_results'] = {
            'predictions': predictions,
            'targets': targets,
            'num_samples': len(predictions)
        }
        
        return results
    
    def _save_results(self, results):
        """평가 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON 결과 저장
        results_file = self.results_dir / f'evaluation_results_{timestamp}.json'
        
        # 상세 결과는 제외하고 저장 (파일 크기 문제)
        save_results = results.copy()
        save_results.pop('detailed_results', None)
        
        with open(results_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        debug_log(f"Evaluation results saved: {results_file}")
        
        # 요약 결과 출력
        self._print_summary(results)
    
    def _print_summary(self, results):
        """평가 결과 요약 출력"""
        print("\n" + "="*50)
        print("EVALUATION RESULTS SUMMARY")
        print("="*50)
        
        # 전체 메트릭 출력
        print("\nOverall Metrics:")
        for metric_name, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
        
        # 클래스별 메트릭 출력
        if results['class_metrics']:
            print("\nClass-wise Metrics:")
            for iou_key, class_aps in results['class_metrics'].items():
                print(f"\n  {iou_key}:")
                for class_id, ap_score in class_aps.items():
                    class_name = results['class_names'][class_id] if results['class_names'] else f"Class_{class_id}"
                    print(f"    {class_name}: {ap_score:.4f}")
        
        print("="*50)
    
    def evaluate_single_image(self, image, target=None):
        """
        단일 이미지 평가
        
        Args:
            image: 평가할 이미지 (tensor)
            target: 실제값 (선택사항)
            
        Returns:
            예측 결과
        """
        self.model.eval()
        
        with torch.no_grad():
            image = image.to(self.device).unsqueeze(0)  # 배치 차원 추가
            output = self.model(image)
            
            # 예측 처리
            predictions = self._process_predictions(output, 1)
            
            if target is not None:
                # 타겟이 있는 경우 메트릭 계산
                targets = self._process_targets([target])
                metrics = self._calculate_metrics(predictions, targets, [0.5], None)
                return predictions[0], metrics
            else:
                return predictions[0]

def create_evaluator(model, val_loader, device='cpu', config=None):
    """평가기 생성 헬퍼 함수"""
    return Evaluator(
        model=model,
        val_loader=val_loader,
        device=device,
        config=config
    ) 