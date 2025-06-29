"""
Prediction Module
모델 추론을 위한 범용적인 예측 모듈

이 모듈은 다음과 같은 기능을 제공합니다:
1. 단일 이미지 추론
2. 배치 이미지 추론
3. 비디오 추론
4. 결과 시각화
5. 추론 결과 저장
"""

import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import json
from datetime import datetime
from tqdm import tqdm
import warnings

from utility.debug_logger import debug_log
from utility.visualizer import draw_detections

class Predictor:
    """
    범용적인 모델 추론 클래스
    """
    
    def __init__(self, model, device='cpu', config=None):
        """
        추론기 초기화
        
        Args:
            model: 추론할 모델
            device: 추론 디바이스
            config: 추론 설정
        """
        self.model = model
        self.device = device
        self.config = config or {}
        
        # 모델을 디바이스로 이동
        self.model.to(device)
        self.model.eval()
        
        # 결과 저장 경로
        self.results_dir = Path(self.config.get('results_dir', './prediction_results'))
        self.results_dir.mkdir(exist_ok=True)
        
        # 추론 설정
        self.conf_threshold = self.config.get('conf_threshold', 0.25)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        self.max_detections = self.config.get('max_detections', 300)
        
        debug_log("Predictor initialized")
    
    def predict_image(self, image_path, save_result=True, class_names=None):
        """
        단일 이미지 추론
        
        Args:
            image_path: 이미지 경로 또는 PIL Image
            save_result: 결과 저장 여부
            class_names: 클래스 이름 리스트
            
        Returns:
            추론 결과 딕셔너리
        """
        # 이미지 로드
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
            image_name = Path(image_path).name
        elif isinstance(image_path, Image.Image):
            image = image_path
            image_name = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        else:
            raise ValueError(f"Unsupported image format: {type(image_path)}")
        
        # 이미지 전처리
        input_tensor = self._preprocess_image(image)
        
        # 추론 실행
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # 결과 후처리
        detections = self._postprocess_predictions(predictions, input_tensor.shape[2:])
        
        # 결과 구성
        result = {
            'image_name': image_name,
            'detections': detections,
            'image_size': image.size,
            'inference_time': datetime.now().isoformat()
        }
        
        # 결과 시각화 및 저장
        if save_result:
            self._save_image_result(image, detections, image_name, class_names)
        
        return result
    
    def predict_batch(self, image_paths, save_results=True, class_names=None):
        """
        배치 이미지 추론
        
        Args:
            image_paths: 이미지 경로 리스트
            save_results: 결과 저장 여부
            class_names: 클래스 이름 리스트
            
        Returns:
            추론 결과 리스트
        """
        results = []
        
        for image_path in tqdm(image_paths, desc="Predicting"):
            try:
                result = self.predict_image(image_path, save_results, class_names)
                results.append(result)
            except Exception as e:
                debug_log(f"Error predicting {image_path}: {e}")
                results.append({
                    'image_name': Path(image_path).name,
                    'error': str(e)
                })
        
        return results
    
    def predict_dataset(self, test_loader, save_results=True, class_names=None):
        """
        데이터셋 추론
        
        Args:
            test_loader: 테스트 데이터로더
            save_results: 결과 저장 여부
            class_names: 클래스 이름 리스트
            
        Returns:
            추론 결과 리스트
        """
        results = []
        
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Predicting dataset")):
            # 배치 데이터 준비
            if isinstance(batch, (list, tuple)):
                images, targets = batch
            elif isinstance(batch, dict):
                images = batch['images']
                targets = batch.get('targets', None)
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")
            
            # 디바이스로 이동
            images = images.to(self.device)
            
            # 추론 실행
            with torch.no_grad():
                predictions = self.model(images)
            
            # 배치 결과 처리
            batch_results = self._process_batch_predictions(
                predictions, images, targets, batch_idx, class_names
            )
            results.extend(batch_results)
        
        # 전체 결과 저장
        if save_results:
            self._save_dataset_results(results)
        
        return results
    
    def predict_video(self, video_path, output_path=None, class_names=None, 
                     frame_skip=1, save_frames=False):
        """
        비디오 추론
        
        Args:
            video_path: 비디오 파일 경로
            output_path: 출력 비디오 경로
            class_names: 클래스 이름 리스트
            frame_skip: 프레임 건너뛰기 간격
            save_frames: 개별 프레임 저장 여부
            
        Returns:
            비디오 추론 결과
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # 비디오 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 출력 비디오 설정
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임 건너뛰기
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 추론
            result = self.predict_image(pil_image, save_result=False, class_names=class_names)
            result['frame_number'] = frame_count
            results.append(result)
            
            # 결과 시각화
            if output_path or save_frames:
                annotated_frame = self._draw_detections_on_frame(frame, result['detections'], class_names)
                
                if output_path:
                    out.write(annotated_frame)
                
                if save_frames:
                    frame_path = self.results_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), annotated_frame)
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
        
        # 비디오 결과 저장
        self._save_video_results(results, video_path)
        
        return results
    
    def _preprocess_image(self, image):
        """이미지 전처리"""
        # 이미지 크기 조정
        target_size = self.config.get('input_size', (640, 640))
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        
        # 리사이즈
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # 텐서 변환
        image_tensor = torch.from_numpy(np.array(image_resized)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # HWC to CHW, normalize
        
        # 배치 차원 추가
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def _postprocess_predictions(self, predictions, original_size):
        """예측 결과 후처리"""
        detections = []
        
        # 모델별 출력 형식 처리
        if hasattr(self.model, 'process_predictions'):
            # 모델에 내장된 처리 함수가 있는 경우
            processed_preds = self.model.process_predictions(predictions)
            if isinstance(processed_preds, list):
                detections = processed_preds[0]  # 첫 번째 이미지 결과
            else:
                detections = processed_preds
        else:
            # 기본 처리 (YOLO 스타일)
            if isinstance(predictions, (list, tuple)):
                # YOLO v5/v8 스타일 출력
                for pred in predictions:
                    if pred is not None and len(pred) > 0:
                        pred_np = pred[0].cpu().numpy()  # 첫 번째 이미지
                        detections = self._filter_detections(pred_np)
                        break
            else:
                # 단일 텐서 출력
                if predictions is not None and len(predictions) > 0:
                    pred_np = predictions[0].cpu().numpy()
                    detections = self._filter_detections(pred_np)
        
        # 좌표 스케일링 (원본 이미지 크기로)
        if len(detections) > 0:
            detections = self._scale_detections(detections, original_size)
        
        return detections
    
    def _filter_detections(self, predictions):
        """신뢰도 및 IoU 임계값에 따른 검출 결과 필터링"""
        if len(predictions) == 0:
            return []
        
        # 신뢰도 임계값 필터링
        conf_mask = predictions[:, 4] >= self.conf_threshold
        filtered_preds = predictions[conf_mask]
        
        if len(filtered_preds) == 0:
            return []
        
        # NMS 적용
        if hasattr(self.model, 'apply_nms'):
            filtered_preds = self.model.apply_nms(filtered_preds, self.iou_threshold)
        else:
            # 기본 NMS 구현
            filtered_preds = self._apply_nms(filtered_preds, self.iou_threshold)
        
        # 최대 검출 수 제한
        if len(filtered_preds) > self.max_detections:
            # 신뢰도 기준으로 정렬하여 상위 결과만 유지
            conf_scores = filtered_preds[:, 4]
            top_indices = np.argsort(conf_scores)[-self.max_detections:]
            filtered_preds = filtered_preds[top_indices]
        
        return filtered_preds
    
    def _apply_nms(self, predictions, iou_threshold):
        """기본 NMS 구현"""
        if len(predictions) == 0:
            return predictions
        
        # 신뢰도 기준으로 정렬
        sorted_indices = np.argsort(predictions[:, 4])[::-1]
        keep_indices = []
        
        while len(sorted_indices) > 0:
            # 가장 높은 신뢰도를 가진 박스 선택
            current_idx = sorted_indices[0]
            keep_indices.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # 현재 박스와 나머지 박스들의 IoU 계산
            current_box = predictions[current_idx, :4]
            remaining_boxes = predictions[sorted_indices[1:], :4]
            
            ious = self._calculate_iou(current_box, remaining_boxes)
            
            # IoU 임계값보다 낮은 박스들만 유지
            low_iou_mask = ious < iou_threshold
            sorted_indices = sorted_indices[1:][low_iou_mask]
        
        return predictions[keep_indices]
    
    def _calculate_iou(self, box, boxes):
        """IoU 계산"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        return intersection / (union + 1e-6)
    
    def _scale_detections(self, detections, original_size):
        """검출 결과를 원본 이미지 크기로 스케일링"""
        if len(detections) == 0:
            return detections
        
        # 모델 입력 크기
        model_size = self.config.get('input_size', (640, 640))
        if isinstance(model_size, int):
            model_size = (model_size, model_size)
        
        # 스케일링 비율
        scale_x = original_size[1] / model_size[0]  # width
        scale_y = original_size[0] / model_size[1]  # height
        
        # 좌표 스케일링
        scaled_detections = detections.copy()
        scaled_detections[:, [0, 2]] *= scale_x  # x coordinates
        scaled_detections[:, [1, 3]] *= scale_y  # y coordinates
        
        return scaled_detections
    
    def _process_batch_predictions(self, predictions, images, targets, batch_idx, class_names):
        """배치 예측 결과 처리"""
        batch_results = []
        
        for i in range(images.shape[0]):
            # 개별 이미지 예측 결과 추출
            if isinstance(predictions, (list, tuple)):
                # YOLO 스타일 출력
                image_preds = []
                for pred in predictions:
                    if pred is not None and len(pred) > i:
                        image_preds.append(pred[i])
                
                if image_preds:
                    # 모든 레이어의 예측을 결합
                    combined_preds = torch.cat(image_preds, dim=0)
                    detections = self._filter_detections(combined_preds.cpu().numpy())
                else:
                    detections = []
            else:
                # 단일 텐서 출력
                if predictions is not None and len(predictions) > i:
                    pred = predictions[i]
                    detections = self._filter_detections(pred.cpu().numpy())
                else:
                    detections = []
            
            # 좌표 스케일링
            if len(detections) > 0:
                detections = self._scale_detections(detections, images.shape[2:])
            
            # 결과 구성
            result = {
                'batch_idx': batch_idx,
                'image_idx': i,
                'detections': detections,
                'image_size': images.shape[2:],
                'inference_time': datetime.now().isoformat()
            }
            
            # 타겟이 있는 경우 추가
            if targets is not None:
                if isinstance(targets, list):
                    result['target'] = targets[i] if i < len(targets) else None
                else:
                    result['target'] = targets[i] if i < len(targets) else None
            
            batch_results.append(result)
        
        return batch_results
    
    def _save_image_result(self, image, detections, image_name, class_names):
        """이미지 추론 결과 저장"""
        # 결과 이미지 생성
        result_image = draw_detections(image, detections, class_names)
        
        # 이미지 저장
        image_path = self.results_dir / f"pred_{image_name}"
        result_image.save(image_path)
        
        # JSON 결과 저장
        json_path = self.results_dir / f"pred_{Path(image_name).stem}.json"
        result_data = {
            'image_name': image_name,
            'detections': detections.tolist() if len(detections) > 0 else [],
            'image_size': image.size,
            'inference_time': datetime.now().isoformat()
        }
        
        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    def _draw_detections_on_frame(self, frame, detections, class_names):
        """프레임에 검출 결과 그리기"""
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls_id = detection
            
            # 박스 그리기
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 라벨 그리기
            class_name = class_names[int(cls_id)] if class_names and int(cls_id) < len(class_names) else f"Class_{int(cls_id)}"
            label = f"{class_name}: {conf:.2f}"
            
            # 라벨 배경
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 10), 
                         (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
            
            # 라벨 텍스트
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated_frame
    
    def _save_dataset_results(self, results):
        """데이터셋 추론 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 전체 결과 저장
        results_file = self.results_dir / f"dataset_predictions_{timestamp}.json"
        
        # 상세 결과는 제외하고 저장
        save_results = []
        for result in results:
            save_result = result.copy()
            if 'target' in save_result:
                save_result['target'] = save_result['target'].tolist() if save_result['target'] is not None else None
            save_results.append(save_result)
        
        with open(results_file, 'w') as f:
            json.dump(save_results, f, indent=2)
        
        debug_log(f"Dataset prediction results saved: {results_file}")
    
    def _save_video_results(self, results, video_path):
        """비디오 추론 결과 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        
        # 비디오 결과 저장
        results_file = self.results_dir / f"video_predictions_{video_name}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        debug_log(f"Video prediction results saved: {results_file}")

def create_predictor(model, device='cpu', config=None):
    """
    예측기 생성 함수
    
    Args:
        model: 예측할 모델
        device: 예측 디바이스
        config: 예측 설정
        
    Returns:
        Predictor 인스턴스
    """
    return Predictor(model, device, config)

# 독립적인 함수로 export
def predict_batch(model, image_paths, device='cpu', config=None, 
                 save_results=True, class_names=None):
    """
    배치 이미지 예측 (독립 함수)
    
    Args:
        model: 예측할 모델
        image_paths: 이미지 경로 리스트
        device: 예측 디바이스
        config: 예측 설정
        save_results: 결과 저장 여부
        class_names: 클래스 이름 리스트
        
    Returns:
        예측 결과 리스트
    """
    predictor = Predictor(model, device, config)
    return predictor.predict_batch(image_paths, save_results, class_names) 