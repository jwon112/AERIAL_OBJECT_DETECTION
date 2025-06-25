import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import pickle
import shutil
import os
import yaml
import argparse
import glob

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from utils.utils import cvtColor, preprocess_input, resize_image, get_classes
from yolo import YOLO

#---------------------------------------------------------------------------#
#   map_mode用于指定该文件运行时计算的内容
#   map_mode为0代表整个map计算流程，包括获得预测结果、计算map。
#   map_mode为1代表仅仅获得预测结果。
#   map_mode为2代表仅仅获得计算map。
#---------------------------------------------------------------------------#
map_mode = 0

class mAP_YOLO(YOLO):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image_id, image, results, clsid2catid):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            outputs = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if outputs[0] is None: 
                return outputs

            top_label   = np.array(outputs[0][:, 5], dtype = 'int32')
            top_conf    = outputs[0][:, 4]
            top_boxes   = outputs[0][:, :4]

        for i, c in enumerate(top_label):
            result                      = {}
            top, left, bottom, right    = top_boxes[i]

            result["image_id"]      = int(image_id)
            result["category_id"]   = clsid2catid[c]
            result["bbox"]          = [float(left),float(top),float(right-left),float(bottom-top)]
            result["score"]         = float(top_conf[i])
            results.append(result)
        return results

def generate_ground_truth(data_yaml, map_out_path):
    """
    YOLO 형식의 라벨을 MSNet 평가 형식으로 변환하여 ground-truth 파일들을 생성
    """
    print("Generating ground truth files...")
    
    # data.yaml에서 정보 읽기
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get('names', [])
    val_path = data_config.get('val', '')
    
    # 절대 경로로 변환
    if not os.path.isabs(val_path):
        val_path = os.path.join(os.path.dirname(data_yaml), val_path)
    
    # ground-truth 디렉토리 생성
    ground_truth_path = os.path.join(map_out_path, 'ground-truth')
    if not os.path.exists(ground_truth_path):
        os.makedirs(ground_truth_path)
    
    # validation 이미지 목록 읽기
    if os.path.isfile(val_path):
        # val_path가 파일인 경우 (이미지 경로 목록)
        with open(val_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
    else:
        # val_path가 디렉토리인 경우
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(val_path, ext)))
    
    print(f"Found {len(image_paths)} validation images")
    
    # 각 이미지에 대해 ground truth 파일 생성
    for image_path in tqdm(image_paths, desc="Processing ground truth"):
        # 이미지 파일명에서 라벨 파일 경로 추정
        image_name = os.path.basename(image_path)
        image_id = os.path.splitext(image_name)[0]
        
        # 라벨 파일 경로 찾기
        label_path = None
        possible_label_dirs = [
            os.path.join(os.path.dirname(image_path), '..', 'labels'),
            os.path.join(os.path.dirname(image_path), 'labels'),
            os.path.dirname(image_path).replace('images', 'labels')
        ]
        
        for label_dir in possible_label_dirs:
            potential_label_path = os.path.join(label_dir, image_id + '.txt')
            if os.path.exists(potential_label_path):
                label_path = potential_label_path
                break
        
        if not label_path or not os.path.exists(label_path):
            # 라벨 파일이 없으면 빈 ground truth 파일 생성
            gt_file_path = os.path.join(ground_truth_path, image_id + '.txt')
            with open(gt_file_path, 'w') as f:
                pass  # 빈 파일 생성
            continue
        
        # 이미지 크기 읽기
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except:
            print(f"Warning: Could not read image {image_path}, using default size")
            img_width, img_height = 640, 640
        
        # YOLO 형식 라벨을 MSNet 형식으로 변환
        gt_file_path = os.path.join(ground_truth_path, image_id + '.txt')
        with open(gt_file_path, 'w') as gt_file:
            with open(label_path, 'r') as label_file:
                for line in label_file:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    
                    class_id, center_x, center_y, width, height = map(float, parts)
                    class_id = int(class_id)
                    
                    if class_id >= len(class_names):
                        continue
                    
                    # 정규화된 좌표를 절대 좌표로 변환
                    center_x *= img_width
                    center_y *= img_height
                    width *= img_width
                    height *= img_height
                    
                    # center 좌표를 corner 좌표로 변환
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2
                    
                    # MSNet 형식으로 저장: class_name x1 y1 x2 y2
                    class_name = class_names[class_id]
                    gt_file.write(f"{class_name} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
    
    print(f"Ground truth files generated in {ground_truth_path}")

def generate_detections(yolo_model, data_yaml, map_out_path):
    """
    학습된 YOLO 모델로 validation 이미지들에 대해 추론을 수행하여 detection-results 파일들을 생성
    """
    print("Generating detection results...")
    
    # data.yaml에서 정보 읽기
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get('names', [])
    val_path = data_config.get('val', '')
    
    # 절대 경로로 변환
    if not os.path.isabs(val_path):
        val_path = os.path.join(os.path.dirname(data_yaml), val_path)
    
    # validation 이미지 목록 읽기
    if os.path.isfile(val_path):
        # val_path가 파일인 경우 (이미지 경로 목록)
        with open(val_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
    else:
        # val_path가 디렉토리인 경우
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(val_path, ext)))
    
    print(f"Processing {len(image_paths)} validation images for detection")
    
    # 각 이미지에 대해 detection 수행
    for image_path in tqdm(image_paths, desc="Processing detections"):
        image_name = os.path.basename(image_path)
        image_id = os.path.splitext(image_name)[0]
        
        try:
            # 이미지 로드
            image = Image.open(image_path)
            
            # YOLO의 get_map_txt 메서드를 사용하여 detection 결과 생성
            yolo_model.get_map_txt(image_id, image, class_names, map_out_path)
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            # 에러가 발생해도 빈 detection 파일 생성
            detection_results_path = os.path.join(map_out_path, 'detection-results')
            if not os.path.exists(detection_results_path):
                os.makedirs(detection_results_path)
            det_file_path = os.path.join(detection_results_path, image_id + '.txt')
            with open(det_file_path, 'w') as det_file:
                pass
    
    print(f"Detection results generated in {os.path.join(map_out_path, 'detection-results')}")

#------------------------------------------
def get_coco_map(class_names, path, data_yaml):
    #---------------------------------------------------#
    #   map_out에 있는 detection-results와 ground-truth를 이용하여 mAP를 계산
    #---------------------------------------------------#
    
    # 데이터 YAML 파일에서 val 이미지 목록 읽기
    image_paths = []
    try:
        with open(data_yaml, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        val_path = data_config.get('val', '')
        if not os.path.isabs(val_path):
            val_path = os.path.join(os.path.dirname(data_yaml), val_path)
        with open(val_path, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error reading validation image paths from {data_yaml}: {e}")

    # 이미지 ID를 키로, 실제 크기를 값으로 하는 딕셔너리 생성
    image_dims = {}
    if image_paths:
        print("Reading image dimensions...")
        for image_path in tqdm(image_paths, desc="Reading image sizes"):
            try:
                image_id_str = os.path.splitext(os.path.basename(image_path))[0]
                image_id = int(image_id_str)
                with Image.open(image_path) as img:
                    width, height = img.size
                    image_dims[image_id] = (width, height)
                    # 실제 이미지 크기 확인용 로그 추가
                    if len(image_dims) < 5: # 처음 5개 이미지만 로그 출력
                        print(f"[DEBUG] Image ID {image_id}: 실제 크기 {width}x{height} 읽음")
            except Exception as e:
                print(f"Warning: Could not process image {image_path}: {e}")

    MINOVERLAP = 0.5
    #-------------------------------------------------------#
    #   ground-truth와 detection-results의 경로
    #-------------------------------------------------------#
    ground_truth_path = os.path.join(path, 'ground-truth')
    if not os.path.exists(ground_truth_path):
        os.makedirs(ground_truth_path)
            
    detection_results_path = os.path.join(path, 'detection-results')
    if not os.path.exists(detection_results_path):
        os.makedirs(detection_results_path)
    
    # ground-truth와 detection-results 파일이 있는지 확인
    gt_files = [f for f in os.listdir(ground_truth_path) if f.endswith('.txt')]
    dt_files = [f for f in os.listdir(detection_results_path) if f.endswith('.txt')]
    
    if not gt_files:
        print("Warning: No ground truth files found!")
        return {
            'mAP': 0.0,
            'AP50': 0.0,
            'AP75': 0.0,
            'APs': 0.0,
            'APm': 0.0,
            'APl': 0.0
        }
    
    if not dt_files:
        print("Warning: No detection result files found!")
        return {
            'mAP': 0.0,
            'AP50': 0.0,
            'AP75': 0.0,
            'APs': 0.0,
            'APm': 0.0,
            'APl': 0.0
        }
            
    #-------------------------------------------------------#
    #   COCO 형식으로 변환
    #-------------------------------------------------------#
    # Ground Truth JSON 생성
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # 카테고리 정보 추가
    for i, class_name in enumerate(class_names, 1):
        coco_gt["categories"].append({
            "id": i,
            "name": class_name,
            "supercategory": "none"
        })
    
    # Ground Truth 이미지 및 어노테이션 추가
    annotation_id = 1
    for gt_file in os.listdir(ground_truth_path):
        if not gt_file.endswith('.txt'):
            continue
            
        image_name = gt_file.replace('.txt', '.jpg')
        image_id = int(gt_file.replace('.txt', ''))  # 파일명을 숫자로 변환 (001.txt -> 1)
        
        # 이미지 정보 추가 (수정된 부분)
        width, height = image_dims.get(image_id, (640, 640)) # 실제 크기 사용, 없으면 기본값
        
        # JSON에 기록될 크기 확인용 로그 추가
        if image_id in list(image_dims.keys())[:5]: # 처음 5개 ID에 대해서만 로그 출력
            print(f"[DEBUG] Image ID {image_id}: JSON에 {width}x{height} 크기로 기록 예정")
            
        coco_gt["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })
        
        # 어노테이션 정보 추가
        with open(os.path.join(ground_truth_path, gt_file), 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_name = " ".join(parts[:-4])
            x1, y1, x2, y2 = parts[-4:]
            
            if class_name not in class_names:
                continue
                
            category_id = class_names.index(class_name) + 1
            
            coco_gt["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(x2)-float(x1), float(y2)-float(y1)],
                "area": (float(x2)-float(x1)) * (float(y2)-float(y1)),
                "iscrowd": 0
            })
            annotation_id += 1
    
    # COCO 형식의 ground truth 저장
    coco_gt_path = os.path.join(path, 'coco_gt.json')
    with open(coco_gt_path, 'w') as f:
        json.dump(coco_gt, f)
    
    # Detection Results JSON 생성
    coco_dt = []
    for dt_file in os.listdir(detection_results_path):
        if not dt_file.endswith('.txt'):
            continue
            
        image_id = int(dt_file.replace('.txt', ''))  # 파일명을 숫자로 변환 (001.txt -> 1)
        
        # 이미지 크기 가져오기
        img_width, img_height = image_dims.get(image_id, (640, 640))
        
        with open(os.path.join(detection_results_path, dt_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:  # class_name, confidence, left, top, right, bottom
                    continue
                    
                class_name = " ".join(parts[:-5])  # 클래스명이 여러 단어일 수 있음
                if class_name not in class_names:
                    continue
                    
                confidence, left, top, right, bottom = map(float, parts[-5:])
                category_id = class_names.index(class_name) + 1
                
                # confidence 값을 0~1 범위로 정규화 (sigmoid 적용)
                if confidence > 1.0:
                    confidence = 1.0 / (1.0 + np.exp(-confidence))
                
                # MSNet 형식은 이미 절대 좌표이므로 COCO 형식으로 변환만 하면 됨
                x = left
                y = top
                w = right - left
                h = bottom - top
                
                # 좌표가 이미지 범위를 벗어나지 않도록 클리핑
                x = max(0, min(x, img_width - 1))
                y = max(0, min(y, img_height - 1))
                w = max(1, min(w, img_width - x))
                h = max(1, min(h, img_height - y))
                
                # confidence가 너무 낮으면 제외
                if confidence < 0.3:
                    continue
                
                coco_dt.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "score": confidence
                })
    
    # COCO 형식의 detection results 저장
    coco_dt_path = os.path.join(path, 'coco_dt.json')
    with open(coco_dt_path, 'w') as f:
        json.dump(coco_dt, f)
    
    # detection results가 비어있는지 확인
    if not coco_dt:
        print("Warning: No detections found in any image!")
        return {
            'mAP': 0.0,
            'AP50': 0.0,
            'AP75': 0.0,
            'APs': 0.0,
            'APm': 0.0,
            'APl': 0.0
        }
    
    # COCO 평가 수행
    try:
        cocoGt = COCO(coco_gt_path)
        cocoDt = cocoGt.loadRes(coco_dt_path)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        
        # 결과 저장
        results = {
            'mAP': cocoEval.stats[0],
            'AP50': cocoEval.stats[1],
            'AP75': cocoEval.stats[2],
            'APs': cocoEval.stats[3],
            'APm': cocoEval.stats[4],
            'APl': cocoEval.stats[5]
        }
        
        # 결과를 텍스트 파일로 저장
        results_path = os.path.join(path, 'results.txt')
        with open(results_path, 'w') as f:
            f.write(f"mAP: {results['mAP']:.4f}\n")
            f.write(f"AP50: {results['AP50']:.4f}\n")
            f.write(f"AP75: {results['AP75']:.4f}\n")
            f.write(f"APs: {results['APs']:.4f}\n")
            f.write(f"APm: {results['APm']:.4f}\n")
            f.write(f"APl: {results['APl']:.4f}\n")
        
        return results
            
    except Exception as e:
        print(f"Error during COCO evaluation: {e}")
        return {
            'mAP': 0.0,
            'AP50': 0.0,
            'AP75': 0.0,
            'APs': 0.0,
            'APm': 0.0,
            'APl': 0.0
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='path to model weights')
    parser.add_argument('--data_yaml', type=str, required=True, help='path to data.yaml file')
    parser.add_argument('--map_out_path', type=str, default='map_out', help='path to save evaluation results')
    parser.add_argument('--confidence', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--nms_iou', type=float, default=0.3, help='nms iou threshold')
    parser.add_argument('--input_shape', type=int, nargs='+', default=[640, 640], help='input shape')
    parser.add_argument('--phi', type=str, default='l', help='model size')
    parser.add_argument('--cuda', type=bool, default=True, help='use cuda')
    parser.add_argument('--map_mode', type=int, default=0, help='0: full evaluation, 1: generate predictions only, 2: calculate mAP only')
    opt = parser.parse_args()

    # data.yaml에서 클래스 정보 읽기
    with open(opt.data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config.get('names', [])

    # map_out_path 디렉토리 생성 (절대 경로 변환 제거)
    if not os.path.exists(opt.map_out_path):
        os.makedirs(opt.map_out_path)

    if opt.map_mode == 0 or opt.map_mode == 1:
        # Ground truth 생성
        generate_ground_truth(opt.data_yaml, opt.map_out_path)
        
        # YOLO 객체 생성
        yolo = YOLO(
            model_path=opt.model_path,
            data_yaml=opt.data_yaml,
            input_shape=opt.input_shape,
            phi=opt.phi,
            confidence=opt.confidence,
            nms_iou=opt.nms_iou,
            cuda=opt.cuda
        )
        
        # Detection 결과 생성
        generate_detections(yolo, opt.data_yaml, opt.map_out_path)

    if opt.map_mode == 0 or opt.map_mode == 2:
        # COCO 평가 수행
        results = get_coco_map(class_names, opt.map_out_path, opt.data_yaml)
        print("Evaluation Results:")
        for key, value in results.items():
            print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
