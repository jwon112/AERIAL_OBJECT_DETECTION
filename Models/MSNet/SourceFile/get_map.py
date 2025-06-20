import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import sys
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
import json
import tempfile

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

def get_classes_from_yaml(yaml_path):
    """data.yaml에서 클래스 정보를 읽어옵니다."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        return data_config.get('names', [])
    except Exception as e:
        print(f"[ERROR] Failed to read classes from {yaml_path}: {e}")
        return []

def get_dataset_root(annotation_path):
    """Get the correct dataset root path from annotation path."""
    # annotation_path 예시: .../NWPU_VHR10_YOLO/test_iter_01.txt
    dataset_dir = os.path.dirname(annotation_path)  # .../NWPU_VHR10_YOLO
    return dataset_dir  # 중복 경로 제거

def get_image_path(image_id, dataset_root):
    """Get the correct image path from image ID."""
    # 이미지 ID에서 파일 이름 추출 (예: 183 -> 183.png)
    image_name = f"{image_id}.png"
    return os.path.join(dataset_root, 'images', image_name)

def get_label_path(image_id, dataset_root):
    """Get the correct label path from image ID."""
    # 이미지 ID에서 라벨 파일 이름 추출 (예: 183 -> 183.txt)
    label_name = f"{image_id}.txt"
    return os.path.join(dataset_root, 'labels', label_name)

def main():
    '''
    Recall和Precision不像AP是一个面积的概念，因此在门限值（Confidence）不同时，网络的Recall和Precision值是不同的。
    默认情况下，本代码计算的Recall和Precision代表的是当门限值（Confidence）为0.5时，所对应的Recall和Precision值。

    受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算不同门限条件下的Recall和Precision值
    因此，本代码获得的map_out/detection-results/里面的txt的框的数量一般会比直接predict多一些，目的是列出所有可能的预测框，
    '''
    # Add argument parsing
    parser = argparse.ArgumentParser(description='MSNet mAP calculation')
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--input_shape', type=str, default='[640, 640]', help='Model input shape')
    parser.add_argument('--phi', type=str, default='l', help='Model phi value')
    parser.add_argument('--val_annotation_path', type=str, help='Path to validation annotation file')
    parser.add_argument('--classes_path', type=str, required=True, help='Path to classes.txt file')
    args = parser.parse_args()
    
    print(f"[DEBUG] Using data_yaml: {args.data_yaml}")
    
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表整个map计算流程，包括获得预测结果、获得真实框、计算VOC_map。
    #   map_mode为1代表仅仅获得预测结果。
    #   map_mode为2代表仅仅获得真实框。
    #   map_mode为3代表仅仅计算VOC_map。
    #   map_mode为4代表利用COCO工具箱计算当前数据集的0.50:0.95map。需要获得预测结果、获得真实框后并安装pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 4
    #--------------------------------------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #--------------------------------------------------------------------------------------#
    classes_path    = args.classes_path
    
    #--------------------------------------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x。
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75。
    #
    #   当某一预测框与真实框重合度大于MINOVERLAP时，该预测框被认为是正样本，否则为负样본。
    #   因此MINOVERLAP的值越大，预测框要预测的越准确才能被认为是正样本，此时算出来的mAP值越低，
    #   设置为0.1进行初步调试
    #--------------------------------------------------------------------------------------#
    MINOVERLAP      = 0.1
    #--------------------------------------------------------------------------------------#
    #   受到mAP计算原理的限制，网络在计算mAP时需要获得近乎所有的预测框，这样才可以计算mAP
    #   因此，confidence的值应当设置的尽量小进而获得全部可能的预测框。
    #   
    #   该值一般不调整。因为计算mAP需要获得近乎所有的预测框，此处的confidence不能随便更改。
    #   想要获得不同门限值下的Recall和Precision值，请修改下方的score_threhold。
    #--------------------------------------------------------------------------------------#
    confidence      = 0.001
    #--------------------------------------------------------------------------------------#
    #   预测时使用到的非极大抑制值的大小，越大表示非极大抑制越不严格。
    #   
    #   该值一般不调整。
    #--------------------------------------------------------------------------------------#
    nms_iou         = 0.5
    #---------------------------------------------------------------------------------------------------------------#
    #   Recall和Precision不像AP是一个面积的概念，因此在门限值不同时，网络的Recall和Precision值是不同的。
    #   
    #   默认情况下，본代码计算的Recall和Precision代表的는当门限值为0.1（此处定义为score_threhold）时所对应的Recall和Precision값。
    #   因为计算mAP需要获得近乎所有的预测框，上面定义的confidence不能随便更改。
    #   这里专门定义一个score_threhold用于代表门限值，进而在计算mAP时找到门限值对应的Recall和Precision값。
    #---------------------------------------------------------------------------------------------------------------#
    score_threhold  = 0.1
    #-------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    # 请修改路径为自己的数据集路径
    VOCdevkit_path  = '../../Datasets/NWPU/VOCdevkit'
    
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为map_out
    #-------------------------------------------------------#
    map_out_path    = 'map_out'

    # 데이터셋 루트 경로 설정
    dataset_root = os.path.dirname(args.val_annotation_path)
    print(f"[DEBUG] Using dataset root: {dataset_root}")
    
    # 이미지와 라벨 디렉토리 확인
    image_dir = os.path.join(dataset_root, 'images')
    label_dir = os.path.join(dataset_root, 'labels')
    
    print(f"[DEBUG] Image directory: {image_dir}")
    print(f"[DEBUG] Label directory: {label_dir}")
    
    if not os.path.exists(image_dir):
        print(f"[ERROR] Image directory not found: {image_dir}")
        return
    if not os.path.exists(label_dir):
        print(f"[ERROR] Label directory not found: {label_dir}")
        return
        
    print(f"[DEBUG] Using image directory: {image_dir}")
    print(f"[DEBUG] Using label directory: {label_dir}")
    
    # 클래스 이름 로드
    class_names = []
    with open(args.classes_path, 'r') as f:
        for line in f:
            class_names.append(line.strip())
    print(f"[DEBUG] Loaded {len(class_names)} classes: {class_names}")
    
    # YOLO 모델 초기화
    print("Load model.")
    yolo = YOLO(
        model_path=args.model_path,
        classes_path=args.classes_path,
        confidence=confidence,
        nms_iou=nms_iou,
        input_shape=[int(x) for x in args.input_shape.strip('[]').split(',')] if args.input_shape else [640, 640],
        phi=args.phi if args.phi else 'l'
    )
    print("Load model done.")

    # Use YOLO format validation data if provided, otherwise use VOC format
    if args.val_annotation_path:
        print(f"[DEBUG] Using YOLO format validation data: {args.val_annotation_path}")
        
        # 이미지 ID 목록 읽기
        with open(args.val_annotation_path, 'r') as f:
            image_ids = []
            for line in f:
                # 이미지 경로에서 ID 추출 (예: Datasets\NWPU_VHR10_YOLO\images\183.png -> 183)
                image_id = os.path.splitext(os.path.basename(line.strip()))[0]
                image_ids.append(image_id)
        
        print(f"[DEBUG] Found {len(image_ids)} images in validation set")
        print(f"[DEBUG] First few image IDs: {image_ids[:5]}")
        
        # ground truth와 detection results 파일 생성 확인
        gt_path = os.path.join(map_out_path, "ground-truth")
        dr_path = os.path.join(map_out_path, "detection-results")
        
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)
        if not os.path.exists(dr_path):
            os.makedirs(dr_path)
        
        print(f"[DEBUG] Ground truth path: {gt_path}")
        print(f"[DEBUG] Detection results path: {dr_path}")
        
        # 각 이미지에 대해 처리
        for image_id in tqdm(image_ids, desc="Processing images"):
            # 이미지 파일 경로
            image_path = os.path.join(image_dir, f"{image_id}.png")
            if not os.path.exists(image_path):
                print(f"[DEBUG] Image not found: {image_path}")
                continue
                
            # 라벨 파일 경로
            label_path = os.path.join(label_dir, f"{image_id}.txt")
            print(f"[DEBUG] Checking label path: {label_path}")
            if not os.path.exists(label_path):
                print(f"[DEBUG] Label not found: {label_path}")
                continue
            
            # 이미지 로드
            image = Image.open(image_path)
            
            # ground truth 처리
            gt_file = os.path.join(gt_path, f"{image_id}.txt")
            with open(gt_file, "w") as f:
                with open(label_path, 'r') as label_file:
                    lines = label_file.readlines()
                    for line in lines:
                        # YOLO 형식: class_id x_center y_center width height
                        # VOC 형식: class_name x1 y1 x2 y2
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        class_name = class_names[int(class_id)]
                        
                        # YOLO -> VOC 변환
                        x1 = (x_center - width/2) * image.size[0]
                        y1 = (y_center - height/2) * image.size[1]
                        x2 = (x_center + width/2) * image.size[0]
                        y2 = (y_center + height/2) * image.size[1]
                        
                        f.write(f"{class_name} {x1} {y1} {x2} {y2}\n")
            
            print(f"[DEBUG] Created ground truth file: {gt_file}")
            
            # 예측 수행
            dr_file = os.path.join(dr_path, f"{image_id}.txt")
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
            print(f"[DEBUG] Created detection results file: {dr_file}")

    # 파일 생성 확인
    gt_files = os.listdir(gt_path)
    dr_files = os.listdir(dr_path)
    print(f"[DEBUG] Number of ground truth files: {len(gt_files)}")
    print(f"[DEBUG] Number of detection results files: {len(dr_files)}")
    
    if len(gt_files) == 0 or len(dr_files) == 0:
        print("[ERROR] No ground truth or detection results files were created!")
        return
    
    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        # VOC 형식 mAP 계산
        get_map(MINOVERLAP, True, score_threhold=0.001, path=map_out_path)
        
        # COCO 형식 mAP 계산
        print("\nCalculating COCO format mAP...")
        
        # COCO 형식으로 변환
        coco_gt = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # 이미지 정보 추가
        for image_id in image_ids:
            # 이미지 정보
            image_path = os.path.join(image_dir, f"{image_id}.png")
            if not os.path.exists(image_path):
                continue
            image = Image.open(image_path)
            coco_gt["images"].append({
                "id": int(image_id),
                "width": image.size[0],
                "height": image.size[1],
                "file_name": f"{image_id}.png"
            })
            
            # annotation 정보
            label_path = os.path.join(label_dir, f"{image_id}.txt")
            if not os.path.exists(label_path):
                continue
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                class_name = class_names[int(class_id)]
                
                # YOLO -> VOC 변환
                x1 = (x_center - width/2) * image.size[0]
                y1 = (y_center - height/2) * image.size[1]
                x2 = (x_center + width/2) * image.size[0]
                y2 = (y_center + height/2) * image.size[1]
                
                # COCO 형식 annotation 추가
                coco_gt["annotations"].append({
                    "id": len(coco_gt["annotations"]) + 1,
                    "image_id": int(image_id),
                    "category_id": int(class_id) + 1,
                    "bbox": [x1, y1, x2-x1, y2-y1],
                    "area": (x2-x1) * (y2-y1),
                    "iscrowd": 0
                })
        
        # 카테고리 정보 추가
        for i, class_name in enumerate(class_names):
            coco_gt["categories"].append({
                "id": i + 1,
                "name": class_name,
                "supercategory": "none"
            })
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_gt, f)
            temp_file_path = f.name
        
        try:
            # COCO 객체 생성
            cocoGt = COCO(temp_file_path)
            
            # detection results를 COCO 형식으로 변환
            coco_dt = []
            for image_id in image_ids:
                detection_path = os.path.join(map_out_path, "detection-results", f"{image_id}.txt")
                if not os.path.exists(detection_path):
                    continue
                
                with open(detection_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) != 6:
                        continue
                    
                    class_name, score, x1, y1, x2, y2 = parts
                    class_id = class_names.index(class_name)
                    
                    coco_dt.append({
                        "image_id": int(image_id),
                        "category_id": class_id + 1,
                        "bbox": [float(x1), float(y1), float(x2)-float(x1), float(y2)-float(y1)],
                        "score": float(score)
                    })
            
            # COCO 평가
            cocoDt = cocoGt.loadRes(coco_dt)
            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            
            # 결과 출력
            print("\nCOCO format mAP results:")
            print(f"mAP@0.5:0.95 = {cocoEval.stats[0]:.4f}")
            print(f"mAP@0.5 = {cocoEval.stats[1]:.4f}")
            print(f"mAP@0.75 = {cocoEval.stats[2]:.4f}")
            print(f"mAP@small = {cocoEval.stats[3]:.4f}")
            print(f"mAP@medium = {cocoEval.stats[4]:.4f}")
            print(f"mAP@large = {cocoEval.stats[5]:.4f}")
            print(f"AR@1 = {cocoEval.stats[6]:.4f}")
            print(f"AR@10 = {cocoEval.stats[7]:.4f}")
            print(f"AR@100 = {cocoEval.stats[8]:.4f}")
            print(f"AR@small = {cocoEval.stats[9]:.4f}")
            print(f"AR@medium = {cocoEval.stats[10]:.4f}")
            print(f"AR@large = {cocoEval.stats[11]:.4f}")
        
        finally:
            # 임시 파일 삭제
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")

    print("Get map done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSNet mAP calculation')
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--input_shape', type=str, default='[640, 640]', help='Model input shape')
    parser.add_argument('--phi', type=str, default='l', help='Model phi value')
    parser.add_argument('--val_annotation_path', type=str, help='Path to validation annotation file')
    parser.add_argument('--classes_path', type=str, required=True, help='Path to classes.txt file')
    args = parser.parse_args()
    
    main()
