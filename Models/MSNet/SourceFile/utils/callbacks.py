import datetime
import os

import torch
import matplotlib
matplotlib.use('Agg')
import scipy.signal
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from .utils import cvtColor, preprocess_input, resize_image
from .utils_bbox import DecodeBox
from .utils_map import get_coco_map, get_map


class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        # try:
        #     dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, dummy_input)
        # except:
        #     pass

    def append_loss(self, epoch, loss, val_loss):
        '''
        Apeend Loss
        Args:
            epoch: 循环次数
            loss: 训练集损失
            val_loss: 验证集损失

        Returns: None

        '''
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        # try:
        #     if len(self.losses) < 25:
        #         num = 5
        #     else:
        #         num = 15
            
        #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=10):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.bbox_util          = DecodeBox(self.num_classes, (self.input_shape[0], self.input_shape[1]))
        
        self.maps       = [0]
        self.epoches    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
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
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
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
            print(f"[DEBUG] Raw outputs type: {type(outputs)}, len: {len(outputs) if hasattr(outputs, '__len__') else 'N/A'}")
            outputs = self.bbox_util.decode_box(outputs)
            print(f"[DEBUG] Decoded outputs shape: {outputs.shape}")
            print(f"[DEBUG] Outputs max confidence: {outputs[..., 4:].max():.6f}")
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
            print(f"[DEBUG] After NMS: {len(results[0]) if results[0] is not None else 0} predictions")
                                                    
            if results[0] is None: 
                print(f"[DEBUG] No predictions for image {image_id} (after NMS)")
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        print(f"[DEBUG] Image {image_id}: Found {len(top_label)} predictions")
        print(f"[DEBUG] Available class_names: {class_names}")
        print(f"[DEBUG] Model class_names: {self.class_names}")
        valid_predictions = 0
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                print(f"[DEBUG] Skipping class '{predicted_class}' (not in class_names)")
                continue
            valid_predictions += 1

            line_to_write = "%s %s %s %s %s %s\n" % (predicted_class, str(int(left)), str(int(top)), str(int(right)), str(int(bottom)), score[:6])
            f.write(line_to_write)
            print(f"[DEBUG] Writing: {line_to_write.strip()}")

        f.close()
        print(f"[DEBUG] Image {image_id}: Written {valid_predictions} valid predictions")
        return 
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            # 디버그: 모델 상태 확인
            self.net.eval()  # 반드시 평가 모드로
            print(f"[DEBUG] Model mode: training={self.net.training}")
            print(f"[DEBUG] Model device: {next(self.net.parameters()).device}")
            print(f"[DEBUG] Model weights mean: {next(self.net.parameters()).data.mean():.6f}")
            if not os.path.exists(self.map_out_path):
                os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            print("Get map.")
            for annotation_line in tqdm(self.val_lines):
                line        = annotation_line.split()
                image_path  = line[0].strip()
                image_id    = os.path.basename(image_path).split('.')[0]
                #------------------------------#
                #   读取图像并转换成RGB图像
                #------------------------------#
                image       = Image.open(image_path)
                
                #------------------------------#
                #   데이터 형식 자동 감지 및 처리
                #------------------------------#
                gt_boxes = []
                
                # VOC 형식 감지: annotation_line에 bbox 정보가 포함된 경우
                if len(line) > 1:
                    print(f"[DEBUG] Detected VOC format for {image_id}")
                    # VOC 형식: image_path x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id ...
                    for box_info in line[1:]:
                        try:
                            coords = box_info.split(',')
                            if len(coords) >= 5:
                                x1, y1, x2, y2, class_id = map(int, coords[:5])
                                gt_boxes.append([x1, y1, x2, y2, class_id])
                        except ValueError:
                            print(f"[DEBUG] Invalid VOC box format: {box_info}")
                            continue
                
                # YOLO 형식: annotation_line에 image_path만 있는 경우
                else:
                    print(f"[DEBUG] Detected YOLO format for {image_id}")
                    # YOLO 형식: 별도 labels 디렉토리에서 label 파일 찾기
                    image_dir = os.path.dirname(image_path)
                    dataset_root = os.path.dirname(image_dir)  # images의 상위 디렉토리
                    label_path = os.path.join(dataset_root, "labels", image_id + ".txt")
                    
                    print(f"[DEBUG] Looking for YOLO label: {label_path}")
                    
                    if os.path.exists(label_path):
                        img_width, img_height = image.size  # PIL Image는 (width, height)
                        
                        with open(label_path, 'r') as f:
                            for line_content in f:
                                parts = line_content.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    # YOLO 형식: center_x, center_y, width, height (normalized)
                                    center_x, center_y, width, height = map(float, parts[1:5])
                                    
                                    # Normalized -> Absolute coordinates
                                    center_x *= img_width
                                    center_y *= img_height
                                    width *= img_width
                                    height *= img_height
                                    
                                    # Center format -> Corner format (x1,y1,x2,y2)
                                    x1 = int(center_x - width / 2)
                                    y1 = int(center_y - height / 2)
                                    x2 = int(center_x + width / 2)
                                    y2 = int(center_y + height / 2)
                                    
                                    gt_boxes.append([x1, y1, x2, y2, class_id])
                    else:
                        print(f"[DEBUG] YOLO label file not found: {label_path}")
                
                gt_boxes = np.array(gt_boxes) if gt_boxes else np.array([]).reshape(0, 5)
                print(f"[DEBUG] Found {len(gt_boxes)} ground truth boxes for {image_id}")
                
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                
                #------------------------------#
                #   获得真实框txt - 통합된 VOC 형식으로 저장
                #------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        x1, y1, x2, y2, obj = box
                        if 0 <= obj < len(self.class_names):  # 유효한 클래스 ID 확인
                            obj_name = self.class_names[obj]
                            new_f.write("%s %s %s %s %s\n" % (obj_name, x1, y1, x2, y2))
                            print(f"[DEBUG] GT written for {image_id}: {obj_name} {x1} {y1} {x2} {y2}")
                        else:
                            print(f"[DEBUG] Invalid class_id {obj} for image {image_id} (max: {len(self.class_names)-1})")
                
            print("Calculate Map.")
            try:
                temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path)[1]
            except:
                temp_map = get_map(self.MINOVERLAP, False, score_threhold=0.1, path = self.map_out_path)
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)
