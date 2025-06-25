import numpy as np
import torch
import os
from torchvision.ops import nms
import pkg_resources as pkg

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def check_version(current: str = "0.0.0",
                  minimum: str = "0.0.0",
                  name: str = "version ",
                  pinned: bool = False) -> bool:
	current, minimum = (pkg.parse_version(x) for x in (current, minimum))
	result = (current == minimum) if pinned else (current >= minimum)  # bool
	return result


TORCH_1_10 = check_version(torch.__version__, '1.10.0')


def make_anchors(feats, strides, grid_cell_offset=0.5):
	"""Generate anchors from features."""
	anchor_points, stride_tensor = [], []
	assert feats is not None
	dtype, device = feats[0].dtype, feats[0].device
	for i, stride in enumerate(strides):
		_, _, h, w = feats[i].shape
		sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
		sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
		# sy, sx      = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
		# TODO --> Pytorch 1.9.0 以上 indexing='ij' 参数已删除
		sy, sx = torch.meshgrid(sy, sx) if TORCH_1_10 else torch.meshgrid(sy, sx)
		anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
		stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
	return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1, img_size=None):
	"""Transform distance(ltrb) to box(xywh or xyxy)."""
	# 좌상우하
	lt, rb = torch.split(distance, 2, dim)
	
	# DEBUG: distance 예측값 상세 확인
	print(f"[DEBUG dist2bbox] Input distance shape: {distance.shape}")
	print(f"[DEBUG dist2bbox] lt range: [{lt.min():.3f}, {lt.max():.3f}]")
	print(f"[DEBUG dist2bbox] rb range: [{rb.min():.3f}, {rb.max():.3f}]")
	print(f"[DEBUG dist2bbox] anchor_points range: [{anchor_points.min():.3f}, {anchor_points.max():.3f}]")
	
	x1y1 = anchor_points - lt
	x2y2 = anchor_points + rb
	
	print(f"[DEBUG dist2bbox] x1y1 range: [{x1y1.min():.3f}, {x1y1.max():.3f}]")
	print(f"[DEBUG dist2bbox] x2y2 range: [{x2y2.min():.3f}, {x2y2.max():.3f}]")
	
	# DEBUG: 음수 좌표 발생 확인
	if (x1y1 < 0).any() or (x2y2 < 0).any():
		neg_count_x1y1 = (x1y1 < 0).sum()
		neg_count_x2y2 = (x2y2 < 0).sum()
		print(f"[DEBUG dist2bbox] NEGATIVE DETECTED! x1y1: {neg_count_x1y1}, x2y2: {neg_count_x2y2}")
		
		# 가장 작은 값들 샘플링
		print(f"[DEBUG dist2bbox] Min x1y1: {x1y1.min():.3f}, Min x2y2: {x2y2.min():.3f}")
		print(f"[DEBUG dist2bbox] Sample lt values: {lt.flatten()[:5]}")
		print(f"[DEBUG dist2bbox] Sample anchor values: {anchor_points.flatten()[:5]}")
	
	# 좌표 클리핑 추가
	if img_size is not None:
		# img_size가 텐서인지 스칼라인지 확인
		if isinstance(img_size, torch.Tensor):
			# 텐서인 경우 max 값 사용
			max_size = img_size.max().item()
			x1y1 = torch.clamp(x1y1, min=0, max=max_size)
			x2y2 = torch.clamp(x2y2, min=0, max=max_size)
		else:
			# 스칼라인 경우
			x1y1 = torch.clamp(x1y1, min=0, max=img_size)
			x2y2 = torch.clamp(x2y2, min=0, max=img_size)
	else:
		# 기본적으로 0 이상으로 클리핑
		x1y1 = torch.clamp(x1y1, min=0)
		x2y2 = torch.clamp(x2y2, min=0)
	
	if xywh:
		c_xy = (x1y1 + x2y2) / 2
		wh = x2y2 - x1y1
		return torch.cat((c_xy, wh), dim)  # xywh bbox
	return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class DecodeBox():
	def __init__(self, num_classes, input_shape):
		super(DecodeBox, self).__init__()
		self.num_classes = num_classes
		self.bbox_attrs = 4 + num_classes
		self.input_shape = input_shape
	
	def decode_box(self, inputs):
		# dbox  batch_size, 4, 8400
		# cls   batch_size, 20, 8400
		dbox, cls, origin_cls, anchors, strides = inputs
		
		# 디버깅: 입력값 확인
		print(f"[DEBUG] Input shapes - dbox: {dbox.shape}, cls: {cls.shape}, anchors: {anchors.shape}")
		print(f"[DEBUG] dbox range - min: {dbox.min().item():.4f}, max: {dbox.max().item():.4f}")
		
		# 获得中心宽高坐标
		dbox = dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1, img_size=max(self.input_shape)) * strides
		
		# 디버깅: dist2bbox 후 확인
		print(f"[DEBUG] After dist2bbox - shape: {dbox.shape}")
		print(f"[DEBUG] dbox range - min: {dbox.min().item():.4f}, max: {dbox.max().item():.4f}")
		
		y = torch.cat((dbox, cls.sigmoid()), 1).permute(0, 2, 1)
		
		# 디버깅: cat 후 확인
		print(f"[DEBUG] After cat - shape: {y.shape}")
		print(f"[DEBUG] box coordinates range - min: {y[:, :, :4].min().item():.4f}, max: {y[:, :, :4].max().item():.4f}")
		
		# 进行归一化，到0~1之间
		y[:, :, :4] = y[:, :, :4] / torch.Tensor([self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]]).to(y.device)
		
		# 디버깅: 정규화 후 확인
		print(f"[DEBUG] After normalization - box coordinates range - min: {y[:, :, :4].min().item():.4f}, max: {y[:, :, :4].max().item():.4f}")
		
		return y
	
	def yolo_correct_boxes(self, box_xy, box_wh, input_shape, image_shape, letterbox_image):
		# -----------------------------------------------------------------#
		#   把y轴放前面是因为方便预测框和图像的宽高进行相乘
		# -----------------------------------------------------------------#
		box_yx = box_xy[..., ::-1]
		box_hw = box_wh[..., ::-1]
		input_shape = np.array(input_shape)
		image_shape = np.array(image_shape)
		
		if letterbox_image:
			# -----------------------------------------------------------------#
			#   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
			#   new_shape指的是宽高缩放情况
			# -----------------------------------------------------------------#
			new_shape = np.round(image_shape * np.min(input_shape / image_shape))
			offset = (input_shape - new_shape) / 2. / input_shape
			scale = input_shape / new_shape
			
			box_yx = (box_yx - offset) * scale
			box_hw *= scale
		
		box_mins = box_yx - (box_hw / 2.)
		box_maxes = box_yx + (box_hw / 2.)
		boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
		boxes *= np.concatenate([image_shape, image_shape], axis=-1)
		return boxes
	
	def non_max_suppression(self, prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.3, nms_thres=0.4):
		# ----------------------------------------------------------#
		#   将预测结果的格式转换成左上角右下角的格式。
		#   prediction  [batch_size, num_anchors, 85]
		# ----------------------------------------------------------#
		box_corner = prediction.new(prediction.shape)
		box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
		box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
		box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
		box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
		prediction[:, :, :4] = box_corner[:, :, :4]
		
		# DEBUG: 예측 결과 확인
		print(f"[DEBUG NMS] Input prediction shape: {prediction.shape}")
		print(f"[DEBUG NMS] Box coordinates range - min: {prediction[:, :, :4].min().item():.3f}, max: {prediction[:, :, :4].max().item():.3f}")
		
		output = [None for _ in range(len(prediction))]
		for i, image_pred in enumerate(prediction):
			# ----------------------------------------------------------#
			#   对种类预测部分取max。
			#   class_conf  [num_anchors, 1]    种类置信度
			#   class_pred  [num_anchors, 1]    种类
			# ----------------------------------------------------------#
			class_conf, class_pred = torch.max(image_pred[:, 4:4 + num_classes], 1, keepdim=True)
			
			# DEBUG: 클래스 예측 확인
			print(f"[DEBUG NMS] Class confidence range - min: {class_conf.min().item():.3f}, max: {class_conf.max().item():.3f}")
			
			# ----------------------------------------------------------#
			#   利用置信度进行第一轮筛选
			# ----------------------------------------------------------#
			conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
			
			# DEBUG: confidence threshold 필터링 결과
			print(f"[DEBUG NMS] Before confidence filter: {image_pred.size(0)} boxes")
			
			# ----------------------------------------------------------#
			#   根据置信度进行预测结果的筛选
			# ----------------------------------------------------------#
			image_pred = image_pred[conf_mask]
			class_conf = class_conf[conf_mask]
			class_pred = class_pred[conf_mask]
			
			# DEBUG: confidence threshold 필터링 후 결과
			print(f"[DEBUG NMS] After confidence filter: {image_pred.size(0)} boxes")
			
			if not image_pred.size(0):
				print("[DEBUG NMS] No boxes left after confidence filter")
				continue
				
			# -------------------------------------------------------------------------#
			#   detections  [num_anchors, 6]
			#   6的内容为：x1, y1, x2, y2, class_conf, class_pred
			# -------------------------------------------------------------------------#
			detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)
			
			# ------------------------------------------#
			#   获得预测结果中包含的所有种类
			# ------------------------------------------#
			unique_labels = detections[:, -1].cpu().unique()
			
			if prediction.is_cuda:
				unique_labels = unique_labels.cuda()
				detections = detections.cuda()
			
			# DEBUG: 클래스별 박스 수
			print(f"[DEBUG NMS] Number of unique classes: {len(unique_labels)}")
			
			for c in unique_labels:
				# ------------------------------------------#
				#   获得某一类得分筛选后全部的预测结果
				# ------------------------------------------#
				detections_class = detections[detections[:, -1] == c]
				
				# DEBUG: 클래스별 박스 수
				print(f"[DEBUG NMS] Class {c}: {detections_class.size(0)} boxes before NMS")
				
				# ------------------------------------------#
				#   使用官方自带的非极大抑制会速度更快一些！
				#   筛选出一定区域内，属于同一种类得分最大的框
				# ------------------------------------------#
				keep = nms(
					detections_class[:, :4],
					detections_class[:, 4],
					nms_thres
				)
				max_detections = detections_class[keep]
				
				# DEBUG: NMS 후 남은 박스 수
				print(f"[DEBUG NMS] Class {c}: {max_detections.size(0)} boxes after NMS")
				
				# Add max detections to outputs
				output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))
			
			if output[i] is not None:
				output[i] = output[i].cpu().numpy()
				box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
				output[i][:, :4] = self.yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
				
				# DEBUG: 최종 결과
				print(f"[DEBUG NMS] Final output: {output[i].shape[0]} boxes")
			else:
				print("[DEBUG NMS] No boxes in final output")
				
		return output


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	import numpy as np
	
	
	# ---------------------------------------------------#
	#   将预测值的每个特征层调成真实值
	# ---------------------------------------------------#
	def get_anchors_and_decode(input, input_shape, anchors, anchors_mask, num_classes):
		# -----------------------------------------------#
		#   input   batch_size, 3 * (4 + 1 + num_classes), 20, 20
		# -----------------------------------------------#
		batch_size = input.size(0)
		input_height = input.size(2)
		input_width = input.size(3)
		
		# -----------------------------------------------#
		#   输入为640x640时 input_shape = [640, 640]  input_height = 20, input_width = 20
		#   640 / 20 = 32
		#   stride_h = stride_w = 32
		# -----------------------------------------------#
		stride_h = input_shape[0] / input_height
		stride_w = input_shape[1] / input_width
		# -------------------------------------------------#
		#   此时获得的scaled_anchors大小是相对于特征层的
		#   anchor_width, anchor_height / stride_h, stride_w
		# -------------------------------------------------#
		scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in anchors[anchors_mask[2]]]
		
		# -----------------------------------------------#
		#   batch_size, 3 * (4 + 1 + num_classes), 20, 20 =>
		#   batch_size, 3, 5 + num_classes, 20, 20  =>
		#   batch_size, 3, 20, 20, 4 + 1 + num_classes
		# -----------------------------------------------#
		prediction = input.view(batch_size, len(anchors_mask[2]),
		                        num_classes + 5, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()
		
		# -----------------------------------------------#
		#   先验框的中心位置的调整参数
		# -----------------------------------------------#
		x = torch.sigmoid(prediction[..., 0])
		y = torch.sigmoid(prediction[..., 1])
		# -----------------------------------------------#
		#   先验框的宽高调整参数
		# -----------------------------------------------#
		w = torch.sigmoid(prediction[..., 2])
		h = torch.sigmoid(prediction[..., 3])
		# -----------------------------------------------#
		#   获得置信度，是否有物体 0 - 1
		# -----------------------------------------------#
		conf = torch.sigmoid(prediction[..., 4])
		# -----------------------------------------------#
		#   种类置信度 0 - 1
		# -----------------------------------------------#
		pred_cls = torch.sigmoid(prediction[..., 5:])
		
		FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
		LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
		
		# ----------------------------------------------------------#
		#   生成网格，先验框中心，网格左上角
		#   batch_size,3,20,20
		#   range(20)
		#   [
		#       [0, 1, 2, 3 ……, 19],
		#       [0, 1, 2, 3 ……, 19],
		#       …… （20次）
		#       [0, 1, 2, 3 ……, 19]
		#   ] * (batch_size * 3)
		#   [batch_size, 3, 20, 20]
		#
		#   [
		#       [0, 1, 2, 3 ……, 19],
		#       [0, 1, 2, 3 ……, 19],
		#       …… （20次）
		#       [0, 1, 2, 3 ……, 19]
		#   ].T * (batch_size * 3)
		#   [batch_size, 3, 20, 20]
		# ----------------------------------------------------------#
		grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
			batch_size * len(anchors_mask[2]), 1, 1).view(x.shape).type(FloatTensor)
		grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
			batch_size * len(anchors_mask[2]), 1, 1).view(y.shape).type(FloatTensor)
		
		# ----------------------------------------------------------#
		#   按照网格格式生成先验框的宽高
		#   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
		#   batch_size, 3, 20 * 20 => batch_size, 3, 20, 20
		# ----------------------------------------------------------#
		anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
		anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
		anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
		anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)
		
		# ----------------------------------------------------------#
		#   利用预测结果对先验框进行调整
		#   首先调整先验框的中心，从先验框中心向右下角偏移
		#   再调整先验框的宽高。
		#   x  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_x
		#   y  0 ~ 1 => 0 ~ 2 => -0.5 ~ 1.5 + grid_y
		#   w  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_w
		#   h  0 ~ 1 => 0 ~ 2 => 0 ~ 4 * anchor_h
		# ----------------------------------------------------------#
		pred_boxes = FloatTensor(prediction[..., :4].shape)
		pred_boxes[..., 0] = x.data * 2. - 0.5 + grid_x
		pred_boxes[..., 1] = y.data * 2. - 0.5 + grid_y
		pred_boxes[..., 2] = (w.data * 2) ** 2 * anchor_w
		pred_boxes[..., 3] = (h.data * 2) ** 2 * anchor_h
		
		point_h = 5
		point_w = 5
		
		box_xy = pred_boxes[..., 0:2].cpu().numpy() * 32
		box_wh = pred_boxes[..., 2:4].cpu().numpy() * 32
		grid_x = grid_x.cpu().numpy() * 32
		grid_y = grid_y.cpu().numpy() * 32
		anchor_w = anchor_w.cpu().numpy() * 32
		anchor_h = anchor_h.cpu().numpy() * 32
		
		fig = plt.figure()
		ax = fig.add_subplot(121)
		from PIL import Image
		img = Image.open("../img/street.jpg").resize([640, 640])
		plt.imshow(img, alpha=0.5)
		plt.ylim(-30, 650)
		plt.xlim(-30, 650)
		plt.scatter(grid_x, grid_y)
		plt.scatter(point_h * 32, point_w * 32, c='black')
		plt.gca().invert_yaxis()
		
		anchor_left = grid_x - anchor_w / 2
		anchor_top = grid_y - anchor_h / 2
		
		rect1 = plt.Rectangle([anchor_left[0, 0, point_h, point_w], anchor_top[0, 0, point_h, point_w]], \
		                      anchor_w[0, 0, point_h, point_w], anchor_h[0, 0, point_h, point_w], color="r", fill=False)
		rect2 = plt.Rectangle([anchor_left[0, 1, point_h, point_w], anchor_top[0, 1, point_h, point_w]], \
		                      anchor_w[0, 1, point_h, point_w], anchor_h[0, 1, point_h, point_w], color="r", fill=False)
		rect3 = plt.Rectangle([anchor_left[0, 2, point_h, point_w], anchor_top[0, 2, point_h, point_w]], \
		                      anchor_w[0, 2, point_h, point_w], anchor_h[0, 2, point_h, point_w], color="r", fill=False)
		
		ax.add_patch(rect1)
		ax.add_patch(rect2)
		ax.add_patch(rect3)
		
		ax = fig.add_subplot(122)
		plt.imshow(img, alpha=0.5)
		plt.ylim(-30, 650)
		plt.xlim(-30, 650)
		plt.scatter(grid_x, grid_y)
		plt.scatter(point_h * 32, point_w * 32, c='black')
		plt.scatter(box_xy[0, :, point_h, point_w, 0], box_xy[0, :, point_h, point_w, 1], c='r')
		plt.gca().invert_yaxis()
		
		pre_left = box_xy[..., 0] - box_wh[..., 0] / 2
		pre_top = box_xy[..., 1] - box_wh[..., 1] / 2
		
		rect1 = plt.Rectangle([pre_left[0, 0, point_h, point_w], pre_top[0, 0, point_h, point_w]], \
		                      box_wh[0, 0, point_h, point_w, 0], box_wh[0, 0, point_h, point_w, 1], color="r", fill=False)
		rect2 = plt.Rectangle([pre_left[0, 1, point_h, point_w], pre_top[0, 1, point_h, point_w]], \
		                      box_wh[0, 1, point_h, point_w, 0], box_wh[0, 1, point_h, point_w, 1], color="r", fill=False)
		rect3 = plt.Rectangle([pre_left[0, 2, point_h, point_w], pre_top[0, 2, point_h, point_w]], \
		                      box_wh[0, 2, point_h, point_w, 0], box_wh[0, 2, point_h, point_w, 1], color="r", fill=False)
		
		ax.add_patch(rect1)
		ax.add_patch(rect2)
		ax.add_patch(rect3)
		
		plt.show()
	
	
	#
	
	feat = torch.from_numpy(np.random.normal(0.2, 0.5, [4, 255, 20, 20])).float()
	anchors = np.array([[116, 90], [156, 198], [373, 326], [30, 61], [62, 45], [59, 119], [10, 13], [16, 30], [33, 23]])
	anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
	get_anchors_and_decode(feat, [640, 640], anchors, anchors_mask, 80)
