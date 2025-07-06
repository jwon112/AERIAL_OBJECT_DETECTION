import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from matplotlib import pyplot as plt
import time
import torchvision.transforms as transforms

from nets.backbone import Backbone, C2f, Conv
from nets.yolo_training import weights_init
from utils.utils_bbox import make_anchors
from nets.CreativePoints import *



def fuse_conv_and_bn(conv, bn):
	# 混合Conv2d + BatchNorm2d 减少计算量
	# Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
	fusedconv = nn.Conv2d(conv.in_channels,
	                      conv.out_channels,
	                      kernel_size=conv.kernel_size,
	                      stride=conv.stride,
	                      padding=conv.padding,
	                      dilation=conv.dilation,
	                      groups=conv.groups,
	                      bias=True).requires_grad_(False).to(conv.weight.device)
	
	# 准备kernel
	w_conv = conv.weight.clone().view(conv.out_channels, -1)
	w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
	fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
	
	# 准备bias
	b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
	b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
	fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
	
	return fusedconv


class DFL(nn.Module):
	# DFL模块
	# Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
	def __init__(self, c1=16):
		super().__init__()
		self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
		x = torch.arange(c1, dtype=torch.float)
		self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
		self.c1 = c1
	
	def forward(self, x):
		# bs, self.reg_max * 4, 8400
		b, c, a = x.shape
		# bs, 4, self.reg_max, 8400 => bs, self.reg_max, 4, 8400 => b, 4, 8400
		# 以softmax的방식, 대해0~16的수자계산백분비, 획득최종수자。
		reshaped = x.view(b, 4, self.c1, a).transpose(2, 1)
		softmaxed = reshaped.softmax(1)
		result = self.conv(softmaxed).view(b, 4, a)
		
		# distance prediction이 음수가 되지 않도록 0 이상으로 제한
		result = result.clamp(min=0.0, max=self.c1 - 0.01)
		
		return result

# ---------------------------------------------------#
#   yolo_body
# ---------------------------------------------------#
class YoloBody(nn.Module):
	def __init__(self, input_shape, num_classes, phi, pretrained=False):
		super(YoloBody, self).__init__()
		
		# 디버깅: 모델 생성 정보
		print(f"[DEBUG Model] Model creation:")
		print(f"  Input shape: {input_shape}")
		print(f"  Num classes: {num_classes}")
		print(f"  Phi: {phi}")
		print(f"  Pretrained: {pretrained}")
		
		depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00, }
		width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
		deep_width_dict = {'n': 1.00, 's': 1.00, 'm': 0.75, 'l': 0.50, 'x': 0.50, }
		dep_mul, wid_mul, deep_mul = depth_dict[phi], width_dict[phi], deep_width_dict[phi]
		
		base_channels = int(wid_mul * 64)  # 64
		self.base_channels = int(wid_mul * 64)  # 64
		base_depth = max(round(dep_mul * 3), 1)
		self.backbone = Backbone(base_channels, base_depth, deep_mul, phi, pretrained=pretrained)
		
		# ------------------------加强特征提取网络------------------------#
		# feat3 채널 수에 맞춰 CARAFE 수정 (deep_mul 고려)
		feat3_channels = int(base_channels * 16 * deep_mul)
		self.upsample1 = CARAFE(feat3_channels, base_channels * 8)  # feat3 -> P5_upsample
		self.upsample2 = CARAFE(base_channels * 8, base_channels * 8)  # P4 -> P4_upsample
		
		# P5_upsample(base_channels*8) + feat2(base_channels*8) => base_channels*8
		self.conv3_for_upsample1 = ACmix(base_channels * 16, base_channels * 8)
		# 768, 80, 80 => 256, 80, 80
		self.conv3_for_upsample2 = ACmix(base_channels * 8 + base_channels * 4, base_channels * 4)
		
		# 256, 80, 80 => 256, 40, 40
		self.down_sample1 = Conv(base_channels * 4, base_channels * 4, 3, 2)
		# P3_downsample(base_channels*4) + P4(base_channels*8) + feat2(base_channels*8) => base_channels*20
		self.conv3_for_downsample1 = ACmix(base_channels * 20, base_channels * 8)
		
		# 512, 40, 40 => 512, 20, 20
		self.down_sample2 = Conv(base_channels * 8, base_channels * 8, 3, 2)
		# P4_downsample(base_channels*8) + feat3(deep_mul*base_channels*16) => (8+16*deep_mul)*base_channels
		self.conv3_for_downsample2 = ACmix(int(base_channels * 16 * deep_mul) + base_channels * 8, int(base_channels * 16 * deep_mul))
		
		ch = [base_channels * 4, base_channels * 8, int(base_channels * 16 * deep_mul)]
		self.shape = None
		self.nl = len(ch)
		# self.stride     = torch.zeros(self.nl)
		self.stride = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
		self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
		self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
		self.num_classes = num_classes
		
		# 디버깅: 네트워크 구조 정보
		print(f"[DEBUG Model] Network structure:")
		print(f"  Base channels: {base_channels}")
		print(f"  Channels: {ch}")
		print(f"  Strides: {self.stride}")
		print(f"  Reg max: {self.reg_max}")
		print(f"  Num outputs per anchor: {self.no}")
		print(f"  Num classes: {self.num_classes}")
		
		c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], num_classes)  # channels
		self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
		self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, num_classes, 1)) for x in ch)
		
		# 디버깅: cv3 레이어 정보
		print(f"[DEBUG Model] CV3 layers:")
		for i, m in enumerate(self.cv3):
			print(f"  cv3[{i}] input channels: {m[0].conv.in_channels}")
			print(f"  cv3[{i}] output channels: {m[-1].out_channels}")
			print(f"  cv3[{i}] expected output: {num_classes}")
		
		if not pretrained:
			# 디버깅: 초기화 전 가중치 확인
			print(f"[DEBUG Model] Before initialization:")
			for i, m in enumerate(self.cv3):
				final_conv = m[-1]  # 마지막 Conv2d 레이어
				print(f"  cv3[{i}] final conv - weight range: [{final_conv.weight.data.min().item():.4f}, {final_conv.weight.data.max().item():.4f}]")
				print(f"  cv3[{i}] final conv - weight mean: {final_conv.weight.data.mean().item():.4f}, std: {final_conv.weight.data.std().item():.4f}")
				if final_conv.bias is not None:
					print(f"  cv3[{i}] final conv - bias range: [{final_conv.bias.data.min().item():.4f}, {final_conv.bias.data.max().item():.4f}]")
			
			weights_init(self)
			
			# 디버깅: 초기화 후 가중치 확인
			print(f"[DEBUG Model] After initialization:")
			for i, m in enumerate(self.cv3):
				final_conv = m[-1]  # 마지막 Conv2d 레이어
				print(f"  cv3[{i}] final conv - weight range: [{final_conv.weight.data.min().item():.4f}, {final_conv.weight.data.max().item():.4f}]")
				print(f"  cv3[{i}] final conv - weight mean: {final_conv.weight.data.mean().item():.4f}, std: {final_conv.weight.data.std().item():.4f}")
				if final_conv.bias is not None:
					print(f"  cv3[{i}] final conv - bias range: [{final_conv.bias.data.min().item():.4f}, {final_conv.bias.data.max().item():.4f}]")
		
		self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
	
	def fuse(self):
		print('Fusing layers... ')
		for m in self.modules():
			if type(m) is Conv and hasattr(m, 'bn'):
				m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
				delattr(m, 'bn')  # remove batchnorm
				m.forward = m.forward_fuse  # update forward
		return self
	
	def parallel_fpn(self, feat1, feat2, feat3):
		"""
		Args:
			feat1: feat1_mask
			feat2: feat2_mask
			feat3: feat3_mask
		"""
		# ------------------------加强特征提取网络------------------------#
		# feat3 (deep_mul*base_channels*16), 20, 20 => base_channels*8, 40, 40
		P5_upsample = self.upsample1(feat3)
		
		# base_channels*8, 40, 40 cat base_channels*8, 40, 40 => base_channels*16, 40, 40 --> base_channels*8, 40, 40
		P4 = torch.cat([P5_upsample, feat2], 1)
		
		# base_channels*16, 40, 40 => base_channels*8, 40, 40
		P4 = self.conv3_for_upsample1(P4)
		
		# base_channels*8, 40, 40 => base_channels*8, 80, 80
		P4_upsample = self.upsample2(P4)
		
		# base_channels*8, 80, 80 cat base_channels*4, 80, 80 => base_channels*12, 80, 80
		P3 = torch.cat([P4_upsample, feat1], 1)
		
		# base_channels*12, 80, 80 => base_channels*4, 80, 80
		P3 = self.conv3_for_upsample2(P3)
		
		# base_channels*4, 80, 80 => base_channels*4, 40, 40
		P3_downsample = self.down_sample1(P3)
		
		# base_channels*8, 40, 40 cat base_channels*4, 40, 40 => base_channels*12, 40, 40
		P4 = torch.cat([P3_downsample, P4], 1)
		
		# base_channels*12, 40, 40 => base_channels*20, 40, 40
		P4 = torch.cat([P4, feat2], 1)
		# base_channels*20, 40, 40 => base_channels*8, 40, 40
		P4 = self.conv3_for_downsample1(P4)
		
		# base_channels*8, 40, 40 => base_channels*8, 20, 20
		P4_downsample = self.down_sample2(P4)
		
		# base_channels*8, 20, 20 cat deep_mul*base_channels*16, 20, 20 => (8+16*deep_mul)*base_channels, 20, 20
		P5 = torch.cat([P4_downsample, feat3], 1)
		# (8+16*deep_mul)*base_channels, 20, 20 => deep_mul*base_channels*16, 20, 20
		P5 = self.conv3_for_downsample2(P5)
		# ------------------------加强特征提取网络------------------------#
		# P3 base_channels*4, 80, 80
		# P4 base_channels*8, 40, 40
		# P5 deep_mul*base_channels*16, 20, 20
		shape = P3.shape  # BCHW
		return [P3, P4, P5, shape]
	
	def forward(self, x):
		# backbone
		feat1, feat2, feat3 = self.backbone.forward(x)
	
		# draw_features(1, 1, feat1.cpu().numpy()[:, :, :, :], "test1")
		x = self.parallel_fpn(feat1, feat2, feat3)
		shape = x[-1]
		x = x[:-1]  # remove shape
		# P3 256, 80, 80 => num_classes + self.reg_max * 4, 80, 80
		# P4 512, 40, 40 => num_classes + self.reg_max * 4, 40, 40
		# P5 1024 * deep_mul, 20, 20 => num_classes + self.reg_max * 4, 20, 20
		for i in range(self.nl):
			# 클래스 예측 레이어의 출력 확인
			cls_pred = self.cv3[i](x[i])
			print(f"[DEBUG] Layer {i} class predictions:")
			print(f"  Shape: {cls_pred.shape}")
			print(f"  Raw range: [{cls_pred.min().item():.4f}, {cls_pred.max().item():.4f}]")
			print(f"  After sigmoid: [{torch.sigmoid(cls_pred).min().item():.4f}, {torch.sigmoid(cls_pred).max().item():.4f}]")
			
			# 가중치 통계
			for name, param in self.cv3[i].named_parameters():
				if 'weight' in name:
					print(f"  {name} - range: [{param.data.min().item():.4f}, {param.data.max().item():.4f}]")
					print(f"  {name} - mean: {param.data.mean().item():.4f}, std: {param.data.std().item():.4f}")
			
			x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
		
		if self.shape != shape:
			self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
			self.shape = shape
		
		# num_classes + self.reg_max * 4 , 8400 =>  cls num_classes, 8400;
		#                                           box self.reg_max * 4, 8400
		box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.num_classes), 1)
		
		# 최종 클래스 예측값 확인
		print(f"[DEBUG] Final class predictions:")
		print(f"  Shape: {cls.shape}")
		print(f"  Raw range: [{cls.min().item():.4f}, {cls.max().item():.4f}]")
		print(f"  After sigmoid: [{torch.sigmoid(cls).min().item():.4f}, {torch.sigmoid(cls).max().item():.4f}]")
		
		dbox = self.dfl(box)
		return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)
