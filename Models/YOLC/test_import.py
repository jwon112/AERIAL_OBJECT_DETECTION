#!/usr/bin/env python3

print("=== YOLC 디렉토리에서 import 테스트 ===")

# train.py와 동일한 import 순서로 테스트
import argparse
import copy
import os
import os.path as osp
import time
import warnings

print("✅ 기본 import 성공")

import mmcv
print("✅ mmcv import 성공")

import torch
print("✅ torch import 성공")

import torch.distributed as dist
print("✅ torch.distributed import 성공")

# mmcv 버전 호환성을 위한 Config import (train.py와 동일)
try:
    from mmcv import Config, DictAction
    print("✅ from mmcv import Config, DictAction 성공")
except ImportError as e:
    print(f"❌ from mmcv import Config, DictAction 실패: {e}")
    try:
        from mmengine import Config
        from mmcv import DictAction
        print("✅ mmengine + mmcv 조합 성공")
    except ImportError as e:
        print(f"❌ mmengine + mmcv 조합 실패: {e}")
        try:
            from mmcv.utils import Config, DictAction
            print("✅ mmcv.utils 조합 성공")
        except ImportError as e:
            print(f"❌ mmcv.utils 조합 실패: {e}")
            print("모든 import 방법이 실패했습니다.")

print("=== 테스트 완료 ===") 