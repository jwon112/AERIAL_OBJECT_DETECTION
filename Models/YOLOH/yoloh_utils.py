# ─────────────────────────────────────────────────────
# (A) import 정리 – 기존 맨 위 부분 교체/추가
# ─────────────────────────────────────────────────────
from datetime import datetime
import os, sys
import torch
from pathlib import Path

# model 내부 파일 import
from utility.path_manager import use_model_root, _MODEL_ROOTS
with use_model_root("YOLOH", verbose=False):
    from models.yoloh.yoloh import YOLOH
    from config.yoloh_config import *
    from utils.criterion import Criterion      # YOLOH 전용 loss

# 공용 유틸
from utility.dataloader_utils import get_dataloader
from utility.optimizer         import build_optimizer
from torch.optim.lr_scheduler  import CosineAnnealingLR
from utility.trainer           import run_train_loop, check_anchors
from utility.metrics           import (BoxResults, EvalResults, compute_coco_map,
                                       ConfusionMatrix, get_nms, scale_coords,
                                       xywh2xyxy, box_iou)
import numpy as np

def get_path():
    return print(__name__)


def build_yoloh_model(cfg=None, ex_dict=None):
    if isinstance(cfg, str):
        cfg = yoloh_config[cfg]
    device = ex_dict['Device']
    model = YOLOH(cfg=cfg, device=device, num_classes=ex_dict['Number of Classes'])
    model.to(device)
    return model

def _yolo_txt_to_targets(batch_targets):
    """
    YOLOTxtDataset 이 내놓는 (N,6) 텐서를
    Criterion 이 기대하는 list[dict] 로 변환.
    batch_targets : Tensor  [bi cls cx cy w h]  (0 또는 N행)
    """
    tgt_list = []
    if batch_targets.numel() == 0:           # 배치에 라벨이 없을 때
        tgt_list.append({'boxes': torch.zeros((0,4), device=batch_targets.device),
                         'labels': torch.zeros((0,),  device=batch_targets.device, dtype=torch.long)})
        return tgt_list

    for bi in batch_targets[:,0].unique():
        t = batch_targets[batch_targets[:,0] == bi][:,1:]   # cls | cx cy w h
        boxes  = t[:,1:]                                    # (k,4)
        labels = t[:,0].long()                              # (k,)
        tgt_list.append({'boxes': boxes,
                         'labels': labels,})
    return tgt_list



class YoloHLossWrapper:
    """
    Criterion → (loss_cls , loss_box , total)
    run_train_loop → (total , (box,obj,cls,total))  로 어댑터.
    Obj loss는 YOLOH 에 없으므로 0.0
    """
    def __init__(self, model, num_classes):
        self.model = model
        self.crit  = Criterion(cfg=model.cfg,
                               device=next(model.parameters()).device,
                               num_classes=num_classes,
                               loss_cls_weight=1.0,
                               loss_reg_weight=1.0)

    def __call__(self, preds, targets):
        # txt → Criterion 포맷
        tgt_list = _yolo_txt_to_targets(targets)

        # Criterion forward
        loss_cls, loss_box, total = self.crit(preds, tgt_list,
                                              anchor_boxes=self.model.anchor_boxes)

        loss_total = total
        loss_items = ( float(loss_box),   # box
                       0.0,               # obj (no objectness in YOLOH)
                       float(loss_cls),   # cls
                       float(total) )     # total
        return loss_total, loss_items


def train_yoloh_model(ex_dict):
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    device  = ex_dict['Device']
    model   = ex_dict['Model']
    epochs  = ex_dict['Epochs']
    name    = ex_dict['Model Name']

    # ── output dir
    project = os.path.join(
        ex_dict['Output Dir'],
        ex_dict['Experiment Time'],
        f"{ex_dict['Train Time']}_{name}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    )
    os.makedirs(project, exist_ok=True)

    # ── dataloader & anchor check (YOLOH anchor_size 사용)
    train_loader = get_dataloader("train", ex_dict)
    if hasattr(model.cfg, 'anchor_size'):
        # anchor_size: [[w,h], ...]  → (M,2) tensor
        model.anchors = torch.tensor(model.cfg.anchor_size, dtype=torch.float32, device=device)
        check_anchors(train_loader.dataset, model, imgsz=ex_dict['Image Size'])

    # ── optimizer / scheduler
    optimizer  = build_optimizer(model,
                                 base_lr      = ex_dict['LR'],
                                 name         = ex_dict['Optimizer'],
                                 momentum     = ex_dict['Momentum'],
                                 weight_decay = ex_dict['Weight Decay'])
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs)

    # ── loss wrapper
    num_cls   = ex_dict['Number of Classes']
    criterion  = YoloHLossWrapper(model, num_cls)

    # ── train loop (공용)
    ex_dict = run_train_loop(
        model       = model,
        train_loader= train_loader,
        criterion   = criterion,
        optimizer   = optimizer,
        scheduler   = scheduler,
        epochs      = epochs,
        device      = device,
        model_name  = "YOLOH",
        print_interval = 10,
        eval_fn     = eval_yoloh_model,   # per-epoch val (다음 섹션)
        ex_dict     = ex_dict
    )

    # ── save checkpoint (최종)
    pt_path = os.path.join(project, "Train", "weights", "best.pt")
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    torch.save(model.state_dict(), pt_path)
    ex_dict['PT path'] = pt_path
    return ex_dict


@torch.no_grad()
def eval_yoloh_model(ex_dict):
    device  = ex_dict['Device']
    model   = ex_dict['Model'].eval()
    loader  = get_dataloader("val", ex_dict, shuffle=False)
    criterion = YoloHLossWrapper(model, ex_dict['Number of Classes'])            # 같은 wrapper 사용

    total, nb = 0.0, len(loader)
    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss, _ = criterion(preds, targets)
        total += float(loss)

    avg = total / nb
    ex_dict["Val Results"] = {"total_loss": avg}
    print(f"[validate_yoloh_model] ✅ Avg Loss: {avg:.4f}")
    return ex_dict


@torch.no_grad()
def test_yoloh_model(ex_dict):
    """
    YOLOH 모델용 COCO-style mAP(0.5, 0.75, 0.5:0.95) 평가 + 속도 측정
    결과 형식은 YoloOW와 동일하게 EvalResults(BoxResults, speed) 반환.
    """
    import time, numpy as np
    from utility.metrics import (compute_coco_map, ConfusionMatrix, get_nms,
                                 scale_coords, xywh2xyxy, box_iou, BoxResults, EvalResults)

    device = ex_dict["Device"]
    model  = ex_dict["Model"].to(device).eval()
    test_loader = get_dataloader("test", ex_dict, shuffle=False)

    names = ex_dict["Class Names"]; nc = len(names)
    iouv  = np.linspace(0.5, 0.95, 10)

    stats, cm = [], ConfusionMatrix(nc)
    total_inf, total_nms, img_cnt = .0, .0, 0
    nms_fn = get_nms(version="v7")           # YOLOH도 v7 NMS 포맷과 동일

    for imgs, targets in test_loader:
        bs = imgs.size(0); img_cnt += bs
        imgs = imgs.to(device); targets = targets.to(device)

        # ── Inference 시간
        t0 = time.time(); preds_raw = model(imgs)
        total_inf += (time.time() - t0) * 1000

        preds = preds_raw[0] if isinstance(preds_raw, tuple) else preds_raw
        if isinstance(preds, torch.Tensor) and preds.ndim == 3:
            preds = [p for p in preds]

        # ── NMS
        t1 = time.time(); preds = nms_fn(preds, 0.25, 0.45)
        total_nms += (time.time() - t1) * 1000

        # ── 통계 누적
        for si, pred in enumerate(preds):
            lbl = targets[targets[:, 0] == si, 1:]    # (nl,5)
            nl  = len(lbl); tcls = lbl[:, 0].tolist() if nl else []
            if len(pred) == 0:
                if nl: stats.append((torch.zeros(0), torch.Tensor(), torch.Tensor(), tcls))
                continue

            predn = pred.clone()
            scale_coords(imgs[si].shape[1:], predn[:, :4], imgs[si].shape[1:])
            cm.process_batch(predn, torch.cat((lbl[:, :1], xywh2xyxy(lbl[:, 1:])), 1))

            correct = torch.zeros(pred.shape[0], dtype=torch.bool, device=device)
            if nl:
                detected = []
                tcls_t = lbl[:, 0]; tbox = xywh2xyxy(lbl[:, 1:])
                scale_coords(imgs[si].shape[1:], tbox, imgs[si].shape[1:])
                for cls in torch.unique(tcls_t):
                    ti = (cls == tcls_t).nonzero(as_tuple=False).squeeze(1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).squeeze(1)
                    if pi.numel():
                        ious, idx = box_iou(predn[pi, :4], tbox[ti]).max(1)
                        for j in (ious > 0.5).nonzero(as_tuple=False):
                            d = ti[idx[j]]
                            if d.item() not in detected:
                                detected.append(d.item())
                                correct[pi[j]] = True
                                if len(detected) == nl: break

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # ── mAP 계산
    stats = [torch.cat(x, 0).numpy() if isinstance(x[0], torch.Tensor)
             else np.concatenate(x, 0) for x in zip(*stats)]
    tp, conf, pred_cls, target_cls = stats
    p50, r50, map50_v, map5_95_v, map75_v, ap_cls = compute_coco_map(
        tp, conf, pred_cls, target_cls, iouv=iouv
    )

    box = BoxResults(
        mp   = p50.mean(),          mr   = r50.mean(),
        map50= map50_v.mean(),      map  = map5_95_v.mean(),   map75 = map75_v.mean(),
        ap_class_index = ap_cls,    names = names,
        per_class_data=[
            (p50[i], r50[i], map50_v[i], map5_95_v[i]) for i in range(len(ap_cls))
        ]
    )
    speed = { 'inference (ms/img)': round(total_inf/img_cnt,1),
              'nms (ms/img)'      : round(total_nms/img_cnt,1)}
    ex_dict["Test Results"] = EvalResults(box, speed)
    return ex_dict