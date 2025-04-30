from datetime import datetime
import os,sys
import torch
from pathlib import Path
from utility.trainer import run_train_loop, autoanchor_check_anchors, check_anchors
import utility.dataloader_utils as dlu
sys.modules['utility.dataloader_utils'] = dlu

#from Models.YoloOW.models.yolo import Model as YoloOWModel
#from Models.YoloOW.utils.loss import ComputeLoss
from utility.path_manager import use_model_root, _MODEL_ROOTS
with use_model_root("YoloOW"):
    from models.yolo import Model as YoloOWModel
    from utils.loss import ComputeLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from utility.metrics import *
from utility.optimizer import build_optimizer
import yaml

def build_yoloow_model(cfg, ex_dict=None):
    cfg_path = Path(cfg)
    if not cfg_path.is_file():
        cfg_path = _MODEL_ROOTS["YoloOW"] / "cfg" / "training" / cfg
        if not cfg_path.is_file():
            raise FileNotFoundError(f"[YoloOW] config file not found: {cfg_path}")

    print(f"[YoloOW] Using config → {cfg_path}")         # 디버그 로그
    # 하이퍼 파라미터 세팅

    # YAML 로드
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)


    model = YoloOWModel(cfg=str(cfg_path), ch=3, nc=ex_dict['Number of Classes']).to(ex_dict['Device'])

    # 앵커 정보 붙이기
    #    config 파일에 anchors: [[w1,h1],[w2,h2],...] 형태로 정의돼 있어야 합니다.
    if 'anchors' in cfg_dict:
        # (M,2) 텐서로 변환
        anchor_np = cfg_dict['anchors']
        model.anchors = torch.tensor(anchor_np, dtype=torch.float32, device=ex_dict['Device'])
    else:
        model.anchors = None  # 없으면 체크 스킵
        
    hyp_path = _MODEL_ROOTS["YoloOW"] / "data" / "hyp.scratch.p5.yaml"
    with open(hyp_path, 'r') as f:
        model.hyp = yaml.safe_load(f)

    return model

def train_yoloow_model(ex_dict):
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")
    model = ex_dict['Model']
    device = ex_dict['Device']
    epochs = ex_dict['Epochs']
    model_name = ex_dict['Model Name']
    output_dir = ex_dict['Output Dir']

    experiment_time = ex_dict['Experiment Time']
    project = os.path.join(
        output_dir,
        experiment_time,
        f"{ex_dict['Train Time']}_{model_name}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    )
    os.makedirs(project, exist_ok=True)

    train_loader = dlu.get_dataloader("train", ex_dict)

    if ex_dict.get('AutoAnchor', False):
        autoanchor_check_anchors(train_loader.dataset, model, thr=4.0, imgsz=ex_dict['Image Size'])
    else:
        check_anchors(train_loader.dataset, model, imgsz=ex_dict['Image Size'])

    model.gr = 1.0 #default

    criterion = ComputeLoss(model)
    optimizer = build_optimizer(
        model, 
        base_lr=ex_dict['LR'],
        name=ex_dict['Optimizer'],
        momentum=ex_dict['Momentum'],
        weight_decay=ex_dict['Weight Decay']
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) 

    model.train()

    #학습 시작
    ex_dict = run_train_loop(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=ex_dict["Epochs"],
        device=device,
        model_name="YoloOW",
        print_interval=10,
        eval_fn=eval_yoloow_model,
        ex_dict=ex_dict
    )
        

    pt_path = os.path.join(project, "Train", "weights", "best.pt")
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    torch.save(model.state_dict(), pt_path)
    ex_dict['PT path'] = pt_path

    return ex_dict

@torch.no_grad()
def eval_yoloow_model(ex_dict):
    model   = ex_dict["Model"]
    device  = ex_dict["Device"]
    val_loader = dlu.get_dataloader("val", ex_dict, shuffle=False)
    criterion = ComputeLoss(model)          # model.hyp, model.gr 필요
    
    model.eval()

    nb              = len(val_loader)       # #batches
    loss_sum        = torch.zeros(4, device=device)  # box/obj/cls/total

    for imgs, targets in val_loader:
        imgs     = imgs.to(device)
        targets  = targets.to(device)

        # forward
        preds = model(imgs)

        # YOLOv7 계열은 (pred_infer, pred_train) tuple 을 내놓을 수 있음
        preds_for_loss = preds[1] if (isinstance(preds, tuple) and preds[1] is not None) else preds

        loss, loss_items = criterion(preds_for_loss, targets)  # loss_items: [box, obj, cls, total]
        loss_sum += loss_items

    # ───── 결과 집계 ───────────────────────────────────
    mean_loss = loss_sum / nb                            # 4-tensor
    box_l, obj_l, cls_l, tot_l = mean_loss.tolist()

    ex_dict["Val Results"] = {
        "box_loss"  : box_l,
        "obj_loss"  : obj_l,
        "cls_loss"  : cls_l,
        "total_loss": tot_l,
    }

    print(f"[validate_yoloow_model] ✅  Avg Loss  (box/obj/cls/total): "
          f"{box_l:.4f} / {obj_l:.4f} / {cls_l:.4f} / {tot_l:.4f}")

    return ex_dict


@torch.no_grad()
def test_yoloow_model(ex_dict):
    import time
    import numpy as np
    from utility.metrics import (compute_coco_map, ConfusionMatrix, get_nms,
                                 scale_coords, xywh2xyxy, box_iou)

    device = ex_dict['Device']
    model  = ex_dict['Model'].to(device).eval()
    test_loader = dlu.get_dataloader("test", ex_dict, shuffle=False)
    # Debug용
    img, targets = next(iter(test_loader))
    print("[TEST] img.shape =", img.shape)
    print("[TEST] targets.shape =", targets.shape)


    names = ex_dict["Class Names"]
    nc    = len(names)
    iouv  = np.linspace(0.5, 0.95, 10)

    # ── 통계 수집용 버퍼 ────────────────────────────────────
    stats = []                       # (tp, conf, pred_cls, target_cls)
    confusion_matrix = ConfusionMatrix(nc)
    total_inf, total_nms, img_cnt = 0., 0., 0

    nms_fn = get_nms(version="v7")

    for imgs, targets in test_loader:
        bs       = imgs.size(0)
        img_cnt += bs
        imgs     = imgs.to(device)
        targets  = targets.to(device)

        # ── 1) Inference time ───────────────────────────
        t0 = time.time()
        preds_raw = model(imgs)
        total_inf += (time.time() - t0) * 1000  # ms

        # ── 2) list[Tensor] 형태로 변환 ───────────────
        preds = preds_raw[0] if isinstance(preds_raw, tuple) else preds_raw
        if isinstance(preds, torch.Tensor) and preds.ndim == 3:
            preds = [p for p in preds]

        # ── 3) NMS time ────────────────────────────────
        t1 = time.time()
        preds = nms_fn(preds, conf_thres=0.01, iou_thres=0.45)
        print(f"[DEBUG] number of preds after NMS: {[len(p) for p in preds]}")
        print(f"[DEBUG] number of targets in batch: {targets.shape[0]}")

                # 🧩 4-1. 그리고 여기에 **IoU 매칭 디버깅 코드** 붙여넣기
        if preds and targets.numel() > 0:
            from utility.metrics import box_iou

            for si, pred in enumerate(preds):
                gt = targets[targets[:, 0] == si]  # batch별 target
                if gt.numel() == 0:
                    print(f"[DEBUG] batch {si}: no GTs")
                    continue
                gt_boxes = gt[:, 1:5]  # class, cx, cy, w, h
                pred_boxes = pred[:, :4]

                # 필요하면 gt_boxes를 xywh2xyxy로 변환하는 코드도 추가할 수 있음
                ious = box_iou(pred_boxes, gt_boxes)

                max_ious = ious.max(1)[0]
                matched = (max_ious > 0.5).sum().item()

                print(f"[DEBUG] batch {si}: {len(pred_boxes)} preds vs {len(gt_boxes)} GTs -> {matched} matches (IoU>0.5)")
        total_nms += (time.time() - t1) * 1000

        # ── 4) 통계 누적 ───────────────────────────────
        for si, pred in enumerate(preds):
            lbl = targets[targets[:, 0] == si, 1:]      # (nl,5)  cls,cx,cy,w,h
            nl  = len(lbl)
            tcls = lbl[:, 0].tolist() if nl else []

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # bbox 좌표 복원
            predn = pred.clone()
            scale_coords(imgs[si].shape[1:], predn[:, :4], imgs[si].shape[1:])

            # confusion matrix
            confusion_matrix.process_batch(predn,
                                            torch.cat((lbl[:, :1], xywh2xyxy(lbl[:, 1:])), 1))

            # T/F 판정용
            correct = torch.zeros(pred.shape[0], dtype=torch.bool, device=device)
            if nl:
                detected = []
                tcls_t   = lbl[:, 0]
                tbox     = xywh2xyxy(lbl[:, 1:])
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
                                if len(detected) == nl:
                                    break

            stats.append((correct.cpu(),
                          pred[:, 4].cpu(),
                          pred[:, 5].cpu(),
                          tcls))

    # ── 5) COCO-mAP 계산 ───────────────────────────────
    #   stats: List[(tp(bool), conf, pred_cls, target_cls)]  → 4 array
    stats = [torch.cat(x, 0).numpy() if isinstance(x[0], torch.Tensor)
            else np.concatenate(x, 0) for x in zip(*stats)]
    tp, conf, pred_cls, target_cls = stats

    p50, r50, map50_vec, map5_95_vec, map75_vec, ap_class = compute_coco_map(
        tp, conf, pred_cls, target_cls, iouv=iouv
    )

    overall_map50   = map50_vec.mean()
    overall_map5_95 = map5_95_vec.mean()
    overall_map75   = map75_vec.mean()

    # ── 6) BoxResults 변환 ─────────────────────────────
    box_results = BoxResults(
        mp   = p50.mean(),
        mr   = r50.mean(),
        map50= overall_map50,
        map  = overall_map5_95,
        map75= overall_map75,
        ap_class_index = ap_class,
        names = names,
        per_class_data=[
            (p50[i], r50[i], map50_vec[i], map5_95_vec[i]) for i in range(len(ap_class))
        ]
    )

    # 속도 정보
    speed_dict = {
        'inference (ms/img)': round(total_inf / img_cnt, 1),
        'nms (ms/img)':        round(total_nms / img_cnt, 1)
    }
    box_results.speed = speed_dict


    ex_dict['Test Results'] = EvalResults(box_results, speed_dict)
    # ✅ Test 결과 출력 추가
    if "Test Results" in ex_dict:
        print("\n[TEST] 📋 Test Results Summary:")
        box = ex_dict["Test Results"].box
        print(f"  - Mean Precision     : {box.mp:.4f}")
        print(f"  - Mean Recall        : {box.mr:.4f}")
        print(f"  - mAP@0.5            : {box.map50:.4f}")
        print(f"  - mAP@0.5:0.95       : {box.map:.4f}")
        print(f"  - mAP@0.75           : {box.map75:.4f}")

    return ex_dict


