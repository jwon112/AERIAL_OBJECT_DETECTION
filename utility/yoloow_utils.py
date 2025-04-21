from datetime import datetime
import os,sys
import torch
from pathlib import Path
from utility.dataloader_utils import get_dataloader 
#from Models.YoloOW.models.yolo import Model as YoloOWModel
#from Models.YoloOW.utils.loss import ComputeLoss
from utility.path_manager import use_model_root, _MODEL_ROOTS
with use_model_root("YoloOW"):
    from models.yolo import Model as YoloOWModel
    from utils.loss import ComputeLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from coco_eval import CocoEvaluator
from utility.optimizer import build_optimizer
import yaml

def build_yoloow_model(cfg, ex_dict=None):
    cfg_path = Path(cfg)
    if cfg_path.is_file():
        pass
    else:
        cfg_path = _MODEL_ROOTS["YoloOW"] / "cfg" / "training" / cfg
        if not cfg_path.is_file():                     
            raise FileNotFoundError(f"[YoloOW] config file not found: {cfg_path}")

    print(f"[YoloOW] Using config → {cfg_path}")         # 디버그 로그
    # 하이퍼 파라미터 세팅
    model = YoloOWModel(cfg=str(cfg_path), ch=3, nc=ex_dict['Number of Classes']).to(ex_dict['Device'])
    hyp_path = _MODEL_ROOTS["YoloOW"] / "data" / "hyp.scratch.p5.yaml"
    with open(hyp_path, 'r') as f:
        model.hyp = yaml.safe_load(f)

    # gradient ratio
    model.gr = 1.0 #defualt
    
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

    train_loader = get_dataloader("train", ex_dict)
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
    for epoch in range(epochs):
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss, _ = criterion(preds, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"[YoloOW] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        

    pt_path = os.path.join(project, "Train", "weights", "best.pt")
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    torch.save(model.state_dict(), pt_path)
    ex_dict['PT path'] = pt_path

    return ex_dict 

def eval_yoloow_model(ex_dict):
    model = ex_dict['Model']
    device = ex_dict['Device']
    pt_path = ex_dict.get('PT path')

    if pt_path and os.path.exists(pt_path):
        model.load_state_dict(torch.load(pt_path, map_location=device))

    val_loader = get_dataloader("val", ex_dict, shuffle=False)
    evaluator = CocoEvaluator()
    model.eval()
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            evaluator.update(preds, targets)

    ex_dict['Test Results'] = evaluator.summarize()
    return ex_dict
