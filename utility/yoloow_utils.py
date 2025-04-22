from datetime import datetime
import os,sys
import torch
from pathlib import Path
from utility.trainer import run_train_loop
from utility.dataloader_utils import get_dataloader 
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



def eval_yoloow_model(ex_dict):

    from utility.metrics import evaluate_yolo_predictions

    model = ex_dict['Model']
    device = ex_dict['Device']
    pt_path = ex_dict.get('PT path')

    if pt_path and os.path.exists(pt_path):
        model.load_state_dict(torch.load(pt_path, map_location=device))

##############################
    # [1] 예측 결과 및 GT 준비
    preds = []
    gts = []
    # [2] 저장 경로 정의
    save_dir = os.path.join(
        ex_dict["Output Dir"],
        ex_dict["Experiment Time"],
        f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}",
        "Test"
    )

    from utility.metrics import evaluate_yolo_predictions, BoxResults
    # [3] 평가 실행
    precision, recall, ap, ap_class = evaluate_yolo_predictions(
        preds=preds,
        gts=gts,
        iou_thres=0.5,
        save_dir=save_dir  # 이 경로에 PR/ROC curves 저장됨
    )

     # [4] BoxResults 객체 생성
    per_class_data = [
        (precision[i], recall[i], ap[i], ap[i]) for i in range(len(ap_class))
    ]

    box_results = BoxResults(
        mp=precision.mean(),
        mr=recall.mean(),
        map50=ap.mean(),
        map=ap.mean(),
        map75=ap.mean(),  # 동일한 값으로 설정
        ap_class_index=ap_class,
        names=ex_dict["Class Names"],
        per_class_data=per_class_data
    )

    # [5] 결과를 ex_dict에 저장 (main에서 format_measures로 후처리 가능)
    ex_dict["Test Results"] = {
        "box": box_results,
        "speed": {
            "inference": 0.0,
            "nms": 0.0,
            "total": 0.0
        }
    }


    return ex_dict 

################################

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
