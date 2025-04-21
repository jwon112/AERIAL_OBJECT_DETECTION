from datetime import datetime
import os, sys
import torch
from utility.trainer import run_train_loop
#from Models.YOLOH.models.yoloh.yoloh import YOLOH
#from Models.YOLOH.config.yoloh_config import *
#from Models.YOLOH.utils.criterion import Criterion
from utility.path_manager import use_model_root,  _MODEL_ROOTS
with use_model_root("YOLOH", verbose=True):    
    root = _MODEL_ROOTS["YOLOH"]
    print(" 👀  현재 sys.path[0] :", sys.path[0])
    print(" 📂  root 내용:", os.listdir(root)[:10])
    print("📂  root/models/yoloh :", (root / "models"/ "yoloh").exists())             # 🔑 경로 오염 방지!
    from models.yoloh.yoloh import YOLOH   
    from config.yoloh_config import *
    from utils.criterion import Criterion

from utility.dataloader_utils import get_dataloader
from utility.optimizer import build_optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from coco_eval import CocoEvaluator

from utility.path_manager import use_model_root
with use_model_root("YOLOH"):                 # 🔑 경로 오염 방지!
    from models.yoloh.yoloh import YOLOH   
    from config.yoloh_config import *
    from utils.criterion import Criterion
def get_path():
    return print(__name__)


def build_yoloh_model(cfg=None, ex_dict=None):
    if isinstance(cfg, str):
        cfg = yoloh_config[cfg]
    model = YOLOH(cfg=cfg, device=ex_dict['Device'], num_classes=ex_dict['Number of Classes']).to(ex_dict['Device'])
    return model

def train_yoloh_model(ex_dict):

    model = ex_dict['Model']
    device = ex_dict['Device']
    epochs = ex_dict['Epochs']
    ex_dict['Train Time'] = datetime.now().strftime("%y%m%d_%H%M%S")

    data_root = ex_dict['Data Config']['root']
    batch_size = ex_dict['Batch Size']
 
    name = "Train"
    project = os.path.join(
        ex_dict['Output Dir'],
        ex_dict['Experiment Time'],
        f"{ex_dict['Train Time']}_{ex_dict['Model Name']}_{ex_dict['Dataset Name']}_Iter_{ex_dict['Iteration']}"
    )
    os.makedirs(project, exist_ok=True)

    optimizer = build_optimizer(
        model, 
        base_lr=ex_dict['LR'],
        name=ex_dict['Optimizer'],
        momentum=ex_dict['Momentum'],
        weight_decay=ex_dict['Weight Decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    train_loader = get_dataloader("train", ex_dict)
    criterion = Criterion()
    model.train()
    

    ex_dict = run_train_loop(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=ex_dict["Epochs"],
        device=device,
        model_name="YOLOH",
        print_interval=10,
        eval_fn=eval_yoloh_model,
        ex_dict=ex_dict
    )

    pt_path = os.path.join(project, name, "weights", "best.pt")
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    torch.save(model.state_dict(), pt_path)
    ex_dict['PT path'] = pt_path

    return ex_dict

def eval_yoloh_model(ex_dict):
    model = ex_dict['Model']
    device = ex_dict['Device']
    #data_config = ex_dict['Data Config']
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

class YOLOHLossWrapper:
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, preds, targets):
        loss = self.criterion(preds, targets)
        # YOLOH의 경우 구성 요소가 없으므로 dummy 값을 반환
        loss_items = [loss.item(), 0.0, 0.0]  # box, obj, cls 대신 단일 loss만
        return loss, loss_items