# utility/trainer.py

import time
import torch
from tqdm import tqdm
import numpy as np
from utility.metrics import xywh2xyxy, box_iou 
from utility.autoanchor import check_anchors as autoanchor_check_anchors
import numpy as np
import torch
from utility.metrics import box_iou

def check_anchors(dataset, model, imgsz=640, threshold=4.0):
    print("\n[trainer] 🔍 Checking anchor fit to dataset...")
    if not hasattr(model, "anchors") or model.anchors is None:
        print("[trainer] ⚠️ model has no `.anchors` attribute. Skipping anchor check.")
        return

    labels = np.concatenate(dataset.labels, 0)
    if labels.size == 0:
        print("[trainer] ⚠️ No labels found in dataset. Skipping anchor check.")
        return

    # normalized w,h → 픽셀 단위 numpy array
    wh_np = labels[:, 3:5] * imgsz  # shape (N,2)

    # GPU에 올라간 앵커 벡터
    anchor_vec = model.anchors.clone().view(-1, 2)  # tensor, e.g. device=cuda:0
    device = anchor_vec.device

    try:
        # 4-coords 박스 (0,0,w,h)
        b1 = torch.from_numpy(
            np.hstack([np.zeros_like(wh_np), wh_np])
        ).float().to(device)       # (N,4), .to(device) 로 GPU로 이동

        b2 = torch.cat([
            torch.zeros_like(anchor_vec),  # x1=0, y1=0
            anchor_vec                      # x2=aw, y2=ah
        ], dim=1)                          # (M,4) already on `device`

        # IoU 계산
        ious = box_iou(b1, b2)            # (N,M)
        best_per_wh = ious.max(1)[0]      # 각 gt-box별 best IoU
        best_ratio = best_per_wh.mean().item()

    except Exception as e:
        print(f"[trainer] ⚠️ Anchor check skipped: {e}")
        return

    if best_ratio < threshold:
        print(f"[trainer] ⚠️ Low anchor fit ({best_ratio:.2f} < {threshold}). Consider running autoanchor.")
    else:
        print(f"[trainer] ✅ Anchor fit okay (mean best IoU: {best_ratio:.2f}).")


def run_train_loop(
    model,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    device,
    model_name="model",
    print_interval=10,
    eval_fn=None,
    ex_dict=None
):
    model.train()
    global_step = 0
    warmup_iters = min(1000, len(train_loader) * 5)  # warmup iterations

    # Store initial learning rate for logging
    for pg in optimizer.param_groups:
        pg.setdefault("initial_lr", pg["lr"])

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[{model_name}] Epoch {epoch+1}/{epochs}")

        for i, (imgs, targets) in loop:
            
            if i == 0 and epoch == 0:          # 최초 1회만
                # targets → collate_fn 에 의해 (N,6) Tensor 또는 0-크기 Tensor
                print("\n[DEBUG] first batch target rows =", targets.shape[0])
                if targets.shape[0]:
                    print("[DEBUG] first 5 targets:\n", targets[:5].cpu())

            imgs = imgs.to(device)
            targets = [t.to(device) for t in targets] if isinstance(targets, list) else targets.to(device)

            # Warmup
            if global_step <= warmup_iters:
                xi = [0, warmup_iters]
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(global_step, xi, [0.0, x['initial_lr']])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(global_step, xi, [0.8, 0.937])
                if hasattr(model, "gr"):
                    model.gr = np.interp(global_step, xi, [0.0, 1.0])


            optimizer.zero_grad()
            preds = model(imgs)
            # 🔁 Loss는 FPN 피처맵(preds[1]) 기준으로 계산
            preds_for_loss = preds[1] if isinstance(preds, tuple) else preds
            loss, loss_items = criterion(preds_for_loss, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if (i + 1) % print_interval == 0 or (i + 1) == len(train_loader):
                current_lr = optimizer.param_groups[0]["lr"]
                loop.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "box": f"{loss_items[0]:.3f}",
                    "obj": f"{loss_items[1]:.3f}",
                    "cls": f"{loss_items[2]:.3f}",
                    "total loss": f"{loss_items[3]:.3f}",
                    "lr": f"{current_lr:.6f}"
                })

        scheduler.step()
        epoch_time = time.time() - start_time
        print(f"\n[{model_name}] ✅ Epoch {epoch+1}/{epochs} complete in {epoch_time:.2f}s | Avg Loss: {total_loss/len(train_loader):.4f}")

        # 🔍 optional eval step
        if eval_fn:
            print(f"\n[{model_name}] 📊 Running evaluation after epoch {epoch+1} ...")
            ex_dict = eval_fn(ex_dict)

    return ex_dict
