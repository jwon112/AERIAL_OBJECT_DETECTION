# utility/trainer.py

import time
import torch
from tqdm import tqdm

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

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0

        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[{model_name}] Epoch {epoch+1}/{epochs}")

        for i, (imgs, targets) in loop:
            imgs = imgs.to(device)
            targets = [t.to(device) for t in targets] if isinstance(targets, list) else targets.to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss, loss_items = criterion(preds, targets)  # loss_items = [box, obj, cls]
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
