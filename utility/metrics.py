# utility/metrics.py
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from utils.loss import bbox_iou


class BoxResults:
    def __init__(self, mp, mr, map50, map, map75, ap_class_index, names, per_class_data):
        self.mp = mp                  # Mean Precision
        self.mr = mr                  # Mean Recall
        self.map50 = map50            # mAP@0.5
        self.map = map                # mAP@0.5:0.95 (fallback to map50 if not available)
        self.map75 = map75            # mAP@0.75 (optional)
        self.ap_class_index = ap_class_index  # class indices
        self.names = names            # class names
        self.per_class_data = per_class_data  # list of tuples: (p, r, ap50, ap95)

    def class_result(self, i):
        return self.per_class_data[i]  # returns (p, r, ap50, ap95)


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    # Compute AP, precision, and recall per class
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes = np.unique(target_cls)
    ap, precision, recall = [], [], []

    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 or n_gt == 0:
            ap.append(0.0)
            recall.append(0.0)
            precision.append(0.0)
            continue

        fpc = (1 - tp[i]).cumsum()
        tpc = tp[i].cumsum()

        recall_curve = tpc / (n_gt + eps)
        recall.append(recall_curve[-1])

        precision_curve = tpc / (tpc + fpc + eps)
        precision.append(precision_curve[-1])

        ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(precision), np.array(recall), np.array(ap), unique_classes.astype(int)


def compute_ap(recall, precision):
    # 101-point interpolated average precision
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def xywh2xyxy(x):
    # Convert [cx, cy, w, h] to [x1, y1, x2, y2]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def evaluate_yolo_predictions(preds, gts, iou_thres=0.5, save_dir=None):
    # Evaluate predictions and optionally save PR/ROC curves
    tp, conf, pred_cls, target_cls = [], [], [], []

    for i, (pred, gt) in enumerate(zip(preds, gts)):
        if pred is None or len(pred) == 0:
            if gt is not None:
                for target in gt:
                    target_cls.append(int(target[0]))
            continue

        pred = torch.tensor(pred)
        gt = torch.tensor(gt)

        correct = torch.zeros(pred.shape[0], dtype=torch.bool)

        tcls = gt[:, 0].tolist()
        tboxes = gt[:, 1:5]

        if len(tboxes):
            ious = bbox_iou(pred[:, :4], tboxes)
            x = torch.where(ious >= iou_thres)
            if x[0].numel():
                matches = torch.cat((torch.stack(x, 1), ious[x[0], x[1]].unsqueeze(1)), 1)
                matches = matches.cpu().numpy()
                if matches.shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

                correct[matches[:, 0].astype(int)] = True

        tp.extend(correct.numpy())
        conf.extend(pred[:, 4].numpy())
        pred_cls.extend(pred[:, 5].int().numpy())
        target_cls.extend(gt[:, 0].int().numpy())

    tp = np.array(tp)
    conf = np.array(conf)
    pred_cls = np.array(pred_cls)
    target_cls = np.array(target_cls)

    precision, recall, ap, ap_class = ap_per_class(tp, conf, pred_cls, target_cls)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save PR and ROC curves
        plot_pr_curve(tp, conf, pred_cls, target_cls, save_dir / "pr_curve.png")
        plot_roc_curve(tp, conf, pred_cls, target_cls, save_dir / "roc_curve.png")

    return precision, recall, ap, ap_class


def plot_pr_curve(tp, conf, pred_cls, target_cls, save_path):
    # Save precision-recall curve to file
    precision, recall, _ = precision_recall_curve(tp, conf)
    plt.figure()
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(tp, conf, pred_cls, target_cls, save_path):
    # Save ROC curve to file
    fpr, tpr, _ = roc_curve(tp, conf)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()