# utility/metrics.py
import inspect
import warnings
import numpy as np
import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:   # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            if DIoU:
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


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


# ──────────────────────────────────────────────────────────────
# COCO-style mAP 계산기 (per-class + 전체)
# ──────────────────────────────────────────────────────────────
from utility.metrics import ap_per_class   # 기존 함수 재사용

def compute_coco_map(tp, conf, pred_cls, target_cls, iouv=None):
    """
    ⬇️  IoU set(iouv)별 Precision/Recall/AP 계산
    반환: (p50, r50, map50_vec, map5_95_vec, map75_vec, ap_class_index)
    - p50, r50      : Precision·Recall 벡터(클래스별) @ IoU 0.5
    - map50_vec     : 클래스별 AP  @ IoU 0.5
    - map5_95_vec   : 클래스별 AP  @ IoU 0.5:0.95 (COCO)
    - map75_vec     : 클래스별 AP  @ IoU 0.75
    - ap_class_index: 클래스 id 배열
    """
    if iouv is None:
        iouv = np.array([0.5 + i * 0.05 for i in range(10)])

    # --- IoU 0.5 고정 결과 -----------
    p50, r50, ap50_vec, ap_class = ap_per_class(tp, conf, pred_cls, target_cls, iouv[0])

    # --- 0.5:0.95 (COCO) ------------
    ap_mat = []           # [cls, iou_idx]
    for iou in iouv:
        _, _, ap_i, _ = ap_per_class(tp, conf, pred_cls, target_cls, iou)
        ap_mat.append(ap_i)
    ap_mat = np.stack(ap_mat, axis=1)     # shape (n_cls, 10)

    map5_95_vec = ap_mat.mean(1)
    map75_vec   = ap_mat[:, iouv.tolist().index(0.75)]

    return p50, r50, ap50_vec, map5_95_vec, map75_vec, ap_class


class EvalResults:      
    def __init__(self, box, speed=None):
        self.box = box
        self.speed = speed or {}


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



def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if isinstance(ratio_pad, (int, float)):
        warnings.warn(f"[DBG] bad call from {inspect.stack()[1].filename}:{inspect.stack()[1].lineno} "f"ratio_pad={ratio_pad!r}")
        raise TypeError(f"[scale_coords] ratio_pad must be tuple, not {type(ratio_pad)}")

    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        if len(ratio_pad) == 2 and isinstance(ratio_pad[0], (int, float)):
            gain, pad = ratio_pad            # (gain, (pad_w, pad_h))
        else:
            gain, pad = ratio_pad[0][0], ratio_pad[1]


    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords




class ConfusionMatrix:
    def __init__(self, nc):
        self.matrix = np.zeros((nc + 1, nc + 1))  # nc classes + background
        self.nc = nc

    def process_batch(self, detections, labels):
        """
        Updates confusion matrix given detections and labels for a batch.

        detections: tensor (nx6) = [x1, y1, x2, y2, conf, cls_pred]
        labels: tensor (mx5) = [cls_true, x1, y1, x2, y2]
        """
        detections = detections.cpu().numpy()
        labels = labels.cpu().numpy()
        gt_classes = labels[:, 0].astype(int)
        pred_classes = detections[:, 5].astype(int)

        iou = box_iou(torch.tensor(detections[:, :4]), torch.tensor(labels[:, 1:5]))
        matches = (iou > 0.5).nonzero(as_tuple=False)

        detected = set()
        for pred_idx, label_idx in matches:
            if label_idx.item() not in detected:
                self.matrix[pred_classes[pred_idx], gt_classes[label_idx]] += 1
                detected.add(label_idx.item())

        for i, gt in enumerate(gt_classes):
            if i not in detected:
                self.matrix[self.nc, gt] += 1  # false negative

        for i in range(len(detections)):
            if i not in matches[:, 0]:
                self.matrix[pred_classes[i], self.nc] += 1  # false positive

    def matrix_normalized(self):
        return self.matrix / (self.matrix.sum(0, keepdims=True) + 1e-6)

    def plot(self, normalize=True, save_path='confusion_matrix.png', names=()):
        import seaborn as sns
        import matplotlib.pyplot as plt

        array = self.matrix_normalized() if normalize else self.matrix
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(array, annot=True, fmt='.2f', square=True, cmap='Blues', cbar=False,
                    xticklabels=names + ['background'], yticklabels=names + ['background'])
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.tight_layout()
        plt.savefig(save_path)
        plt.close()


def non_max_suppression_v7(prediction, conf_thres=0.25, iou_thres=0.45):
    if isinstance(prediction, tuple): 
        print(f"[DEBUG] tuple contents: {[type(p) for p in prediction]}")
        prediction = prediction[0]  # 튜플 강제 분해

    if isinstance(prediction, torch.Tensor) and prediction.ndim == 3:
        prediction = [p for p in prediction] #[B, N, 5+c] -> B x [N, 5+C]

    output = []
    for pred in prediction:  #[N, 5+c] -> N x [5+C]
        obj_conf = pred[:, 4:5]  
        cls_conf, cls_id = pred[:, 5:].max(1, keepdim=True) 
        conf = obj_conf * cls_conf
        
        pred = torch.cat((pred[:, :4], conf, cls_id.float()), dim=1)  # → [x1 y1 x2 y2 conf cls]
        pred = pred[conf.view(-1) >= conf_thres]                      # conf 필터링

        if not pred.shape[0]:
            output.append(torch.zeros((0, 6), device=pred.device))
            continue

        keep = torchvision.ops.nms(pred[:, :4], pred[:, 4], iou_thres)
        output.append(pred[keep])
    return output

def non_max_suppression_v7_multilabel(prediction, conf_thres=0.25, iou_thres=0.45):
    output = []
    for pred in prediction:
        pred = pred[pred[:, 4] >= conf_thres]
        if not pred.shape[0]:
            output.append(torch.zeros((0, 6), device=pred.device))
            continue

        # Multi-label handling
        if pred.shape[1] > 6:  # if class scores exist beyond class id
            box = pred[:, :4]
            obj_conf = pred[:, 4].unsqueeze(1)
            class_conf, class_pred = pred[:, 5:].max(1, keepdim=True)
            conf = obj_conf * class_conf  # final confidence
            pred = torch.cat((box, conf, class_pred.float()), 1)

        boxes = pred[:, :4]
        scores = pred[:, 4]
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
        output.append(pred[keep])
    return output


def get_nms(version="v7"):
    if version == "v7":
        return non_max_suppression_v7
    elif version == "v7-multilabel":
        return non_max_suppression_v7_multilabel
    else:
        raise ValueError(f"Unsupported NMS version: {version}")
