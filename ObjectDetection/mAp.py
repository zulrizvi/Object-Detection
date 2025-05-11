from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
import os
import config
from datasets.voc_dataset import VOCDataset
import numpy as np
from utils.box_utils import decode_boxes, nms
from models.ssd_mobilenet import SSD
from utils.box_utils import generate_ssd_priors  # if not already generated
from utils.transforms import get_transforms
from torch.utils.data import DataLoader

def compute_map(model, dataloader, iou_threshold=0.5, score_threshold=0.5):
    model.eval()
    all_preds = [[] for _ in range(config.NUM_CLASSES)]
    all_gts = [[] for _ in range(config.NUM_CLASSES)]

    priors = generate_ssd_priors()  # shape: [num_priors, 4]
    priors = priors.to(config.DEVICE)

    with torch.no_grad():
        for images, targets in dataloader:
            images = torch.stack(images).to(config.DEVICE)
            locs, confs = model(images)

            for i in range(images.size(0)):
                boxes = decode_boxes(locs[i], priors, variances=[0.1, 0.2])
                scores = torch.softmax(confs[i], dim=-1)

                max_scores, labels = scores.max(dim=1)
                mask = max_scores > score_threshold

                final_boxes = boxes[mask]
                final_scores = max_scores[mask]
                final_labels = labels[mask]

                keep = nms(final_boxes, final_scores, threshold=0.45)
                final_boxes = final_boxes[keep]
                final_scores = final_scores[keep]
                final_labels = final_labels[keep]

                # Store predictions for each class
                for cls in range(1, config.NUM_CLASSES):  # skip background
                    cls_mask = final_labels == cls
                    all_preds[cls].extend(final_scores[cls_mask].cpu().tolist())

                # Store ground truths for each class
                for box, label in zip(targets[i]['boxes'], targets[i]['labels']):
                    all_gts[label].append(1.0)  # True positive for each ground truth object

    # Calculate Average Precision (AP) for each class
    aps = []
    for cls in range(1, config.NUM_CLASSES):
        if len(all_preds[cls]) == 0 or len(all_gts[cls]) == 0:
            continue
        y_true = [1] * len(all_gts[cls]) + [0] * (len(all_preds[cls]) - len(all_gts[cls]))
        y_scores = all_preds[cls]
        
        # Ensure the length of y_true matches y_scores
        if len(y_true) != len(y_scores):
            continue

        ap = average_precision_score(y_true[:len(y_scores)], y_scores)
        aps.append(ap)

    # Calculate mean Average Precision (mAP)
    mAP = np.mean(aps) if aps else 0.0
    print(f"mAP: {mAP:.4f}")
    return mAP

if __name__ == "__main__":
    model = SSD(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "ssd.pth"), map_location=config.DEVICE))
    model.to(config.DEVICE)

    val_dataset = VOCDataset(config.DATA_ROOT, image_set='val', transform=get_transforms(train=False))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    compute_map(model, val_loader)
