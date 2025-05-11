import torch
import math

def iou(box1, box2):

    inter_xmin = torch.max(box1[:, None, 0], box2[:, 0])
    inter_ymin = torch.max(box1[:, None, 1], box2[:, 1])
    inter_xmax = torch.min(box1[:, None, 2], box2[:, 2])
    inter_ymax = torch.min(box1[:, None, 3], box2[:, 3])

    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union_area = area1[:, None] + area2 - inter_area + 1e-6 

    return inter_area / union_area

def nms(boxes, scores, threshold=0.5):

    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
        
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1) 
    
    _, order = scores.sort(0, descending=True)
    
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        
        if order.numel() == 1:
            break
            
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        
        w = torch.maximum(torch.tensor(0.0, device=boxes.device), xx2 - xx1)
        h = torch.maximum(torch.tensor(0.0, device=boxes.device), yy2 - yy1)
        inter = w * h
       
        current_box_iou_input = boxes[i].unsqueeze(0)
        other_boxes_iou_input = boxes[order[1:]]
        
        overlap = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = torch.where(overlap <= threshold)[0]
        order = order[inds + 1] 

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def encode_boxes(gt_boxes_normalized_xyxy, priors_normalized_cxcywh, variances=[0.1, 0.2]):

    gt_cxcy_normalized = (gt_boxes_normalized_xyxy[:, :2] + gt_boxes_normalized_xyxy[:, 2:]) / 2
    gt_wh_normalized = gt_boxes_normalized_xyxy[:, 2:] - gt_boxes_normalized_xyxy[:, :2]

    prior_wh_eps = priors_normalized_cxcywh[:, 2:] + 1e-8 
    gt_wh_eps = gt_wh_normalized + 1e-8 

    encoded_cxcy = (gt_cxcy_normalized - priors_normalized_cxcywh[:, :2]) / (variances[0] * prior_wh_eps)
    encoded_wh = torch.log(gt_wh_eps / prior_wh_eps) / variances[1]

    return torch.cat([encoded_cxcy, encoded_wh], dim=1)

def decode_boxes(pred_loc, priors_normalized_cxcywh, variances=[0.1, 0.2]):
    
    decoded_cxcy = priors_normalized_cxcywh[:, :2] + pred_loc[:, :2] * variances[0] * priors_normalized_cxcywh[:, 2:]
    
    decoded_wh = priors_normalized_cxcywh[:, 2:] * torch.exp(pred_loc[:, 2:] * variances[1])

    decoded_boxes_xyxy_normalized = torch.cat([
        decoded_cxcy - decoded_wh / 2,  # xmin, ymin
        decoded_cxcy + decoded_wh / 2   # xmax, ymax
    ], dim=1)
    
    return decoded_boxes_xyxy_normalized


# Generate SSD priors (anchor boxes)
def generate_ssd_priors(image_size=300, 
                        feature_maps=[19, 10, 5, 3], 
                        min_sizes=[30, 60, 111, 162], 
                        max_sizes=[60, 111, 162, 213], 
                        aspect_ratios=[[2,3], [2,3], [2,3], [2,3]], 
                        steps=[16, 32, 64, 100], 
                        clip=True):
    priors = []
    for k, f_map_size in enumerate(feature_maps):
        for i in range(f_map_size):
            for j in range(f_map_size):
                cx = (j + 0.5) * steps[k] / image_size
                cy = (i + 0.5) * steps[k] / image_size
                
                s_k = min_sizes[k] / image_size
                priors.append([cx, cy, s_k, s_k])
                
                s_k_prime = math.sqrt(s_k * (max_sizes[k] / image_size))
                priors.append([cx, cy, s_k_prime, s_k_prime])
                
                for ar in aspect_ratios[k]:
                    priors.append([cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)])
                    priors.append([cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)])
    
    priors = torch.tensor(priors, dtype=torch.float32)
    if clip:
        priors.clamp_(min=0, max=1)
    return priors


# Match priors with ground truth boxes for training
def match_priors(boxes, labels, priors, threshold=0.5, image_size=300):
    if boxes.numel() == 0: 
        loc_targets = torch.zeros_like(priors) 
        cls_targets = torch.zeros(priors.size(0), dtype=torch.long, device=priors.device)
        return loc_targets, cls_targets

    normalized_gt_boxes = boxes / image_size 

    prior_boxes_normalized_xyxy = torch.cat((
        priors[:, :2] - priors[:, 2:] / 2,
        priors[:, :2] + priors[:, 2:] / 2 
    ), dim=1)
    
    ious = iou(normalized_gt_boxes, prior_boxes_normalized_xyxy)
    
    best_gt_ious_for_each_prior, best_gt_idx_for_each_prior = ious.max(dim=0)
    
    cls_targets = labels[best_gt_idx_for_each_prior] + 1 
    cls_targets[best_gt_ious_for_each_prior < threshold] = 0
    
    best_prior_ious_for_each_gt, best_prior_idx_for_each_gt = ious.max(dim=1)
    for gt_object_idx in range(normalized_gt_boxes.size(0)):
        prior_to_force_match = best_prior_idx_for_each_gt[gt_object_idx]
        best_gt_idx_for_each_prior[prior_to_force_match] = gt_object_idx
        cls_targets[prior_to_force_match] = labels[gt_object_idx] + 1

    assigned_gt_boxes_normalized = normalized_gt_boxes[best_gt_idx_for_each_prior]
    loc_targets = encode_boxes(assigned_gt_boxes_normalized, priors)
    
    loc_targets[cls_targets == 0] = 0 
    return loc_targets, cls_targets