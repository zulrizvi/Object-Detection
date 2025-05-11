# evaluate.py

import torch
from torch.utils.data import DataLoader
from datasets.voc_dataset import VOCDataset, VOC_CLASSES
from models.ssd_mobilenet import SSD
from utils.transforms import get_transforms
from utils.box_utils import decode_boxes, nms, generate_ssd_priors
import config
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torchvision.transforms as T

DENORM_MEAN = torch.tensor([0.485, 0.456, 0.406])
DENORM_STD = torch.tensor([0.229, 0.224, 0.225])

def denormalize_image(tensor_image):
    img = tensor_image.cpu().clone()
    mean = DENORM_MEAN.view(3, 1, 1)
    std = DENORM_STD.view(3, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return img

def load_model(checkpoint_path):
    model = SSD(num_classes=config.NUM_CLASSES)
    # Ensure map_location is properly used, especially if model trained on GPU and inferring on CPU
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(config.DEVICE)))
    model.to(config.DEVICE)
    model.eval()
    return model

def draw_pred_boxes(image_pil, boxes_pixels, labels_idx, scores_val):
    """
    Draws predicted bounding boxes on a PIL image.
    Args:
        image_pil: PIL Image to draw on.
        boxes_pixels: Tensor of bounding boxes [N, 4] in (xmin, ymin, xmax, ymax) PIXEL coordinates.
        labels_idx: Tensor of class indices [N] (0 to num_classes-1).
        scores_val: Tensor of confidence scores [N].
    """
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, label_idx, score in zip(boxes_pixels, labels_idx, scores_val):
        class_name = VOC_CLASSES[label_idx.item()] # Assumes VOC_CLASSES is available
        text = f"{class_name}: {score:.2f}"
        
        box_list = box.tolist()
        # Ensure xmin < xmax and ymin < ymax before drawing
        if box_list[0] >= box_list[2] or box_list[1] >= box_list[3]:
            print(f"Skipping invalid box for drawing: {box_list}")
            continue

        draw.rectangle(box_list, outline='red', width=3)
        
        text_size = draw.textbbox((0,0), text, font=font)
        text_x = box_list[0]
        text_y = box_list[1] - (text_size[3] - text_size[1]) - 2
        
        if text_y < 0: 
            text_y = box_list[1] + 2

        textbox_location = [text_x, text_y, text_x + (text_size[2] - text_size[0]), text_y + (text_size[3] - text_size[1])]
        draw.rectangle(textbox_location, fill='red')
        draw.text((text_x, text_y), text, fill='white', font=font)
    return image_pil

def eval_collate_fn(batch):
    images = []
    targets = [] # Targets might not be used in basic evaluation but good to collate
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    images = torch.stack(images, dim=0)
    return images, targets


def evaluate(model, dataloader, visualize=False, conf_thresh=0.3, nms_thresh=0.45):
    priors = generate_ssd_priors(image_size=config.IMAGE_SIZE).to(config.DEVICE)
    model.eval()

    with torch.no_grad():
        for batch_idx, (images_batch, _) in enumerate(dataloader):
            images_batch = images_batch.to(config.DEVICE)

            locs_batch, confs_batch = model(images_batch)

            for i in range(images_batch.size(0)): 
                loc_preds_single = locs_batch[i] 
                conf_preds_single = confs_batch[i] 

                decoded_boxes_normalized = decode_boxes(loc_preds_single, priors)
                
                class_scores_all = torch.softmax(conf_preds_single, dim=-1) 

                fg_class_scores, fg_class_labels = class_scores_all[:, 1:].max(dim=1)
                
                mask = fg_class_scores > conf_thresh
                
                selected_boxes_normalized = decoded_boxes_normalized[mask]
                selected_scores = fg_class_scores[mask]
                selected_labels = fg_class_labels[mask] # Labels are 0 to num_classes-1

                if selected_boxes_normalized.numel() == 0:
                    if visualize:
                        print(f"No detections above threshold for image {i} in batch {batch_idx}.")
                        denormalized_img_tensor = denormalize_image(images_batch[i])
                        image_pil_display = T.ToPILImage()(denormalized_img_tensor)
                        plt.imshow(image_pil_display)
                        plt.title(f"Image {i}, Batch {batch_idx} (No Detections)")
                        plt.axis('off')
                        plt.show()
                    continue

                # Apply NMS on NORMALIZED boxes
                keep_indices = nms(selected_boxes_normalized, selected_scores, threshold=nms_thresh)
                
                final_boxes_normalized = selected_boxes_normalized[keep_indices]
                final_labels = selected_labels[keep_indices]
                final_scores = selected_scores[keep_indices]

                if visualize:
                    # Denormalize the image tensor for display
                    # images_batch[i] is the input image tensor (e.g., 300x300)
                    denormalized_img_tensor = denormalize_image(images_batch[i])
                    image_pil_display = T.ToPILImage()(denormalized_img_tensor) # This is the 300x300 PIL image
                    
                    img_w, img_h = image_pil_display.size 
                    
                    final_boxes_pixels = final_boxes_normalized.clone().cpu()
                    final_boxes_pixels[:, [0, 2]] *= img_w  # Scale x-coordinates
                    final_boxes_pixels[:, [1, 3]] *= img_h  # Scale y-coordinates

                    # Clip to image boundaries just in case
                    final_boxes_pixels[:, 0].clamp_(min=0, max=img_w -1) # xmin
                    final_boxes_pixels[:, 1].clamp_(min=0, max=img_h -1) # ymin
                    final_boxes_pixels[:, 2].clamp_(min=0, max=img_w -1) # xmax
                    final_boxes_pixels[:, 3].clamp_(min=0, max=img_h -1) # ymax
                    
                    print(f"DEBUG: Drawing boxes for image {i}, batch {batch_idx}: {final_boxes_pixels.tolist()}")
                    print(f"DEBUG: Image size for drawing: {image_pil_display.size}")

                    output_image = draw_pred_boxes(image_pil_display, final_boxes_pixels, final_labels.cpu(), final_scores.cpu())
                    
                    plt.imshow(output_image)
                    plt.title(f"Image {i} in Batch {batch_idx} Predictions")
                    plt.axis('off')
                    plt.show()

if __name__ == "__main__":
    # Ensure a checkpoint exists
    checkpoint_file = os.path.join(config.CHECKPOINT_DIR, "ssd.pth") 
    if not os.path.exists(checkpoint_file):
        epoch_files = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith("ssd_epoch_") and f.endswith(".pth")]
        if epoch_files:
            epoch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True) # Sort by epoch number, descending
            checkpoint_file = os.path.join(config.CHECKPOINT_DIR, epoch_files[0])
            print(f"Final model 'ssd.pth' not found. Using latest epoch checkpoint: {checkpoint_file}")
        else:
            print(f"No checkpoint found in {config.CHECKPOINT_DIR}. Please train the model first.")
            exit()
    
    print(f"Loading model from: {checkpoint_file}")
    model = load_model(checkpoint_file)

    val_dataset = VOCDataset(root=config.DATA_ROOT, image_set='val', 
                             transform=get_transforms(train=False, size=config.IMAGE_SIZE))
    
    val_dataloader = DataLoader(val_dataset, batch_size=max(1, config.BATCH_SIZE // 4), 
                                shuffle=False, collate_fn=eval_collate_fn, num_workers=0) 

    print(f"Evaluating model on {len(val_dataset)} validation images...")
    evaluate(model, val_dataloader, visualize=True, conf_thresh=0.3) 