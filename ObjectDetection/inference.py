# inference.py

import os
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T 
from models.ssd_mobilenet import SSD
from utils.box_utils import decode_boxes, nms, generate_ssd_priors 
import config
import matplotlib.pyplot as plt


VOC_CLASSES = [
   "aeroplane", "bicycle", "bird", "boat", "bottle",
   "bus", "car", "cat", "chair", "cow", "diningtable",
   "dog", "horse", "motorbike", "person", "pottedplant",
   "sheep", "sofa", "train", "tvmonitor"
]

DENORM_MEAN = torch.tensor([0.485, 0.456, 0.406])
DENORM_STD = torch.tensor([0.229, 0.224, 0.225])

def denormalize_image_inference(tensor_image):
    img = tensor_image.cpu().clone()
    if img.ndim == 4 and img.shape[0] == 1:
        img = img.squeeze(0)
    mean = DENORM_MEAN.view(3, 1, 1)
    std = DENORM_STD.view(3, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    return img

def load_model(checkpoint_path):
    model = SSD(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(config.DEVICE)))
    model.to(config.DEVICE)
    model.eval()
    return model

# Draw boxes on image
def draw_pred_boxes_inference(image_pil, boxes_pixels, labels_idx, scores_val):
    draw = ImageDraw.Draw(image_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, label_idx, score in zip(boxes_pixels, labels_idx, scores_val):
        if label_idx < 0 or label_idx >= len(VOC_CLASSES):
            class_name = "Unknown"
        else:
            class_name = VOC_CLASSES[label_idx.item()]
        
        text = f"{class_name}: {score:.2f}"
        box_list = box.tolist()
        if box_list[0] >= box_list[2] or box_list[1] >= box_list[3]:
            print(f"Skipping invalid box for drawing in inference: {box_list}")
            continue
        draw.rectangle(box_list, outline='red', width=3)
        
        text_size = draw.textbbox((0,0), text, font=font)
        text_x = box_list[0]
        text_y = box_list[1] - (text_size[3] - text_size[1]) - 2
        
        if text_y < 0: text_y = box_list[1] + 2 
        textbox_location = [text_x, text_y, text_x + (text_size[2] - text_size[0]), text_y + (text_size[3] - text_size[1])]
        draw.rectangle(textbox_location, fill='red')
        draw.text((text_x, text_y), text, fill='white', font=font)
    return image_pil


def preprocess_image(image_path, image_size_config): # Renamed image_size to avoid conflict with T.Resize's arg
    image_pil = Image.open(image_path).convert("RGB")
    
    # Use standard torchvision transforms for inference preprocessing
    transform_pipeline = T.Compose([
        T.Resize((image_size_config, image_size_config)), # Resize to model's expected input size
        T.ToTensor(),                                     # Convert PIL Image to tensor [0,1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize
    ])
    
    image_tensor = transform_pipeline(image_pil)
    
    image_tensor = image_tensor.unsqueeze(0).to(config.DEVICE)
    
    return image_pil, image_tensor


# Inference function
def run_inference(model, image_tensor_batch, image_orig_pil, conf_thresh=0.3, nms_thresh=0.45):
    priors = generate_ssd_priors(image_size=config.IMAGE_SIZE).to(config.DEVICE)

    with torch.no_grad():
        locs, confs = model(image_tensor_batch) 

    # Predictions are for the first (and only) image in the batch
    loc_preds_single = locs[0]
    conf_preds_single = confs[0]

    # Decode boxes: output is NORMALIZED [0,1] (xmin, ymin, xmax, ymax)
    decoded_boxes_normalized = decode_boxes(loc_preds_single, priors)
    
    class_scores_all = torch.softmax(conf_preds_single, dim=-1)
    fg_class_scores, fg_class_labels = class_scores_all[:, 1:].max(dim=1)

    mask = fg_class_scores > conf_thresh
    
    selected_boxes_normalized = decoded_boxes_normalized[mask]
    selected_scores = fg_class_scores[mask]
    selected_labels = fg_class_labels[mask]

    if selected_boxes_normalized.numel() == 0:
        print(f"No objects detected above threshold in {image_orig_pil.filename if hasattr(image_orig_pil, 'filename') else 'image'}")
        return image_orig_pil 

    keep_indices = nms(selected_boxes_normalized, selected_scores, threshold=nms_thresh)
    
    final_boxes_normalized = selected_boxes_normalized[keep_indices]
    final_labels = selected_labels[keep_indices]
    final_scores = selected_scores[keep_indices]

    ow, oh = image_orig_pil.size 
    
    final_boxes_pixels = final_boxes_normalized.clone().cpu()
    final_boxes_pixels[:, [0, 2]] *= ow  
    final_boxes_pixels[:, [1, 3]] *= oh  
    
    # Clip to original image boundaries
    final_boxes_pixels[:, 0].clamp_(min=0, max=ow -1)
    final_boxes_pixels[:, 1].clamp_(min=0, max=oh -1)
    final_boxes_pixels[:, 2].clamp_(min=0, max=ow -1)
    final_boxes_pixels[:, 3].clamp_(min=0, max=oh -1)

    return draw_pred_boxes_inference(image_orig_pil, final_boxes_pixels, final_labels, final_scores)


if __name__ == "__main__":
    checkpoint_file = os.path.join(config.CHECKPOINT_DIR, "ssd.pth")
    if not os.path.exists(checkpoint_file):
        epoch_files = [f for f in os.listdir(config.CHECKPOINT_DIR) if f.startswith("ssd_epoch_") and f.endswith(".pth")]
        if epoch_files:
            epoch_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
            checkpoint_file = os.path.join(config.CHECKPOINT_DIR, epoch_files[0])
            print(f"Final 'ssd.pth' not found. Using latest epoch checkpoint: {checkpoint_file}")
        else:
            print(f"No checkpoint found in {config.CHECKPOINT_DIR}. Please train the model first.")
            exit()
            
    print(f"Loaded model from {checkpoint_file}")
    model = load_model(checkpoint_file)

    os.makedirs(config.TEST_IMAGE_DIR, exist_ok=True)
    if not os.listdir(config.TEST_IMAGE_DIR):
        print(f"No images found in {config.TEST_IMAGE_DIR}. Please add some test images.")
        exit()

    for fname in os.listdir(config.TEST_IMAGE_DIR):
        if not (fname.lower().endswith(".jpg") or fname.lower().endswith(".png") or fname.lower().endswith(".jpeg")):
            continue

        image_path = os.path.join(config.TEST_IMAGE_DIR, fname)
        print(f"Processing {image_path}...")
        
        original_pil_image, image_tensor = preprocess_image(image_path, config.IMAGE_SIZE)
        
        result_image_pil = run_inference(model, image_tensor, original_pil_image.copy(), conf_thresh=0.3)

        plt.figure(figsize=(10,10))
        plt.imshow(result_image_pil)
        plt.title(fname)
        plt.axis("off")
        plt.show()