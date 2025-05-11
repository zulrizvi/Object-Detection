# app.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T # Using torchvision transforms for preprocessing here
import os
import io # For handling byte streams from uploaded files

# --- Assuming your project structure allows these imports ---
# You might need to adjust paths if app.py is not in the ObjectDetection root
# or ensure your ObjectDetection folder is in PYTHONPATH
try:
    from models.ssd_mobilenet import SSD
    from utils.box_utils import decode_boxes, nms, generate_ssd_priors
    import config # Your config.py
except ImportError as e:
    st.error(f"Error importing project modules: {e}. Make sure app.py is in the correct location or PYTHONPATH is set.")
    st.stop()


# --- Constants and Helper Functions (adapted from inference.py) ---
VOC_CLASSES = [
   "aeroplane", "bicycle", "bird", "boat", "bottle",
   "bus", "car", "cat", "chair", "cow", "diningtable",
   "dog", "horse", "motorbike", "person", "pottedplant",
   "sheep", "sofa", "train", "tvmonitor"
]
DEVICE = torch.device(config.DEVICE) # Use device from config

# --- Model Loading (Cached for performance) ---
@st.cache_resource # Use st.cache_resource for models/non-data objects
def load_model_cached(checkpoint_path):
    model_obj = SSD(num_classes=config.NUM_CLASSES)
    try:
        # Ensure map_location is used correctly
        model_obj.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model_obj.to(DEVICE)
        model_obj.eval()
        return model_obj
    except FileNotFoundError:
        st.error(f"Checkpoint file not found at {checkpoint_path}. Please train the model and place ssd.pth in the checkpoints directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Preprocessing Function (from inference.py, using torchvision.transforms) ---
def preprocess_image_for_streamlit(image_pil, image_size_config):
    transform_pipeline = T.Compose([
        T.Resize((image_size_config, image_size_config)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform_pipeline(image_pil)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    return image_tensor

# --- Drawing Function (from inference.py) ---
def draw_pred_boxes_streamlit(image_pil, boxes_pixels, labels_idx, scores_val):
    # ... (Copy the draw_pred_boxes_inference function here, rename if desired) ...
    # Ensure it uses PIL ImageDraw and ImageFont
    # Example:
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
            # st.warning(f"Skipping invalid box for drawing: {box_list}") # Optional warning
            continue
        draw.rectangle(box_list, outline='red', width=3)
        
        text_size = draw.textbbox((0,0), text, font=font) # Pillow 9.3.0+
        # For older Pillow: text_w, text_h = draw.textsize(text, font=font)
        text_x = box_list[0]
        text_y = box_list[1] - (text_size[3] - text_size[1]) - 2 # text_h -2
        
        if text_y < 0: text_y = box_list[1] + 2 
        textbox_location = [text_x, text_y, text_x + (text_size[2]-text_size[0]), text_y + (text_size[3]-text_size[1])]
        # For older Pillow: textbox_location = [text_x, text_y, text_x + text_w, text_y + text_h]
        draw.rectangle(textbox_location, fill='red')
        draw.text((text_x, text_y), text, fill='white', font=font)
    return image_pil


# --- Inference Function (adapted from inference.py) ---
# Priors can be generated once and cached or passed if they depend on dynamic config
@st.cache_data # Cache priors as they don't change if config.IMAGE_SIZE is fixed
def get_priors_cached(image_size_cfg):
    return generate_ssd_priors(image_size=image_size_cfg).to(DEVICE)

def run_inference_streamlit(model_obj, image_pil_original, conf_thresh=0.3, nms_thresh=0.45):
    if model_obj is None:
        return image_pil_original # Or an error image

    # 1. Preprocess
    image_tensor = preprocess_image_for_streamlit(image_pil_original, config.IMAGE_SIZE)
    priors = get_priors_cached(config.IMAGE_SIZE)

    # 2. Model Inference
    with torch.no_grad():
        locs, confs = model_obj(image_tensor)

    # 3. Decode and Post-process
    loc_preds_single = locs[0]
    conf_preds_single = confs[0]
    decoded_boxes_normalized = decode_boxes(loc_preds_single, priors) # Expects normalized priors
    
    class_scores_all = torch.softmax(conf_preds_single, dim=-1)
    fg_class_scores, fg_class_labels = class_scores_all[:, 1:].max(dim=1)

    mask = fg_class_scores > conf_thresh
    selected_boxes_normalized = decoded_boxes_normalized[mask]
    selected_scores = fg_class_scores[mask]
    selected_labels = fg_class_labels[mask]

    if selected_boxes_normalized.numel() == 0:
        st.info("No objects detected above the confidence threshold.")
        return image_pil_original

    keep_indices = nms(selected_boxes_normalized, selected_scores, threshold=nms_thresh)
    final_boxes_normalized = selected_boxes_normalized[keep_indices]
    final_labels = selected_labels[keep_indices]
    final_scores = selected_scores[keep_indices]

    # 4. Scale boxes to original image size for drawing
    ow, oh = image_pil_original.size
    final_boxes_pixels = final_boxes_normalized.clone().cpu()
    final_boxes_pixels[:, [0, 2]] *= ow
    final_boxes_pixels[:, [1, 3]] *= oh
    final_boxes_pixels[:, 0].clamp_(min=0, max=ow -1)
    final_boxes_pixels[:, 1].clamp_(min=0, max=oh -1)
    final_boxes_pixels[:, 2].clamp_(min=0, max=ow -1)
    final_boxes_pixels[:, 3].clamp_(min=0, max=oh -1)

    # 5. Draw boxes
    result_image_pil = draw_pred_boxes_streamlit(image_pil_original.copy(), final_boxes_pixels, final_labels, final_scores)
    return result_image_pil

# --- Streamlit App UI ---
def main():
    st.set_page_config(page_title="SSD Object Detector", layout="wide")
    st.title("🖼️ SSD MobileNetV2 Object Detector")
    st.markdown("Upload an image to detect objects from the VOC dataset classes.")

    # Load Model
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "ssd.pth")
    model = load_model_cached(checkpoint_path)

    if model is None:
        st.warning("Model could not be loaded. Please check the console for errors and ensure `checkpoints/ssd.pth` exists.")
        return

    # --- Sidebar for controls ---
    st.sidebar.header("⚙️ Detection Controls")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.01)
    nms_threshold = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.01)

    # --- File Uploader ---
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # To read image file buffer
        image_bytes = uploaded_file.getvalue()
        try:
            original_image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            st.error(f"Error opening image: {e}")
            return

        st.subheader("Uploaded Image")
        st.image(original_image_pil, caption="Original Image", use_column_width=True)

        # Perform inference when button is clicked
        if st.button("🔍 Detect Objects"):
            with st.spinner("Detecting objects..."):
                result_image = run_inference_streamlit(model, original_image_pil,
                                                        conf_thresh=conf_threshold,
                                                        nms_thresh=nms_threshold)
            st.subheader("Detection Results")
            st.image(result_image, caption="Image with Detections", use_column_width=True)
    else:
        st.info("Please upload an image file.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed for the Object Detection Project.")

if __name__ == '__main__':
    main()