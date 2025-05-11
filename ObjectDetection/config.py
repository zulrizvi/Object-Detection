import os
import torch

#path congig
BASE_DIR = os.path.abspath(".")
DATA_ROOT = os.path.join(BASE_DIR, "data", "VOCdevkit", "VOC2012")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
TEST_IMAGE_DIR = os.path.join(BASE_DIR, "test_images")

##traiing config
NUM_CLASSES = 20
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 300 # For SSD300; change to 512 if using SSD512