import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    # Initialize the dataset
    def __init__(self, root, image_set='train', transform=None):
        self.root=root
        self.image_set=image_set
        self.transform= transform
        
        # Load the image set
        image_set_path = os.path.join(self.root, "ImageSets", "Main", f"{image_set}.txt")
        with open(image_set_path) as f:
            self.ids = [line.strip() for line in f]

        # Loading the image and the annotation directories
        self.image_dir = os.path.join(self.root, "JPEGImages")
        self.ann_dir = os.path.join(self.root, "Annotations")
    
    
    def __len__(self):
        return len(self.ids)
    
    ##getting single image at a time
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
        ann_path = os.path.join(self.ann_dir, f"{img_id}.xml")
        
        image = Image.open(img_path).convert("RGB")
        boxes, labels = self.parse_voc_xml(ann_path)
        
        #convert to tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels
        }
        
        if self.transform:
            image, target = self.transform(image, target)
            
        return image, target
    
    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot() 
        
        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            name = obj.find("name").text
            
            if name not in VOC_CLASSES:
                continue
            
            label = VOC_CLASSES.index(name)
            
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            boxes.append([xmin,ymin,xmax,ymax])
            labels.append(label)
            
        return boxes, labels
    
VOC_CLASSES = [
   "aeroplane", "bicycle", "bird", "boat", "bottle", 
   "bus", "car", "cat", "chair", "cow", "diningtable",
   "dog", "horse", "motorbike", "person", "pottedplant",
   "sheep", "sofa", "train", "tvmonitor"
   ]