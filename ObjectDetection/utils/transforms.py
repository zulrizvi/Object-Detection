import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize:
    def __init__(self, size): 
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        
    def __call__(self, image, target):
        # Original dimensions
        orig_width, orig_height = image.size
        
        # Resize image
        image = image.resize(self.size, Image.BILINEAR)
        
        # Adjust bounding boxes
        if target is not None and 'boxes' in target and target['boxes'].numel() > 0: # Check if boxes exist
            boxes = target['boxes']
            # Scale boxes
            scale_width = self.size[0] / orig_width
            scale_height = self.size[1] / orig_height
            
            # Apply scaling to boxes [xmin, ymin, xmax, ymax]
            boxes[:, 0] *= scale_width
            boxes[:, 1] *= scale_height
            boxes[:, 2] *= scale_width
            boxes[:, 3] *= scale_height
            
            target['boxes'] = boxes
            
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, image, target):
        if np.random.random() < self.prob:
            # Flip image
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Flip boxes
            if target is not None and 'boxes' in target and target['boxes'].numel() > 0: # Check if boxes exist
                boxes = target['boxes']
                width = image.width 
                
              
                xmin_old = boxes[:, 0].clone()
                xmax_old = boxes[:, 2].clone()
                boxes[:, 0] = width - xmax_old
                boxes[:, 2] = width - xmin_old
                target['boxes'] = boxes
                
        return image, target

class ToTensor:
    """Convert image to tensor and ensure target tensors are properly formatted"""
    def __call__(self, image, target):
        # Convert image to tensor
        image = T.ToTensor()(image)
            
        return image, target

class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        self.transform = T.Normalize(mean=mean, std=std)
        
    def __call__(self, image, target):
        image = self.transform(image)
        return image, target

def get_transforms(train=True, size=300):
    transforms_list = [] # Renamed to avoid conflict
    
    # Resize image and boxes
    transforms_list.append(Resize(size))
    
    if train:
        transforms_list.append(RandomHorizontalFlip())
    
    transforms_list.append(ToTensor())
    
    # Normalize
    transforms_list.append(Normalize())
    
    return Compose(transforms_list)