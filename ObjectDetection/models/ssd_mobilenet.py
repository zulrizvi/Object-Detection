'''
loading MobileNetV2 pretrained on ImageNet
Remove the classifier head
Add extra convolutional layers for SSD feature maps
Add detection heads for class scores and bounding boxes
'''

import torch
import torch.nn as nn
import torchvision.models as models

class SSD(nn.Module):
    def __init__(self, num_classes=20):
        super(SSD, self).__init__()
        self.num_classes = num_classes
        

        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1) 
        self.backbone = mobilenet.features[:14]  
        
        self.extra1 = nn.Sequential( 
            nn.Conv2d(96, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.extra2 = nn.Sequential( 
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.extra3 = nn.Sequential( 
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=0), 
            nn.ReLU(inplace=True)
        )
        
        num_boxes = 6 
        self.loc_head = nn.ModuleList([
            nn.Conv2d(96, num_boxes * 4, kernel_size=3, padding=1),   
            nn.Conv2d(256, num_boxes * 4, kernel_size=3, padding=1),   
            nn.Conv2d(256, num_boxes * 4, kernel_size=3, padding=1),   
            nn.Conv2d(256, num_boxes * 4, kernel_size=3, padding=1)    
        ])
        
        self.cls_head = nn.ModuleList([
            nn.Conv2d(96, num_boxes * (num_classes + 1), kernel_size=3, padding=1),  
            nn.Conv2d(256, num_boxes * (num_classes + 1), kernel_size=3, padding=1),
            nn.Conv2d(256, num_boxes * (num_classes + 1), kernel_size=3, padding=1),
            nn.Conv2d(256, num_boxes * (num_classes + 1), kernel_size=3, padding=1)
        ])
        
    def forward(self, x):
        locs = []
        confs = []
        
        # Backbone feature map
        fmap0 = self.backbone(x) 
        feature_maps = [fmap0]
        
        # Extra feature maps
        fmap1 = self.extra1(fmap0)
        feature_maps.append(fmap1)
        
        fmap2 = self.extra2(fmap1)
        feature_maps.append(fmap2)
        
        fmap3 = self.extra3(fmap2)
        feature_maps.append(fmap3)
        
        # Applying the detection heads
        for i, fmap in enumerate(feature_maps):
            loc = self.loc_head[i](fmap)
            conf = self.cls_head[i](fmap)
            
            loc = loc.permute(0, 2, 3, 1).contiguous()
            conf = conf.permute(0, 2, 3, 1).contiguous()
            
            locs.append(loc.view(loc.size(0), -1, 4))
            confs.append(conf.view(conf.size(0), -1, self.num_classes + 1))  # +1 for background
        
        locs = torch.cat(locs, dim=1)
        confs = torch.cat(confs, dim=1)
        
        return locs, confs