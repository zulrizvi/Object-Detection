import torch
import torch.nn as nn
import torch.nn.functional as F

class SSDLoss(nn.Module):
 
    def __init__(self, neg_pos_ratio=3, num_classes=20): 
        super(SSDLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.num_classes = num_classes 
        
    def forward(self, predictions, targets):
      
        loc_preds, cls_preds = predictions
        loc_targets, cls_targets = targets
        
        batch_size = loc_preds.size(0)
        num_priors = loc_preds.size(1)
        

        pos_mask = cls_targets > 0 
        
        # Number of positive matches
        num_pos = pos_mask.sum(dim=1, keepdim=True) 
        
     
        pos_loc_preds = loc_preds[pos_mask] 
        pos_loc_targets = loc_targets[pos_mask] 
        
        if pos_loc_preds.numel() > 0:
            loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, reduction='sum')
        else:
            loc_loss = torch.tensor(0.0, device=loc_preds.device)

        conf_preds_flat = cls_preds.reshape(-1, self.num_classes + 1)
        conf_targets_flat = cls_targets.reshape(-1)

        # Compute cross entropy loss for all priors (without reduction)
        cross_entropy_loss = F.cross_entropy(conf_preds_flat, conf_targets_flat, reduction='none')
        cross_entropy_loss = cross_entropy_loss.view(batch_size, num_priors) # [batch_size, num_priors]

        # Create a mask for negative samples (background)
        neg_mask = (cls_targets == 0) # [batch_size, num_priors]

        # Loss for positive samples
        loss_c_pos = cross_entropy_loss[pos_mask].sum()


        loss_c_neg = cross_entropy_loss.clone()
        loss_c_neg[pos_mask] = 0. 
        loss_c_neg, _ = loss_c_neg.sort(dim=1, descending=True) # Sort losses for negatives

        num_neg_to_mine = self.neg_pos_ratio * num_pos.squeeze(1) # [batch_size]
        max_neg_available = neg_mask.sum(dim=1) # [batch_size]
        num_neg_to_mine = torch.min(num_neg_to_mine, max_neg_available)
  
        final_neg_loss = 0.
        for i in range(batch_size):
            # Get losses for actual negative samples for this image
            img_neg_losses = cross_entropy_loss[i][neg_mask[i]]
            
            # Sort these negative losses
            img_neg_losses_sorted, _ = img_neg_losses.sort(descending=True)
            
            # Select top num_neg_to_mine[i]
            num_hard_negs_for_img = int(num_neg_to_mine[i].item())
            if num_hard_negs_for_img > 0 and img_neg_losses_sorted.numel() > 0:
                final_neg_loss += img_neg_losses_sorted[:num_hard_negs_for_img].sum()

        cls_loss = loss_c_pos + final_neg_loss
        
        N = max(num_pos.sum().item(), 1.) # Avoid division by zero
        
        loc_loss /= N
        cls_loss /= N
        
        total_loss = loc_loss + cls_loss
        return total_loss