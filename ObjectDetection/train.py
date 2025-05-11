import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.voc_dataset import VOCDataset
from models.ssd_mobilenet import SSD
from utils.transforms import get_transforms
from utils.box_utils import generate_ssd_priors, match_priors
from utils.loss import SSDLoss
import config

# Define collate_fn at the top level of the module for pickling
def collate_batch(batch):
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    images = torch.stack(images, dim=0)
    return images, targets

def train():
    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Load dataset
    train_dataset = VOCDataset(
        root=config.DATA_ROOT,
        image_set="trainval",
        transform=get_transforms(train=True, size=config.IMAGE_SIZE)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Dataset loaded with {len(train_dataset)} images for training.")
    if len(train_dataset) == 0:
        print("!!! ERROR: Dataset is empty. Check DATA_ROOT and image_set. !!!")
        return

    # Initialize model
    model = SSD(num_classes=config.NUM_CLASSES)
    model = model.to(device)

    # Generate prior boxes
    priors = generate_ssd_priors(
        image_size=config.IMAGE_SIZE,
        feature_maps=[19, 10, 5, 3],
        min_sizes=[30, 60, 111, 162],
        max_sizes=[60, 111, 162, 213],
        aspect_ratios=[[2,3], [2,3], [2,3], [2,3]],
        steps=[16, 32, 64, 100]
    ).to(device)
    print(f"Generated {priors.shape[0]} prior boxes.")

    # Loss and optimizer
    criterion = SSDLoss(neg_pos_ratio=3, num_classes=config.NUM_CLASSES)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    # Learning rate scheduler
    milestones = [round(config.NUM_EPOCHS * 0.66), round(config.NUM_EPOCHS * 0.9)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    print(f"Starting training for {config.NUM_EPOCHS} epochs...")

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            batch_size_actual = images.size(0)

            batch_loc_targets = torch.zeros((batch_size_actual, priors.size(0), 4), dtype=torch.float32, device=device)
            batch_cls_targets = torch.zeros((batch_size_actual, priors.size(0)), dtype=torch.long, device=device)

            for idx_in_batch in range(batch_size_actual):
                gt_boxes = targets[idx_in_batch]["boxes"].to(device)
                gt_labels = targets[idx_in_batch]["labels"].to(device)

                if gt_boxes.numel() == 0:
                    continue

                loc_target_single_img, cls_target_single_img = match_priors(
                    gt_boxes, gt_labels, priors,
                    threshold=0.5,
                    image_size=config.IMAGE_SIZE
                )
                
                batch_loc_targets[idx_in_batch] = loc_target_single_img
                batch_cls_targets[idx_in_batch] = cls_target_single_img
            
            optimizer.zero_grad()
            loc_preds, conf_preds = model(images)
            loss = criterion((loc_preds, conf_preds), (batch_loc_targets, batch_cls_targets))
            loss_item = loss.item()

            if torch.isnan(loss).any():
                print(f"!!!!!!!!!!!!!! LOSS IS NAN at Epoch {epoch+1}, Step {i+1}! Stopping. !!!!!!!!!!!!!!")
                return

            loss.backward()
            optimizer.step()

            total_loss += loss_item

            if (i + 1) % 10 == 0 or i == len(train_loader) - 1:
                print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Batch Loss: {loss_item:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] completed. Average Training Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == config.NUM_EPOCHS - 1:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"ssd_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
    final_model_path = os.path.join(config.CHECKPOINT_DIR, "ssd.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed for {config.NUM_EPOCHS} epochs. Final model saved: {final_model_path}")

if __name__ == "__main__":
    train()