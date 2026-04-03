"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path

# Save the model to the target dir
def save_model(model: torch.nn.Module, target_dir: str, name: str):
    """
    Saves a PyTorch model to a target directory.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    check_point_name = f"{name}.pth"
    model_save_path = target_dir_path / check_point_name

    # Save the model state_dict()
    #print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def save_checkpoint(model, optimizer, scheduler, target_dir, name):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, f"{target_dir}/{name}.pth")


def load_model(model: torch.nn.Module, model_path: str, device: torch.device):
    """
    Loads a PyTorch model from a .pth file.
    """
    # Charge state dict
    state_dict = torch.load(model_path, map_location=device)

    model.load_state_dict(state_dict)

    model.eval()

    return model

# Plot the training curve
def plot_curve(results: dict, epochs: int):
    train_ious, val_ious = np.array(results["train_iou"]), np.array(results["val_iou"])
    train_losses, val_losses = np.array(results["train_loss"]), np.array(results["val_loss"])

    plt.plot(np.arange(epochs, step=1), train_losses, label='Train loss')
    plt.plot(np.arange(epochs, step=1), train_ious, label='Train IoU')
    plt.plot(np.arange(epochs, step=1), val_losses, label='Val loss')
    plt.plot(np.arange(epochs, step=1), val_ious, label='Val IoU')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

# helper function for image visualization
def display(name=None, path=None, save=False, **images):
    """
    Plot images in one row
    """
    # clear_output(wait=True)
    num_images = len(images)
    plt.figure(figsize=(12,12))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, num_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=15)
        plt.imshow(image)
    if save:
        plt.savefig(f"{path}.png")
    if name:
        plt.title(name)
    plt.show()
    
# Categorical Cross Entropy Loss
class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        return F.nll_loss(y_hat.log(), y.argmax(dim=1))

# Multiclass Dice Loss
class MultiDiceLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
    
    def dice_coef(self, y_pred, y_true, smooth=0.0001):

        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = torch.sum(y_true_f * y_pred_f)

        return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)

    def dice_coef_multiclass(self, y_pred, y_true, smooth=0.0001):
        dice=0

        for index in range(1, self.num_classes):
            dice += self.dice_coef(y_true[:,index,:,:], y_pred[:,index,:,:], smooth = 0.0001)

        return 1 - dice/(self.num_classes-1)

    def forward(self, y_pred, y_true):
        return self.dice_coef_multiclass(y_pred, y_true)

def np_to_tensor(img_np):
    img_np = img_np.astype(np.float32)
    img_np /= 255.0

    img_tensor = torch.from_numpy(img_np).unsqueeze(0)

    return img_tensor

def rgb_to_class(mask_rgb, color_map):
    """
    Convertit un masque RGB en indices de classes.
    color_map: dict {(R,G,B): class_id}
    """
    mask_class = np.zeros(mask_rgb.shape[:2], dtype=np.int64)  # (H, W)
    for color, class_id in color_map.items():
        match = np.all(mask_rgb == np.array(color), axis=-1)
        mask_class[match] = class_id
    return mask_class




def compute_iou(predicted_mask, ground_truth, device="cpu", num_classes=4):
    """Convertit les masques en one-hot et réutilise MeanIoU."""
    ground_truth = torch.tensor(ground_truth).long()  # (H, W)

    # Convertir en one-hot : (H, W) → (1, C, H, W)
    pred_one_hot = torch.nn.functional.one_hot(predicted_mask.long(), num_classes)  # (H, W, C)
    pred_one_hot = pred_one_hot.permute(2, 0, 1).unsqueeze(0).float().to(device)  # (1, C, H, W)

    true_one_hot = torch.nn.functional.one_hot(ground_truth, num_classes)  # (H, W, C)
    true_one_hot = true_one_hot.permute(2, 0, 1).unsqueeze(0).float().to(device)  # (1, C, H, W)

    metric = MeanIoU()
    return metric(pred_one_hot, true_one_hot).item()

# Mean IoU Score
class MeanIoU(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes
    
    def IoU_coef(self, y_pred, y_true, smooth=0.0001): 

        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = torch.sum(y_true_f * y_pred_f) 
        total = torch.sum(y_true_f + y_pred_f)
        union = total - intersection 
        
        return (intersection + smooth)/(union + smooth)

    def Mean_IoU(self, y_pred, y_true, smooth=0.0001):
        IoU_Score=0
        for index in range(1, self.num_classes):
            IoU_Score += self.IoU_coef(y_true[:,index,:,:], y_pred[:,index,:,:], smooth = 1)

        return IoU_Score/(self.num_classes-1)

    def forward(self, y_pred, y_true):
        #return self.Mean_IoU(torch.softmax(y_pred, dim=1), y_true)
        return self.Mean_IoU(y_pred, y_true)