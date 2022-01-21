from albumentations.augmentations.transforms import VerticalFlip
import numpy
from app.configs import DefaultConfig, SegmentationConfig
from app.loss import SegCrossEntropyLoss
from app.models import Unet
from app.data import BasicSegDataset, CityScapeDataset, get_loaders_cityscape
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from app.utils import (load_checkpoint, save_checkpoint, check_accuracy
                       )  # , check_accuracy, save_predictions_as_imgs)
import matplotlib.pyplot as plt

torch.backends.cudnn.enabled = False
device = "cuda" if torch.cuda.is_available() else "cpu"
test_batch_size = 4

test_img_dir = "dataset/data/"
test_mask_dir = "dataset/mask/"
num_epochs = 400
lr = 0.02
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
num_worker = 6
img_height = 128
img_width = 256
size = (img_height, img_width)
pin_memory = False
load_model = False
train_img_dir = "dataset/data/"
train_mask_dir = "dataset/mask/"
val_img_dir = "dataset/data/"
val_mask_dir = "dataset/mask/"
root_path = "/media/liangdao/DATA/cityscapes"

train_transform = A.Compose([
    A.Resize(height=img_height, width=img_width),
    A.Rotate(limit=35, p=1),
    A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.1),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(height=img_height, width=img_width),
    A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
    ),
    ToTensorV2(),
])


def make_prediction():
    train_loader, val_loader = get_loaders_cityscape(root_path, batch_size, size,
                                                     train_transform,
                                                     val_transform, num_worker,
                                                     pin_memory)

    checkpoint = "/home/liangdao/Documents/DNN/res_checkpoint_best.pth"
    model = Unet(in_channel=3, num_classes=34).to(device)
    load_checkpoint(torch.load(checkpoint), model)
    X, Y = next(iter(val_loader))
    X, Y = X.to(device), Y.to(device)
    Y_pred = model(X)
    print(Y_pred.shape)
    Y_pred = torch.argmax(Y_pred, dim=1)
    print(Y_pred.shape)

    inverse_transform = A.Compose([
        A.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
    ])
    fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5))

    for i in range(test_batch_size):
        # landscape = inverse_transform(image=X[i].permute(1, 2, 0).cpu().detach().numpy())
        image = X[i].permute(1, 2, 0).cpu().detach().numpy()
        label_class = Y[i].cpu().detach().numpy()
        label_class_predicted = Y_pred[i].cpu().detach().numpy()
        axes[i, 0].imshow(image)
        axes[i, 0].set_title("Landscape")
        axes[i, 1].imshow(label_class)
        axes[i, 1].set_title("Label Class")
        axes[i, 2].imshow(label_class_predicted)
        axes[i, 2].set_title("Label Class - Predicted")
    plt.show()

    
if __name__ == "__main__":
    make_prediction()
