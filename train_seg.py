from albumentations.augmentations.transforms import VerticalFlip
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
# hyperparameters
num_epochs = 400
lr = 0.02
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
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
# end tode: configurize


def train_step(loader, model, optimizer, loss_fn, scaler, epoch):
    loop = tqdm(loader)
    model.train()
    loss_epoch = 0
    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.long().to(device=device)
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn._compute_loss(predictions, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_description("Epoch %i" % epoch)
        loop.set_postfix(loss=loss.item())
        loop.update(1)
        loss_epoch += loss.item()
    return loss_epoch/(len(loader))


def main():
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

    model = Unet(in_channel=3, num_classes=34).to(device)
    loss_fn = SegCrossEntropyLoss(ignore_nan_targets=False)
    optimizer = optim.Adam(model.parameters(),
                           lr=lr)  # can be down with app.optimizer

    # train_loader, val_loader = get_loaders(train_img_dir, train_mask_dir,
    #                                       val_img_dir, val_mask_dir,
    #                                       batch_size, train_transform,
    #                                       val_transform, num_worker,
    #                                        pin_memory)

    train_loader, val_loader = get_loaders_cityscape(root_path, batch_size, size,
                                                     train_transform,
                                                     val_transform, num_worker,
                                                     pin_memory)

    if load_model:
        load_checkpoint(torch.load("xxx.pth"), model)

    scaler = torch.cuda.amp.GradScaler() # Automatic Mixed precision

    for epoch in range(num_epochs):
        # training each epoch
        epoch_losses = []
        loss_epoch = train_step(train_loader, model, optimizer, loss_fn, scaler, epoch)
        # save each epoch result
        epoch_losses.append(loss_epoch)
        check_point = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(check_point)
        # check_accuracy(val_loader, model, device=device)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(epoch_losses)


if __name__ == "__main__":
    main()

    # configs = SegmentationConfig()

    # print(configs.net_config.num_classes)
    # net = Unet(configs.net_config)
    # print(net)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = net.to(device)

    # check keras-like model summary using torchsummary
    # summary(model, input_size=(3, 224, 224))