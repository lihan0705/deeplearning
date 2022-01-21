"""
basic dataset class for segmentation task
"""

import os
# from albumentations import augmentations
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
# from torchvision import transforms
import torch


class BasicSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None) -> None:
        super(BasicSegDataset, self).__init__()
        self.img_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.imgs = os.listdir(image_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.imgs[index])
        mask_path = os.path.join(self.mask_dir, self.imgs[index].replace(
            ".jpg", "_mask.jpg"))  # depends on different dataset
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        return image, mask


class CityScapeDataset(Dataset):
    """
    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
        Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
        Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
        Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
        Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
        Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
        Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
        Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
        Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
        Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
        Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
        Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
        Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
        Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
        Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
        Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
        Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
        Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
        Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
        Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
        Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
        Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
        Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
        Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
        Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
        Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
        Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
        Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
        Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
        Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]
    """
    def __init__(self, root, flag, size, transform=None) -> None:
        super(CityScapeDataset, self).__init__()
        self.transform = transform
        self.flag = flag
        self.imgs, self.masks, _ = self.get_image_index(root)
        self.size = size
        self.mapping = {
                    0: 0,  # unlabeled
                    1: 1,  # ego vehicle
                    2: 2,  # rect border
                    3: 3,  # out of roi
                    4: 4,  # static
                    5: 5,  # dynamic
                    6: 6,  # ground
                    7: 7,  # road
                    8: 8,  # sidewalk
                    9: 9,  # parking
                    10: 10,  # rail track
                    11: 11,  # building
                    12: 12,  # wall
                    13: 13,  # fence
                    14: 14,  # guard rail
                    15: 15,  # bridge
                    16: 16,  # tunnel
                    17: 17,  # pole
                    18: 18,  # polegroup
                    19: 19,  # traffic light
                    20: 20,  # traffic sign
                    21: 21,  # vegetation
                    22: 22,  # terrain
                    23: 23,  # sky
                    24: 24,  # person
                    25: 25,  # rider
                    26: 26,  # car
                    27: 27,  # truck
                    28: 28,  # bus
                    29: 29,  # caravan
                    30: 30,  # trailer
                    31: 31,  # train
                    32: 32,  # motorcycle
                    33: 33,  # bicycle
                    -1: 0  # licenseplate
                }

        self.mappingrgb = {
            0: (255, 0, 0),  # unlabeled
            1: (255, 0, 0),  # ego vehicle
            2: (255, 0, 0),  # rect border
            3: (255, 0, 0),  # out of roi
            4: (255, 0, 0),  # static
            5: (255, 0, 0),  # dynamic
            6: (255, 0, 0),  # ground
            7: (0, 255, 0),  # road
            8: (255, 0, 0),  # sidewalk
            9: (255, 0, 0),  # parking
            10: (255, 0, 0),  # rail track
            11: (255, 0, 0),  # building
            12: (255, 0, 0),  # wall
            13: (255, 0, 0),  # fence
            14: (255, 0, 0),  # guard rail
            15: (255, 0, 0),  # bridge
            16: (255, 0, 0),  # tunnel
            17: (255, 0, 0),  # pole
            18: (255, 0, 0),  # polegroup
            19: (255, 0, 0),  # traffic light
            20: (255, 0, 0),  # traffic sign
            21: (255, 0, 0),  # vegetation
            22: (255, 0, 0),  # terrain
            23: (0, 0, 255),  # sky
            24: (255, 0, 0),  # person
            25: (255, 0, 0),  # rider
            26: (255, 255, 0),  # car
            27: (255, 0, 0),  # truck
            28: (255, 0, 0),  # bus
            29: (255, 0, 0),  # caravan
            30: (255, 0, 0),  # trailer
            31: (255, 0, 0),  # train
            32: (255, 0, 0),  # motorcycle
            33: (255, 0, 0),  # bicycle
            -1: (255, 0, 0)  # licenseplate
        }

    def mask_to_class(self, mask):
        maskimg = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mapping:
            maskimg[mask == k] = self.mapping[k]
        return maskimg

    def mask_to_rgb(self, mask):
        rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
        for k in self.mappingrgb:
            rgbimg[0][mask == k] = self.mappingrgb[k][0]
            rgbimg[1][mask == k] = self.mappingrgb[k][1]
            rgbimg[2][mask == k] = self.mappingrgb[k][2]
        return rgbimg

    def get_image_index(self, root):
        images = []
        labels = []
        images_labels_vis = []
        if self.flag == "train":
            cities = os.listdir(root +
                                "/leftImg8bit_trainvaltest/leftImg8bit/train/")
            for city in cities:
                sample_path = os.listdir(
                    root + "/leftImg8bit_trainvaltest/leftImg8bit/train/" +
                    city)

                for smaple in sample_path:
                    if os.path.exists(
                            root +
                            "/gtFine_trainvaltest/gtFine_trainvaltest/gtFine/train/"
                            + city + "/" +
                            smaple.split("_leftImg8bit.png")[0] +
                            "_gtFine_labelIds.png"):
                        images.append(
                            root +
                            "/leftImg8bit_trainvaltest/leftImg8bit/train/" +
                            city + "/" + smaple)
                        labels.append(
                            root +
                            "/gtFine_trainvaltest/gtFine_trainvaltest/gtFine/train/"
                            + city + "/" +
                            smaple.split("_leftImg8bit.png")[0] +
                            "_gtFine_labelIds.png")
                        images_labels_vis.append(
                            root +
                            "/gtFine_trainvaltest/gtFine_trainvaltest/gtFine/train/"
                            + city + "/" +
                            smaple.split("_leftImg8bit.png")[0] +
                            "_gtFine_color.png")
                    else:
                        continue
        if self.flag == "val":
            cities = os.listdir(root +
                                "/leftImg8bit_trainvaltest/leftImg8bit/val/")
            for city in cities:
                sample_path = os.listdir(
                    root + "/leftImg8bit_trainvaltest/leftImg8bit/val/" +
                    city)
                for smaple in sample_path:
                    if os.path.exists(
                            root +
                            "/gtFine_trainvaltest/gtFine_trainvaltest/gtFine/val/"
                            + city + "/" +
                            smaple.split("_leftImg8bit.png")[0] +
                            "_gtFine_labelIds.png"):
                        images.append(
                            root + "/leftImg8bit_trainvaltest/leftImg8bit/val/" +
                            city + "/" + smaple)
                        labels.append(
                            root +
                            "/gtFine_trainvaltest/gtFine_trainvaltest/gtFine/val/"
                            + city + "/" + smaple.split("_leftImg8bit.png")[0] +
                            "_gtFine_labelIds.png")
                        images_labels_vis.append(
                            root +
                            "/gtFine_trainvaltest/gtFine_trainvaltest/gtFine/val/"
                            + city + "/" + smaple.split("_leftImg8bit.png")[0] +
                            "_gtFine_color.png")
                    else:
                        continue

        return images, labels, images_labels_vis

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image = Image.open(self.imgs[index]).convert('RGB')
        mask = Image.open(self.masks[index]).convert('L')

        image = TF.resize(image, size=self.size, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, size=self.size, interpolation=Image.NEAREST)

        image = np.array(image)
        mask = np.array(mask, dtype=np.uint8)
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
        return image, self.mask_to_class(mask)


def get_loaders(train_dir,
                train_maskdir,
                val_dir,
                val_maskdir,
                batch_size,
                train_transform,
                val_trainsform,
                num_workers=4,
                pin_memory=True):

    train_ds = BasicSegDataset(image_dir=train_dir,
                               mask_dir=train_maskdir,
                               train_transform=train_transform)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)

    val_ds = BasicSegDataset(image_dir=val_dir,
                             mask_dir=val_maskdir,
                             train_transform=val_trainsform)

    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=True)
    return train_loader, val_loader


def get_loaders_cityscape(root,
                          batch_size,
                          size,
                          train_transform,
                          val_trainsform,
                          num_workers=4,
                          pin_memory=True):

    train_ds = CityScapeDataset(root, flag="train", size=size, transform=train_transform)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True)

    val_ds = CityScapeDataset(root, flag="val", size=size, transform=val_trainsform)

    val_loader = DataLoader(val_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=True)
    return train_loader, val_loader
