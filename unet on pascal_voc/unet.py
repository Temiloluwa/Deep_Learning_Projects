import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import itertools
import operator
import time
from PIL import Image
from .utils import load_npy, save_npy, tensor_to_numpy, calculate_iou, concat_img
from matplotlib import pyplot as plt

IMG_WIDTH = 252 #508 change to desired size
IMG_HEIGHT = 188 #380 #change to desired size
BATCH_SIZE = 2
EPOCHS = 100
LEARNING_RATE = 1e-3
VALIDATION_INTERVAL = 2
DISPLAY_STATS_INTERVAL = 1
BASE_CONV = 1 #64
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transform_image:
    """Performs data preprocessing steps on input images

    Args:
        img_width (int): Input image width
        img_height (int): Input image height

    """
    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def __call__(self, im, mask=False):
        """Implements the data preprocessing

        Args:
            im (np.ndarray): image

        Returns:
            im (np.ndarray): numpy array containing image data
        """
        h, w, _ = im.shape
        im = Image.fromarray(im)
        if h > w:
            im = im.transpose(method=Image.ROTATE_270)

        if mask:
            im = im.resize((self.img_width, self.img_height), resample=0)
        else:
            im = im.resize((self.img_width, self.img_height))
        
        im = np.array(im)
        im = np.transpose(im, (2, 0, 1))
        if mask:
            im = im[0] #select just one channel]s
            return im
        im = im.astype(np.float32)
        im = im/255.0
        return im

class Dataloader:
    transformer = Transform_image(IMG_WIDTH, IMG_HEIGHT)
    data_dir = "./pascal_voc_seg"
    idx = 0
    def __init__(self, data_group:str, bs:int, shuffle=False):
        self.data_group = data_group
        self.shuffle = shuffle
        self.bs = bs

    def __iter__(self):
        self.data = os.listdir(os.path.join(self.data_dir, self.data_group, "images"))
        if self.shuffle:
            np.random.shuffle(self.data)
        self.images_path = [os.path.join(self.data_dir, self.data_group, "images", i) for i in self.data]
        self.masks_path = [os.path.join(self.data_dir, self.data_group, "masks", i) for i in self.data]
        self.masks_path = [i.replace("jpg", "npy") for i in self.masks_path] 
        return self

    def __next__(self):
        imgs_path = self.images_path[self.idx*self.bs: min((self.idx+1)*self.bs, len(self.images_path))]
        masks_path = self.masks_path[self.idx*self.bs: min((self.idx+1)*self.bs, len(self.images_path))]
        if len(imgs_path) == 0:
            raise StopIteration 
        imgs = np.array([self.transformer(plt.imread(i)) for i in imgs_path])
        masks = np.array([self.transformer(load_npy(i), mask=True) for i in masks_path])
        imgs = torch.from_numpy(imgs).to(device)
        masks = torch.from_numpy(masks).to(device)
        self.idx += 1
        
        return imgs, masks


def double_conv(cin, cout):
    conv = nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(cout, cout, kernel_size=3),
        nn.ReLU(),
    )
    return conv

def decoder_conv(cin, cout):
    conv = nn.Sequential(
        nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2),
        nn.ReLU(),
        nn.Conv2d(cout, cout, kernel_size=3),
        nn.ReLU(),
        nn.Conv2d(cout, cout, kernel_size=3),
        nn.ReLU()
    )
    return conv


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder_conv_1 = double_conv(3, BASE_CONV)
        self.encoder_conv_2 = double_conv(BASE_CONV, BASE_CONV*2)
        self.encoder_conv_3 = double_conv(BASE_CONV*2, BASE_CONV*4)
        self.encoder_conv_4 = double_conv(BASE_CONV*4, BASE_CONV*8)
        self.encoder_conv_5 = double_conv(BASE_CONV*8, BASE_CONV*16)
        self.decoder_conv_1 = decoder_conv(BASE_CONV*16 + BASE_CONV*8, BASE_CONV*8)
        self.decoder_conv_2 = decoder_conv(BASE_CONV*8 + BASE_CONV*4, BASE_CONV*4)
        self.decoder_conv_3 = decoder_conv(BASE_CONV*4 + BASE_CONV*2, BASE_CONV*2)
        self.decoder_conv_4 = decoder_conv(BASE_CONV*2 + BASE_CONV, BASE_CONV)
        self.conv_final = nn.Conv2d(BASE_CONV, 21, kernel_size=1)
        

    def forward(self, img):
        x1 = self.encoder_conv_1(img)
        x1 = self.max_pool_2x2(x1)
        x2 = self.encoder_conv_2(x1)
        x2 = self.max_pool_2x2(x2)
        x3 = self.encoder_conv_3(x2)
        x3 = self.max_pool_2x2(x3)
        x4 = self.encoder_conv_4(x3)
        x4 = self.max_pool_2x2(x4)
        x5 = self.encoder_conv_5(x4)
        x6 = self.decoder_conv_1(concat_img(x5, x4))
        x7 = self.decoder_conv_2(concat_img(x6, x3))
        x8 = self.decoder_conv_3(concat_img(x7, x2))
        x9 = self.decoder_conv_4(concat_img(x8, x1))
        x10 = self.conv_final(x9)
        return x10

if __name__ == "__main__":
    train_stats = []
    val_stats = []
    model = Unet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        losses = []
        ious = []
        for img, mask in Dataloader("train", BATCH_SIZE):
            pred_mask = model(img)
            mask = concat_img(pred_mask, mask, crop_only=True)
            loss_value = loss_fn(pred_mask, mask.long())
            losses.append(loss_value)
            ious.append(calculate_iou(pred_mask, mask))

        loss = torch.sum(torch.stack(losses))
        mean_iou = np.mean(ious)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_epoch_time = time.time() - epoch_start
        train_stats.append({
            "epoch": epoch,
            "train_epoch_time": train_epoch_time/60,
            "train_loss":loss.item(),
            "train_mean_iou": mean_iou
        })

        if epoch == 1:
            print("imgs shape", img.size())
            print("masks shape", mask.size())

        if (epoch % VALIDATION_INTERVAL) == 0:
            losses = []
            ious = []
            for img, mask in Dataloader("val", BATCH_SIZE):
                pred_mask = model(img)
                mask = concat_img(pred_mask, mask, crop_only=True)
                loss = loss_fn(pred_mask, mask.long())
                losses.append(loss)
                ious.append(calculate_iou(pred_mask, mask))

            loss = torch.sum(torch.stack(losses))
            mean_iou = np.mean(ious)

            val_stats.append({
                "epoch": epoch,
                "val_epoch_time": (time.time() - train_epoch_time)/60,
                "val_loss":loss.item(),
                "val_mean_iou": mean_iou
            })

        if (epoch % DISPLAY_STATS_INTERVAL) == 0:
            print(f"train stats: epoch-{epoch:03d}, loss:{train_stats[-1]['train_loss']:3f}, mean_iou:{train_stats[-1]['train_mean_iou']:3f}")
            if (epoch % VALIDATION_INTERVAL) == 0:
                print(f"val stats: epoch-{epoch:03d}, loss:{val_stats[-1]['val_loss']:3f}, mean_iou:{val_stats[-1]['val_mean_iou']:3f}\n")
        







