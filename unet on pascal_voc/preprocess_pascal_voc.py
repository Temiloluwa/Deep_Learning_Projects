import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
import itertools
from .utils import read_file, save_npy, load_npy
from PIL import Image

# color patterns for various masks
class_colors = {(128,0,0):1, #aeroplane
                (0,128,0):2, #bicycle
                (128,128,0):3, #bird
                (0,0,128):4, #boat
                (128,0,128):5, #bottle
                (0,128,128):6, #bus
                (128,128,128):7, #car
                (64,0,0):8, #cat
                (192,0,0):9, #chair
                (64,128,0):10, #cow
                (192,128,0):11, #diningtable
                (64,0,128):12, #dog
                (192,0,128):13, #horse
                (64,128,128):14, #motorbike
                (192,128,128):15, #person
                (0,64,0):16, #pottedplant
                (128,64,0):17, #sheep
                (0,192,0):18, #sofa
                (128,192,0):19, #train
                (0,64,128):20} #tvmonitor


def generate_data():
    train_or_val_dict = {"train":read_file(os.path.join(voc_imgsets, "train.txt")),
                         "val":read_file(os.path.join(voc_imgsets, "val.txt"))}
    total_data = 0
    for train_or_val, train_or_val_list in train_or_val_dict.items():
        print(f"processing {train_or_val} data")
        dest_img_dir = os.path.join("pascal_voc_seg", f"{train_or_val}", "images")
        dest_mask_dir = os.path.join("pascal_voc_seg", f"{train_or_val}", "masks")
      
        if not os.path.exists(dest_mask_dir):
            os.makedirs(dest_img_dir)
            os.makedirs(dest_mask_dir)
        
        for fn in train_or_val_list:
            img_path = os.path.join(imgs_root, f"{fn}.jpg")
            mask_path = os.path.join(masks_root, f"{fn}.png")
            im = plt.imread(mask_path)
            h, w, _ = im.shape
            output_mask = np.zeros_like(im, dtype=np.uint8)
            for i, j in itertools.product(range(h), range(w)):
                pixel_color = tuple((im[i,j]*255).astype(np.uint8))
                if pixel_color in class_colors:
                    _color = class_colors[pixel_color]
                    output_mask[i,j] = [_color, _color, _color]
            
            save_npy(output_mask, os.path.join(dest_mask_dir, f"{fn}"))
            shutil.copy2(img_path, os.path.join(dest_img_dir, f"{fn}.jpg"))
            
        total_data += len(os.listdir(dest_img_dir))
        print(f"Total {train_or_val}: {len(os.listdir(dest_img_dir))}")
    print(f"Total Images: {total_data}")
    
imgs_root = os.path.join("VOCdevkit", "VOC2012", "JPEGImages")
masks_root = os.path.join("VOCdevkit", "VOC2012", "SegmentationClass")
voc_imgsets = os.path.join("VOCdevkit", "VOC2012", "ImageSets", "Segmentation")

if __name__ == "__main__":
    generate_data()
