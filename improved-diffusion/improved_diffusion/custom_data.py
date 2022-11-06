import torch
from torch.utils.data import Dataset
import os
from natsort import natsorted
import cv2
import glob
import numpy as np
from PIL import Image
from skimage import io as img


import blobfile as bf




# New functions to match with SinGAN-Seg process

def make_4_chs_img(image_path, mask_path):
    im = img.imread(image_path)
    mask = img.imread(mask_path)

    # modifications - 22.02.2022
    mask = (mask > 127)*255 # to get clean mask
    # mask = 255 - (mask > 127)*255 # to get inverted mask
    #print(np.unique(mask))

    return np.concatenate((im, mask[:,:,0:1]), axis=2)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def np2torch(x):
    #if opt.nc_im == 3 or opt.nc_im == 4: # added opt.nc_im == 4 by vajira to handle 4 channel image
    x = x[:,:,:]
    x = x.transpose((2, 0, 1))/255
    
    x = torch.from_numpy(x)
    #if not(opt.not_cuda):
    #    x = move_to_gpu(x, opt.device)
    #x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    x = x.type(torch.FloatTensor)
    #x = x.type(torch.FloatTensor)
    x = norm(x)
    return x
    
def imresize(im,scale,opt):
    #s = im.shape
    #print("im shape===", im.shape)
    im = torch2uint8(im)
    im = imresize_in(im, scale_factor=scale)
    im = np2torch(im,opt)
    #im = im[:, :, 0:int(scale * s[2]), 0:int(scale * s[3])]
    return im


class ImageAndMaskDataFromSinGAN(Dataset):

    def __init__(self, img_dir, mask_dir,resolution, transform=None):

        
        self.images = natsorted(glob.glob(img_dir + "/*"))
        self.masks = natsorted(glob.glob(mask_dir + "/*"))

        self.imgs_and_masks = list(zip(self.images, self.masks))
        self.resolution = resolution
        self.transform = transform

    def __len__(self):

        return len(self.imgs_and_masks)

    def __getitem__(self, idx):

        data = self.imgs_and_masks[idx]
        # with bf.BlobFile(data, "rb") as f:
        #     pil_image = Image.open(f)
        #     pil_image.load()
        
       
        # ###################

        # while min(*pil_image.size) >= 2 * self.resolution:
        #     pil_image = pil_image.resize(
        #         tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        #     )

        # scale = self.resolution / min(*pil_image.size)
        # pil_image = pil_image.resize(
        #     tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        # )

        # arr = np.array(pil_image.convert("RGB"))
        # crop_y = (arr.shape[0] - self.resolution) // 2
        # crop_x = (arr.shape[1] - self.resolution) // 2
        # image_path = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]

        # ########
        # while min(*pil_image.size) >= 2 * self.resolution:
        #     pil_image = pil_image.resize(
        #         tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        #     )

        # scale = self.resolution / min(*pil_image.size)
        # pil_image = pil_image.resize(
        #     tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        # )


        # arr = np.array(pil_image.convert("RGB"))
        # crop_y = (arr.shape[0] - self.resolution) // 2
        # crop_x = (arr.shape[1] - self.resolution) // 2
        # mask_path = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]

        #####
        
        image_path = data[0] # image
        mask_path = data[1] # mask 

        sample = make_4_chs_img(image_path, mask_path)#Image.fromarray(sample)

        sample = np2torch(sample)

        sample = sample[0:4,:,:]

        if self.transform:
            sample = self.transform(sample)
            

        return sample




# if __name__ == "__main__":

#     # dataset = ImageAndMaskDataFromSinGAN("/work/vajira/DATA/kvasir_seg/real_images_root/real_images", 
#     #                             "/work/vajira/DATA/kvasir_seg/real_masks_root/real_masks")

#     # print(dataset[1].shape)

#     #cv2.imwrite("test.png", dataset[1])