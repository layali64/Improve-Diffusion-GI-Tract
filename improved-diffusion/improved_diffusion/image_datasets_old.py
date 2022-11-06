import torch
from torch.utils.data import Dataset
import os
from natsort import natsorted
import cv2
import glob
import numpy as np
from PIL import Image
from skimage import io as img

class ImageAndMaskData(Dataset):

    def __init__(self, img_dir, mask_dir, transform=None):

        
        self.images = natsorted(glob.glob(img_dir + "/*"))
        self.masks = natsorted(glob.glob(mask_dir + "/*"))

        self.imgs_and_masks = list(zip(self.images, self.masks))

        self.transform = transform

    def __len__(self):

        return len(self.imgs_and_masks)

    def __getitem__(self, idx):

        data = self.imgs_and_masks[idx]

        img_path = data[0] # image
        mask_path = data[1] # mask 

        #img = cv2.imread(img_path)
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))[:,:,0:1] # take only one channel from mask
        #print(mask.shape)
        #print(mask.sum())

        sample = np.concatenate((img, mask), axis=2)
        #sample = torch.tensor(sample).to(torch.float)

        #sample = img

        sample = Image.fromarray(sample)
        
        #sample = sample.permute((2, 0, 1))

        # convert to 0,1 range
        #sample = sample/255


        #print(sample.shape)

        #print(img.shape)
        #print(mask.shape)
        if self.transform:
            sample = self.transform(sample)
            


        return sample


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



# class ImageAndMaskDataFromSinGAN(Dataset):

#     def __init__(self,resolution, img_dir, mask_dir,transform=None,classes=None, shard=0, num_shards=1):

        
#         self.images = natsorted(glob.glob(img_dir + "/*"))
#         self.masks = natsorted(glob.glob(mask_dir + "/*"))

#         self.imgs_and_masks = list(zip(self.images, self.masks))
#         self.resolution = resolution

#         self.transform = transform

#     def __len__(self):

#         return len(self.imgs_and_masks)

#     def __getitem__(self, idx):

#         data = self.imgs_and_masks[idx]
#         image_path = data[0] # image
#         mask_path = data[1] # mask 

#         sample = make_4_chs_img(image_path, mask_path)#Image.fromarray(sample)

#         sample = np2torch(sample)
        
#         sample = sample[0:4,:,:]
        
        

#         with bf.BlobFile(sample, "rb") as f:
#             pil_image = Image.open(f)
#             pil_image.load()

        
#          # We are not on a new enough PIL to support the `reducing_gap`
#         # argument, which uses BOX downsampling at powers of two first.
#         # Thus, we do it by hand to improve downsample quality.
#         while min(*pil_image.size) >= 2 * self.resolution:
#             pil_image = pil_image.resize(
#                 tuple(x // 2 for x in pil_image.size), resample=Image.BOX
#             )

#         scale = self.resolution / min(*pil_image.size)
#         pil_image = pil_image.resize(
#             tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
#         )

#         arr = np.array(pil_image.convert("RGBA"))
#         #arr = np.array(pil_image.convert("RGB"))
#         crop_y = (arr.shape[0] - self.resolution) // 2
#         crop_x = (arr.shape[1] - self.resolution) // 2
#         arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
#         arr = arr.astype(np.float32) / 127.5 - 1

#         out_dict = {}
#         if self.local_classes is not None:
#             out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        

      


#         if self.transform:
#             sample = self.transform(sample)
            


#         #return sample

#         return np.transpose(arr, [2, 0, 1]), out_dict






from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset


def load_data(
    *, img_dir,mask_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not img_dir or not mask_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(img_dir)
    classes = None
    if class_cond:
        #Assume classes are the first part of the filename,
        #before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
      
    # dataset = ImageAndMaskDataFromSinGAN ( image_size,img_dir,mask_dir,
    #                                         classes=classes,
    #                                         shard=MPI.COMM_WORLD.Get_rank(),
    #                                         num_shards=MPI.COMM_WORLD.Get_size())

    dataset = ImageDataset(
        image_size,
        #all_files,
        img_dir,
        mask_dir,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )  

    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths,mask_paths, transform=None, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        ######
        self.local_mask = mask_paths[shard:][::num_shards]
        self.imgs_and_masks = list(zip(self.local_images, self.local_mask))
        ######
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.transform = transform

    def __len__(self):
        return len(self.imgs_and_masks)

    def __getitem__(self, idx):
        data = self.imgs_and_masks[idx]
        image_path = data[0] # image
        mask_path = data[1] # mask 
        sample = make_4_chs_img(image_path, mask_path) #Image.fromarray(sample)

        sample = np2torch(sample)

        sample = sample[0:4,:,:]
        path = self.sample[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f) 
            pil_image.load()

#####
        

        #####

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGBA"))
        #arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1



        # out_dict = {}
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        #return np.transpose(arr, [2, 0, 1]), out_dict
        return np.transpose(arr, [2, 0, 1])#, out_dict
        #return sample



   