import numpy as np
import torch
from skimage import io, transform
from torchvision import transforms as T, utils
from PIL import Image

# all transform inputs are of shape Slices,x,x where x is at least 256 and should be numpy arrays
def train_transform(x):
    normalize = T.Normalize(mean=[58.09, 58.09, 58.09],
                             std=[49.73, 49.73, 49.73],
                           inplace = True)
    resize = T.Resize(256)
    i, j, h, w = T.RandomCrop.get_params(Image.new('RGB', (256, 256)), output_size=(224, 224))
    hflip, vflip = np.random.choice([True, False]), np.random.choice([True, False])
    angle = np.random.normal()*5
    
    # convert to right shape and type
    x = np.uint8(x/16)
    x = np.moveaxis(x,1,-1)
    slices = []
    for s in range(x.shape[0]):
        sl = Image.fromarray(x[s])
        sl = resize(sl)
        sl = T.functional.crop(sl, i, j, h, w)
        if hflip: sl = T.functional.hflip(sl)
        if vflip: sl = T.functional.vflip(sl)
        sl = T.functional.rotate(sl, angle)
        sl = np.asarray(sl.convert('RGB'))
        sl = np.moveaxis(sl, -1, 0)
        sl = torch.FloatTensor(np.copy(sl))
        normalize(sl)
        slices.append(sl)
    x = torch.stack(slices)
    return x
    
def test_transform(x):
    normalize = T.Normalize(mean=[58.09, 58.09, 58.09],
                             std=[49.73, 49.73, 49.73],
                           inplace = True)
    resize = T.Resize(256)
    center_crop = T.CenterCrop(224)
    
    # convert to right shape and type
    x = np.uint8(x/16)
    x = np.moveaxis(x,1,-1)
    
    slices = []
    for s in range(x.shape[0]):
        sl = Image.fromarray(x[s])
        sl = resize(sl)
        sl = center_crop(sl)
        sl = np.asarray(sl.convert('RGB'))
        sl = np.moveaxis(sl, -1, 0)
        sl = torch.FloatTensor(np.copy(sl))
        normalize(sl)
        slices.append(sl)
    x = torch.stack(slices)
    return x