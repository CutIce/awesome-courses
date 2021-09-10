# import part
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

from tqdm.auto import tqdm


train_tsfm = transforms.Compose([
    # reseize the image to a fixed shape
    transforms.Resize((128, 128)),
    # you can add some transforms here

    # the last layer should be ToTensor()
    transforms.ToTensor(),
])

# we don't need to augmentation in testing and validation
# All we need here is to resize the PIL image and transform it into Tensor
test_tsfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Hyperparameters
batch_size = 64



# dataset
train_set = DatasetFolder('food-11/training/labeled', loader=lambda x: Image.open(x), extensions='jpg', transform=train_tsfm)
unlabeled_set = DatasetFolder('food-11/training/unlabeled', loader=lambda x: Image.open(x), extensions='jpg', transform=train_tsfm)
valid_set = DatasetFolder('food-11/validation', loader=lambda x: Image.open(x), extensions='jpg', transform=test_tsfm)
test_set = DatasetFolder('food-11/testing', loader=lambda x: Image.open(x), extensions='jpg', transform=test_tsfm)

# data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)



