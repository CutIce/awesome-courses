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
renew = False
lr_decline = False
do_demi = False

batch_size = 64
epoches = 50

model_path = './model.ckpt'
learning_rate = 0.001
momentum = 0.9

weight_decay_l1 = 1e-5
weight_decay_l2 = 1e-3



# dataset
train_set = DatasetFolder('food-11/training/labeled', loader=lambda x: Image.open(x), extensions='jpg', transform=train_tsfm)
unlabeled_set = DatasetFolder('food-11/training/unlabeled', loader=lambda x: Image.open(x), extensions='jpg', transform=train_tsfm)
valid_set = DatasetFolder('food-11/validation', loader=lambda x: Image.open(x), extensions='jpg', transform=test_tsfm)
test_set = DatasetFolder('food-11/testing', loader=lambda x: Image.open(x), extensions='jpg', transform=test_tsfm)

# data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


print("Length of Train set    :", len(train_set))
print("Length of Unlabeled set:", len(unlabeled_set))
print("Length of Valid Set    :", len(valid_set))
print("Length of Test Set     :", len(test_set))

print("Length of Train loader    :", len(train_loader))
print("Length of Valid loader    :", len(valid_loader))
print("Length of Test loader     :", len(test_loader))


# Model
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.criterion = nn.CrossEntropyLoss()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 11),
        )

    def forward(self, x):
        x = self.cnn_layers(x)

        x = x.flatten(1)

        x = self.fc_layers(x)

        return x


def get_device():
    d = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"device: {d}")
    return d


def calc_loss(model, w_l1=0, w_l2=0):
    l1 = 0
    l2 = 0
    for name, param in model.named_parameters():
        if name in ['weights']:
            l1 += torch.sum(abs(param))
            l2 += torch.sum(torch.pow(param, 2))
    l1 = l1 * w_l1
    l2 = l2 * w_l2
    return l1+l2, l1, l2


device = get_device()

model = Classifier().to(device)
model.device = device
if renew:
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


best_acc = 0.0
for epoch in range(epoches):

    model.train()
    train_loss = []
    train_accs = []

    for x in enumerate(train_loader):
        print(x.shape)


