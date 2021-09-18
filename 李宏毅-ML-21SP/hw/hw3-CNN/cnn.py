# import part
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder

from tqdm.auto import tqdm


train_tsfm = transforms.Compose([
    # reseize the image to a fixed shape
    transforms.RandomResizedCrop((128, 128)),
    # you can add some transforms here
    transforms.RandomChoice([
        transforms.AutoAugment(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN)
    ]),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    # transforms.RandomErasing(p=0.2, scale=(0.01, 0.1), ratio=(0.3, 3.3)),
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
renew = True
lr_decline = False
do_demi = True

is_predict = True
is_train = False

batch_size = 16
epoches = 10

model_path = './model.ckpt'
learning_rate = 0.00001
momentum = 0.9

weight_decay_l1 = 1e-5
weight_decay_l2 = 1e-3

early_stop = 500


# dataset
train_set = DatasetFolder('food-11/training/labeled', loader=lambda x: Image.open(x), extensions='jpg', transform=train_tsfm)
unlabeled_set = DatasetFolder('food-11/training/unlabeled', loader=lambda x: Image.open(x), extensions='jpg', transform=train_tsfm)
valid_set = DatasetFolder('food-11/validation', loader=lambda x: Image.open(x), extensions='jpg', transform=test_tsfm)
test_set = DatasetFolder('food-11/testing', loader=lambda x: Image.open(x), extensions='jpg', transform=test_tsfm)

# data loader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
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

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(1024 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 11),
        )

    def forward(self, x):
        x = self.cnn_layers(x)

        x = x.flatten(1)

        x = self.fc_layers(x)

        return x

    def calc_loss(self, y_hat, labels, w_l1=0., w_l2=0.):
        l, l1, l2 = 0, 0, 0
        l = self.criterion(y_hat, labels)

        for name, param in self.named_parameters():
            # if name in ['weight']:
            #     l1 += torch.sum(torch.abs(param))
            #     l2 += torch.sum(torch.pow(param, 2))
            if 'weight' in name:
                l1 += torch.sum(torch.abs(param))
                l2 += torch.sum(torch.pow(param, 2))

        l1 *= w_l1
        l2 *= w_l2

        return l+l1+l2, l, l1, l2

def get_device():
    d = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"device: {d}")
    return d


class UnlabeledDataset(Dataset):
    def __init__(self, img_lst, label_list):
        # assert len(img_lst) == label_list, "NOT Equal!"
        self.size = len(img_lst)

        self.data = torch.cat(([img_lst[i] for i in range(self.size)]), 0)
        del img_lst
        self.label = [label for labels in label_list for label in labels]
        del label_list

        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.len


def get_pseudo_labels(dataset, model, threshold=0.8):
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    softmax = nn.Softmax(dim=-1)

    img_lst = []
    labels_lst = []

    for i, batch in enumerate(data_loader):
        img, _ = batch

        with torch.no_grad():
            logits = model(img.to(device))

        probs = softmax(logits)

        scores, labels = probs.max(dim=-1)

        scores, labels = scores.cpu().numpy(), labels.cpu().numpy()

        filter = scores > threshold

        img_lst.append(img[filter])
        labels_lst.append(labels[filter])

    dataset = UnlabeledDataset(img_lst, labels_lst)

    del img_lst, labels_lst, data_loader, scores, labels, filter

    model.train()
    return dataset

# Visualize Part
import matplotlib.pyplot as plt
import numpy as np

def plot(train_loss_record, train_acc_record, valid_loss_record, valid_acc_record):
    plt.figure(1)
    plt.subplot(1, 2, 1)
    x = np.arange(len(train_loss_record))
    plt.plot(x, train_loss_record, color='blue', label='Train')
    plt.plot(x, valid_loss_record, color='red', label='Valid')
    plt.legend(loc="upper right")
    plt.title('Loss Figure')
    del x

    plt.figure(1)
    plt.subplot(1, 2, 2)
    x = np.arange(train_acc_record)
    plt.plot(x, train_acc_record, color='blue', label='Train')
    plt.plot(x, valid_acc_record, color='red', label='Valid')
    plt.legend(loc="upper right")
    plt.title('Acc Figure')

    plt.savefig('./lossAndAcc.png')

device = get_device()

model = Classifier().to(device)
model.device = device
if renew:
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)

if is_train:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    epoch = 0
    early_stop_cnt = 0

    train_epoch_accs = []
    train_epoch_loss = []

    valid_epoch_accs = []
    valid_epoch_loss = []

    while epoch < epoches:

        if do_demi and best_acc >= 0.7:
            pseudo_set = get_pseudo_labels(unlabeled_set, model, threshold=0.7)
            concat_dataset = ConcatDataset([train_set, pseudo_set])
            train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

            tmp = (len(train_loader) - 193) * batch_size
            print("epoch: ", epoch, "Length of pseudo data", tmp)

        model.train()
        train_all_loss = []
        train_loss = []
        train_l1_loss = []
        train_l2_loss = []
        train_accs = []

        for i, batch in enumerate(train_loader):
            imgs, labels = batch

            logits = model(imgs.to(device))

            all_loss, loss, l1_loss, l2_loss = model.calc_loss(logits, labels.to(device), weight_decay_l1, weight_decay_l2)

            optimizer.zero_grad()

            all_loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_all_loss.append(all_loss.item())
            train_loss.append(loss.item())
            train_l1_loss.append(l1_loss)
            train_l2_loss.append(l2_loss)

            train_accs.append(acc)


        model.eval()

        valid_loss = []
        valid_accs = []

        del imgs, labels

        for i, batch in enumerate(valid_loader):
            imgs, labels = batch

            with torch.no_grad():
                logits = model(imgs.to(device))

            all_loss, loss, l1_loss, l2_loss = model.calc_loss(logits, labels.to(device), weight_decay_l1, weight_decay_l2)

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)


        # calc the avg loss and acc in training and valid
        train_loss_avg = sum(train_loss) / len(train_loss)
        train_all_loss_avg = sum(train_all_loss) / len(train_all_loss)
        train_l1_loss_avg = sum(train_l1_loss) / len(train_l1_loss)
        train_l2_loss_avg = sum(train_l2_loss) / len(train_l2_loss)
        train_accs_avg = sum(train_accs) / len(train_accs)

        valid_loss_avg = sum(valid_loss) / len(valid_loss)
        valid_accs_avg = sum(valid_accs) / len(valid_accs)

        if valid_accs_avg > best_acc:
            best_acc = valid_accs_avg
            torch.save(model.state_dict(), model_path)
            print(f"\nsaving model with acc: {valid_accs_avg}")
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        print(f"\n[ {epoch+1:03d}/{epoches:03d} ]: Train: all_loss = {train_all_loss_avg:.5f}, loss = {train_loss_avg:.5f}, l1 = {train_l1_loss_avg:.5f}, l2 = {train_l2_loss_avg:.5f}")
        print(f"             Valid: loss = {valid_loss_avg}")
        print(f"             Train Acc: {train_accs_avg:.5f}, Valid Acc: {valid_accs_avg}")

        epoch += 1
        train_epoch_loss.append(train_loss_avg)
        train_epoch_accs.append(train_accs_avg)
        valid_epoch_loss.append(valid_loss_avg)
        valid_epoch_accs.append(valid_accs_avg)
        
        if early_stop_cnt == early_stop:
            print("Early Stop !!!")
            break

    # plot(train_epoch_loss, valid_epoch_loss, train_epoch_accs, valid_epoch_accs)





if is_predict:

    model.eval()

    predictions = []

    for i, batch in enumerate(test_loader):
        imgs, labels = batch

        with torch.no_grad():
            logits = model(imgs.to(device))

        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

    with open("./predict.csv", 'w') as f:
        f.write("Id,Category\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")
    print("Finish Prediction!")
    del imgs, labels


# get_pseudo_labels(unlabeled_set, model, 0.8)

del model

