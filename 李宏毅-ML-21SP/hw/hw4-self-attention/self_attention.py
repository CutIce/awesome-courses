import os
import json
import torch
import random

import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from tqdm.auto import tqdm

# Hyper Parameters
dropout = 0.2
valid_ratio = 0.1

model_path = './model.ckpt'
output_path = './pred.csv'
data_path = './data'


batch_size = 8
valid_steps = 2000
warmup_steps = 1000
save_steps = 10000
total_steps = 70000

weight_decay_l1 = 3e-6
weight_decay_l2 = 1e-5
# Functions


def get_device():
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Deivce : {d}")
    return d


def collate_batch(batch):
    mel, speaker = zip(*batch)
    mel = pad_sequence(mel, batch_first=True, padding_value=20)
    return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers=0):
    dataset = MyDataset(data_dir, segment_len=128)
    speaker_num = dataset.get_speaker_num()
    train_len = int((1 - valid_ratio) * len(dataset))
    lengths = [train_len, len(dataset) - train_len]

    train_set, valid_set = random_split(dataset, lengths)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0, pin_memory=True, collate_fn=collate_batch)

    return train_loader, valid_loader, speaker_num


class MyDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.segment_len = segment_len

        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path)

        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(metadata_path)

        self.speaker2id = mapping["speaker2id"]
        self.speaker_num = len(metadata.keys())
        self.data = []

        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

    def __getitem__(self, idx):
        feat_path, speaker = self.data[idx]

        mel = torch.load(os.path.join.join(self.data_dir, feat_path))

        if len(mel) > self.segment_len:
            start = random.randint(0, len(mel) - self.segment_len)
            mel = torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
        speaker = torch.FloatTensor([speaker]).long()

        return mel, speaker

    def __len__(self):
        return len(self.data)


    def get_speaker_num(self):
        return self.speaker_num


class Classifier(nn.Module):
    def __init__(self, d_model=80, n_spks=600):
        super(Classifier, self).__init__()

        self.pre_handle = nn.Linear(40, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, n_spks),
        )

    def forward(self, mels):
        """
        arg: (batch size, length, 40)

        return (batch size, n_spks)
        """
        y = self.pre_handle(mels)  # y: (batch size, length, d_model)

        y = y.permute(1, 0, 2)     # y: (length, batch size, d_model)

        y = self.encoder_layer(y)  # y: (batch size, length, d_model)

        y = y.transpose(0, 1)

        stats = y.mean(dim=1)

        out = self.fc_layer(stats)
        return out


def calc_loss(model, w_l1=0.0, w_l2=0.0):
    l1 = 0
    l2 = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1 += torch.sum(torch.abs(param))
            l2 += torch.sum(torch.pow(param, 2))
    l1 *= w_l1
    l2 *= w_l2
    return l1, l2


def model_fn(batch, model, criterion, device):
    mels, labels = batch
    out = model(mels.to(device))
    ce_loss = criterion(out, labels.to(device))
    l1, l2 = calc_loss(model, w_l1=weight_decay_l1, w_l2=weight_decay_l2)
    all_loss = ce_loss + l1 + l2

    preds = out.argmax(1)

    acc = torch.mean((preds == labels.to(device)).float())

    return all_loss, ce_loss, l1, l2, acc


def get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int,
                                    num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):

    def lr_lambda(current_steps):
        # Warm Up!
        if current_steps < num_warmup_steps:
            return float(current_steps) / float(max(1, num_warmup_steps))
        # Decadence
        progress = float(current_steps - num_warmup_steps) / float(max(1, num_training_steps-num_warmup_steps))

        return max(1e-6, 0.5*(1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)


def valid(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc='Valid', unit=' uttr')

    for i,batch in enumerate(dataloader):
        with torch.no_grad():
            all_loss, ce_loss, l1_loss, l2_loss, acc = model_fn(batch, model, criterion, device)
            running_loss += all_loss
            running_accuracy += acc.item()

            pbar.update(dataloader.batch_size)
            pbar.set_postfix(
                loss=f"{running_loss / (i+1):.2f}",
                accuracy=f"{running_accuracy / (i+1):.2f}",
            )
        pbar.close()
        model.train()

        return running_accuracy / len(dataloader)

def train():
    device = get_device()
    train_loader, valid_loader, speaker_num = get_dataloader(data_dir=data_path, batch_size=batch_size)
    print(f"[INFO] Finish Loading data!", flush=True)

    model = Classifier(n_spks=speaker_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[INFO] Finish Creating Model, Setting criterion, optimizer and scheduler", flush=True)

    best_acc = 0.0

    pbar = tqdm(total=total_steps, ncols=0, desc='Train', unit=" step")

    step = 0

    all_loss_in_steps = []
    ce_loss_in_steps = []
    l1_loss_in_steps = []
    l2_loss_in_steps = []
    accs_in_steps = []
    valid_accs = []
    valid_loss = []

    while step < total_steps:
        try:
            batch = next(train_loader)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        all_loss, ce_loss, l1_loss, l2_loss, acc = model_fn(batch, model, criterion, device)

        all_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        batch_loss = ce_loss.item()
        batch_acc = acc.item()

        all_loss_in_steps.append(all_loss)
        ce_loss_in_steps.append(ce_loss.item())
        l1_loss_in_steps.append(l1_loss)
        l2_loss_in_steps.append(l2_loss)
        accs_in_steps.append(acc.item())



        pbar.update()
        pbar.set_postfix(
            loss=f"all loss = {all_loss}, ce loss = {ce_loss.item()}, l1 = {l1_loss}, l2 = {l2_loss}",
            accuracy=f"{acc.item()}",
            step=step+1
        )

        if (step+1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = valid(valid_loader, model, criterion, device)

            if valid_accuracy > best_acc:
                best_acc = valid_accuracy
                torch.save(model.state_dict(), './model.ckpt')
                pbar.write(f"Step {step+1}, saving model with acc {best_acc:.4f}")
            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=' step')
    pbar.close()
