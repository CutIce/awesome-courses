import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split

import os
import json
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

config = {
    'train_path': './data',
    'test_path': './data/testdata.json',
    'save_path': 'model.ckpt',
    'batch_size': 32,
    'n_workers': 8,
    'valid_steps': 2000,
    'warmup_steps': 1000,
    'save_steps': 10000,
    'total_steps': 70000,
    'early_step': 1000,
    'n_epoches': 3000,

    'output_path': 'output.csv'

}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'[INFO]: Use {device} now!')


def myDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        mapping_path = Path(data_dir) / "mapping.json"
        mapping = json.load(mapping_path.open())
        self.speaker2id = mapping['speaker2id']

        metadata_path = Path(data_dir) / "metadata.json"
        metadata = json.load(metadata_path.open())

        self.speaker_num = len(metadata.keys())
        self.data = []

        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["features_path"], self.speaker2id[speaker]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feat_path, speaker = self.data[idx]
