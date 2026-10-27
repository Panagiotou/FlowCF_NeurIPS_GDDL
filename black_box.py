"""The shared black box every method explains (paper Section 4.1).

A single MLP with 1000 hidden units, trained on the one-hot + min-max encoded
features, one per dataset. It is method-agnostic infrastructure: all baselines and
FlowCF load the same checkpoint, so their metrics are directly comparable.

    python main.py --dataname <name> --method blackbox --mode train
    -> data/<name>/black_box_mlp.pkl
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils_train import preprocess, TabularDataset

HIDDEN_SIZE = 1000
EPOCHS = 100
BATCH_SIZE = 4096


class BBMLPCLF(nn.Module):
    def __init__(self, input_size, output_size=2, hidden_size=HIDDEN_SIZE, return_logits=False):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Training uses logits (CrossEntropyLoss); the DiCE model wrapper the methods
        # query expects probabilities, so inference sigmoids the output.
        self.return_logits = return_logits

    def forward(self, x):
        x = self.fc2(F.relu(self.fc1(x)))
        return x if self.return_logits else torch.sigmoid(x)


def main(args):
    data_dir = f'data/{args.dataname}'
    model_filename = f'{data_dir}/black_box_mlp.pkl'
    if os.path.isfile(model_filename):
        print(f"Black box already trained: {model_filename}")
        return

    with open(f'{data_dir}/info.json') as f:
        info = json.load(f)

    X_num_cat_ohe, _, _, _, y_ohe = preprocess(
        data_dir, task_type=info['task_type'], cat_encoding='one-hot', num_encoding='minmax')
    X_train, X_test = X_num_cat_ohe
    y_train, y_test = y_ohe

    loader = lambda X, y: DataLoader(TabularDataset(X, y=y, info=info),
                                     batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    train_loader, test_loader = loader(X_train, y_train), loader(X_test, y_test)

    torch.manual_seed(42)
    clf = BBMLPCLF(X_train.shape[-1], return_logits=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, total=len(train_loader), ncols=80)
        pbar.set_description(f"Black box epoch {epoch + 1}/{EPOCHS}")
        clf.train()
        for batch_num, _, batch_y in pbar:
            optimizer.zero_grad()
            loss = criterion(clf(batch_num), batch_y)
            loss.backward()
            optimizer.step()

    clf.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_num, _, batch_y in test_loader:
            predicted = torch.max(clf(batch_num).data, 1)[1]
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    print(f'  black box test accuracy: {correct / total:.4f}')

    torch.save(clf.state_dict(), model_filename)
    print(f'  saved {model_filename}')
