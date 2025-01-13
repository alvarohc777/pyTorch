import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Métricas
from torchmetrics.functional.classification import binary_stat_scores

# Visualizar datos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Progress bar
from tqdm import tqdm

# Acceder al sistema operativo
import os
import glob
import shutil
import random

import utils.training_functions as tf
from utils.relay_list import signals

# LSTM parameters
hidden_dim = 20
n_signals = 3
N = 64

# _batch_size => m in figure 1.
train_batch_size = 64
dev_batch_size = 16
test_batch_size = 16

# Classification type (binary)
tagset_size = 1

# Set
device = "cuda" if torch.cuda.is_available() else "cpu"

model = tf.FaultDetector(n_signals, hidden_dim, tagset_size).to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Learning rate decay (optional)
decayRate = 0.96
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer=optimizer, gamma=decayRate
)


dataset_dir = "D:\PaperLSTM\database\DB1_nueva_full\DB1_clean"

# get dataset split params
windows_train = 5000
windows_test = 500


for i in tqdm(range(18)):
    relay_number = i + 1
    signal_names = signals[(relay_number - 1) * 3 : (relay_number - 1) * 3 + 3]
    dataset = tf.Form1Dataset(
        dataset_dir,
        signal_names=signal_names,
    )
    windows_amount = len(dataset)
    train_percentage = round(windows_train / windows_amount, 4)
    test_percentage = round(windows_test / windows_amount, 4)
    train_dataset, test_dataset, _ = random_split(
        dataset,
        [train_percentage, test_percentage, 1 - train_percentage - test_percentage],
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    # Entrenamiento
    epochs = 1
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n--------------------------------")
        train_loss = tf.train(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            my_lr_scheduler,
            return_loss=True,
        )
    try:
        mini_batch_list = [index for index, _ in enumerate(train_loss)]
        train_loss_ewm = pd.DataFrame(train_loss).ewm(com=0.95).mean()
        plt.plot(mini_batch_list, train_loss, mini_batch_list, train_loss_ewm)
    except NameError:
        print("Error! Run train loop")
    torch.save(model.state_dict(), f"./models/automation/R{relay_number}_currents.pth")
