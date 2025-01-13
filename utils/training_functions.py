import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import os
import numpy as np
import pandas as pd
import random
import utils_tesis.dataset_creator as dc

# Set
device = "cuda" if torch.cuda.is_available() else "cpu"


# Model Creation
class FaultDetector(nn.Module):
    """Information about FaultDetector"""

    def __init__(self, n_signals, hidden_dim, tagset_size):
        super(FaultDetector, self).__init__()
        self.lstm = nn.LSTM(n_signals, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)
        # self.norm = nn.BatchNorm1d(tagset_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        fc_layer = self.fc(lstm_out[:, -1, :])
        # norm_layer = self.norm(fc_layer)

        return torch.sigmoid(fc_layer)


class Form1Dataset(torch.utils.data.Dataset):
    """Some information about Form1Dataset"""

    def __init__(
        self,
        dataset_dir: str,
        signal_names: list[str],
        dataset_size: int = 999999,
        events_size: int = 999999,
    ):
        super(Form1Dataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.signal_names = signal_names
        self.dataset_size = dataset_size
        self.events_size = events_size
        self.parquets = self.get_database_list()

        _signal = pd.read_parquet(f"{self.dataset_dir}/{self.parquets[0]}")
        self.time = _signal["time"]
        self.N = dc.cycle_info(_signal)["samples_per_cycle"]

        time_info = dc.extract_time_info(_signal, "time")
        self.time = self.calculate_time_array(time_info)

        self.max_window_idx = self.N - 1
        self.windows_event = dc.calculate_windows(_signal, self.N)
        # Reducir dataset size
        self.reduce_dataset()

    def calculate_time_array(self, time_info: dict) -> np.ndarray:
        dt = time_info["dt"]
        total_samples = time_info["total_samples"]
        total_time = time_info["total_time"]
        return np.linspace(0, total_time, total_samples)

    def __len__(self):
        return len(self.parquets) * self.windows_event

    def __getitem__(self, index):
        window_idx = index % self.max_window_idx
        parquet_idx = index // self.max_window_idx
        parquet_name = self.parquets[parquet_idx]
        parquet_path = f"{self.dataset_dir}/{parquet_name}"
        df = pd.read_parquet(parquet_path)[self.signal_names]

        df = df.iloc[window_idx : self.N + window_idx]
        if "F_T" in parquet_name:
            labels = np.array([1])
        if "L_T" in parquet_path:
            labels = np.array([0])
        signals = torch.from_numpy(df.to_numpy()).float()
        labels = torch.from_numpy(labels).float()
        return signals, labels

    def reduce_dataset(self):
        # reducir el tamaño de dataset, menos eventos
        if self.events_size < len(self.parquets):
            self.parquets = random.sample(self.parquets, self.events_size)
            self.dataset_size = len(self) * self.windows_event

        # reducir la cantidad de muestras (ventanas) según se necesite
        if self.dataset_size < len(self):
            self.parquets = random.sample(
                self.parquets, int(self.dataset_size / self.N)
            )

    def get_database_list(self):
        file_set = set()
        for dir_, _, files in os.walk(self.dataset_dir):
            for file_name in files:
                rel_dir = os.path.relpath(dir_, self.dataset_dir)
                rel_file = os.path.join(rel_dir, file_name)
                file_set.add(rel_file)
        parquets = list(file_set)

        return parquets

    def max_idx(self):
        return len(self.parquets) - 1

    def get_event(self, index):
        parquet_name = self.parquets[index]
        parquet_path = f"{self.dataset_dir}/{parquet_name}"
        if "F_T" in parquet_name:
            labels = np.array([1])
        if "L_T" in parquet_path:
            labels = np.array([0])
        return pd.read_parquet(parquet_path)[self.signal_names], labels

    def len_events(self):
        return len(self.parquets)

    def get_indices_event(self, index):
        min_idx = index * self.max_window_idx
        max_idx = min_idx + self.windows_event
        return np.arange(min_idx, max_idx)

    def get_time(self):
        return self.time

    def get_time_window(self, index):
        window_idx = index % self.max_window_idx
        return self.time[window_idx : self.N + window_idx]

    def export_dataset(self) -> dict:
        data = {}
        data["dataset_dir"] = self.dataset_dir
        data["parquets"] = self.parquets
        data["N"] = self.N
        data["time"] = self.time
        data["max_window_idx"] = self.max_window_idx
        data["windows_event"] = self.windows_event
        data["events_size"] = self.events_size
        data["dataset_size"] = self.dataset_size
        return data


class ExistingDataset(Form1Dataset):
    def __init__(
        self,
        data: dict,
        signal_names: list[str],
        dataset_size: int = None,
        events_size: int = None,
    ):
        self.dataset_dir = data["dataset_dir"]
        self.parquets = data["parquets"]
        self.N = data["N"]
        self.time = data["time"]
        self.max_window_idx = data["max_window_idx"]
        self.windows_event = data["windows_event"]
        self.signal_names = signal_names

        self.events_size = data["events_size"]
        self.dataset_size = data["dataset_size"]
        if dataset_size or events_size:
            self.reduce_dataset()


# Create Training / Test / Validation Loops


# Training
def train(
    dataloader,
    model,
    loss_fn,
    optimizer,
    my_lr_scheduler,
    return_loss=False,
):
    size = len(dataloader.dataset)
    model.train()
    loss_list = []

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if return_loss == True:
            loss_list.append(loss.item())
        if batch % 2 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>f} [{current:>5d}/{size:>5d}]")

        if batch % 5 == 0:
            my_lr_scheduler.step()
    if return_loss == True:
        return loss_list


from torchmetrics.functional.classification import binary_stat_scores
