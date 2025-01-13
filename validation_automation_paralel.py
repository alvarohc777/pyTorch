import pickle
import utils.training_functions as tf
import utils.data_exploration as de
from utils.relay_list import signals
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
import os

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
with open("./datasets/dataset_validation.pkl", "rb") as f:
    dataset_params = pickle.load(f)


def relay_validation(
    i: int, dataset_params=dataset_params, model=model, signals=signals
):
    print(f"ejecutando relé R{i+1}")
    relay_number = i + 1
    model.load_state_dict(torch.load(f"./models/automation/R{i+1}_currents.pth"))
    signal_names = signals[(relay_number - 1) * 3 : (relay_number - 1) * 3 + 3]
    dataset_validation = tf.ExistingDataset(dataset_params, signal_names)
    dataset_df, conf_matrix_total = de.dataframe_creation(dataset_validation, model)

    dataset_df.to_parquet(f"parquet_data/automation/R{relay_number}_df.parquet")
    conf_matrix_df = pd.DataFrame(
        conf_matrix_total.cpu(), columns=["TP", "FP", "TF", "FN", "TP + FN"]
    )
    conf_matrix_df.to_parquet(f"parquet_data/automation/R{relay_number}_CM_df.parquet")
    np.save(
        f"parquet_data/automation/R{relay_number}_CM.npy",
        conf_matrix_total.cpu(),
    )


def print_validation(val):
    print(val)


if __name__ == "__main__":

    cores = os.cpu_count() - 1
    # for i in tqdm(range(18)):
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        # Use tqdm to wrap the map for the progress bar
        list(
            tqdm(
                executor.map(relay_validation, range(18)),
                total=len(range(18)),
                desc="Processing Files",
            )
        )
