import concurrent.futures
from tqdm import tqdm
import pandas as pd
import os
import utils.data_exploration as de
import numpy as np


def process_df(relay_number):
    relay_number += 1
    df = pd.read_parquet(f"parquet_data/automation/R{relay_number}_df.parquet")
    conf_matrix = de.df_to_CM(df)
    conf_matrix.to_parquet(f"parquet_data/automation/CM/R{relay_number}_CM_df.parquet")
    np.save(
        f"parquet_data/automation/CM/R{relay_number}_CM.npy",
        conf_matrix,
    )


if __name__ == "__main__":
    cores = os.cpu_count() - 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        list(
            tqdm(
                executor.map(process_df, range(18)),
                total=len(range(18)),
                desc="Processing Files",
            )
        )
