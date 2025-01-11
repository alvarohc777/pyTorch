from utils_tesis.dataset_creator import clean_file
from utils.relay_list import signals
from tqdm import tqdm
import concurrent.futures
import os

signals.append("time")

if __name__ == "__main__":
    print(__name__)


def repeated_clean(file):
    print(file)
    clean_file(
        file,
        window_size=16,
        downsampling=13,
        # keep_types=["time", "I"],
        keep_columns=signals,
        rmv_cycles_start=2,
        rmv_cycles_end=2,
    )


if __name__ == "__main__":
    cores = os.cpu_count() - 2

    dataset_dir = "D:/PaperLSTM/database/DB1"
    file_set = set()
    for dir_, _, files in os.walk(dataset_dir):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, dataset_dir)
            rel_file = os.path.join(rel_dir, file_name)
            file_set.add(f"{dataset_dir}/{rel_file}")
    csv_list = list(file_set)

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        executor.map(repeated_clean, csv_list)
