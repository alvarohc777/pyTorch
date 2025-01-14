from utils_tesis.dataset_creator import clean_file
from utils.relay_list import signals
from tqdm import tqdm
import concurrent.futures
import os

signals.append("time")


def repeated_clean(file):
    clean_file(
        file,
        downsampling=8,
        # keep_types=["time", "I"],
        keep_columns=signals,
        rmv_cycles_start=2,
        # rmv_cycles_end=2,
        frequency=60,
    )


def safe_repeated_clean(file_path):
    try:
        repeated_clean(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


# instalar tqdm, pyarrow y actualizar utils-tesis a versión más reciente
if __name__ == "__main__":
    cores = os.cpu_count() - 1

    dataset_dir = "D:/PaperLSTM/database/DB1_nueva_full/DB1"
    # dataset_dir = "D:/PaperLSTM/database/DB1_nueva"
    file_set = set()
    for dir_, _, files in os.walk(dataset_dir):
        for file_name in files:
            rel_dir = os.path.relpath(dir_, dataset_dir)
            rel_file = os.path.join(rel_dir, file_name)
            file_set.add(f"{dataset_dir}/{rel_file}")
    csv_list = list(file_set)

    with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
        # Use tqdm to wrap the map for the progress bar
        list(
            tqdm(
                executor.map(safe_repeated_clean, csv_list),
                total=len(csv_list),
                desc="Processing Files",
            )
        )
