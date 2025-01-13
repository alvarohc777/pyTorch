import numpy as np
import torch
import pandas as pd

# Plots
import matplotlib.pyplot as plt
import plotly.express as px

from torchmetrics.functional.classification import binary_stat_scores

# Set
device = "cuda" if torch.cuda.is_available() else "cpu"


def conf_matrix_metrics(conf_matrix: torch.LongTensor) -> dict:
    """
    Returns dictionary with metrics from a confusion matrix.

            Parameters:
                    conf_matrix (torch.Tensor): confusion matrix of dimension (1, 5)
                        [TP, FP, TN, FN, TP + FN]

            Returns:
                    metrics (dict): dictionary with following metrics:
                        metrics["TOTAL"] -> total amount of samples.
                        metrics["TPR"]   -> True Positive Rate,  sensibility, recall, hit-rate.
                        metrics["FPR"]   -> False Positive Rate, Fallout.
                        metrics["TNR"]   -> True Negative Rate,  specificity, selectivity
                        metrics["ACC"]   -> Accuracy.
                        metrics["PPV"]   -> Positive Predictive Value, Precision.
    """
    if conf_matrix.shape == (5,):
        conf_matrix = np.expand_dims(conf_matrix, axis=0)

    metrics = {}
    TP = int(conf_matrix[0, 0].item())
    FP = int(conf_matrix[0, 1].item())
    TN = int(conf_matrix[0, 2].item())
    FN = int(conf_matrix[0, 3].item())
    metrics["TP"] = TP
    metrics["FP"] = FP
    metrics["TN"] = TN
    metrics["FN"] = FN
    P = TP + FN
    N = TN + FP
    TOTAL = TP + FP + TN + FN
    metrics["TOTAL"] = TOTAL
    try:
        metrics["TPR"] = TP / (TP + FN)
    except ZeroDivisionError:
        metrics["TPR"] = "ZeroDivisionError"
    try:
        metrics["FPR"] = FP / (FP + TN)
        metrics["TNR"] = TN / (FP + TN)
    except ZeroDivisionError:
        metrics["FPR"] = "ZeroDivisionError"
        metrics["TNR"] = "ZeroDivisionError"
    metrics["ACC"] = (TP + TN) / (TOTAL)
    try:
        metrics["PPV"] = TP / (TP + FP)
    except ZeroDivisionError:
        print("No se puede obtener PPV, división por cero")
    return metrics


# Data visualization (CPU)
def confusion_matrix_labels(pred_label, true_label):
    label = ""
    if int(pred_label) == int(true_label):
        label += "T"
    else:
        label += "F"
    if pred_label == 1:
        label += "P"
    else:
        label += "N"
    return label


confusion_matrix_pandas = np.vectorize(confusion_matrix_labels)


def confusion_matrix(
    preds: torch.FloatTensor, labels: torch.FloatTensor
) -> pd.DataFrame:
    preds = preds.detach()
    labels = labels.detach()
    data = {
        "Pred probability": torch.reshape(preds, (-1,)).cpu().numpy(),
        "Pred label": torch.reshape(torch.round(preds), (-1,)).int().cpu().numpy(),
        "True label": torch.reshape(labels, (-1,)).int().cpu().numpy(),
    }
    df = pd.DataFrame(data)
    df["Result"] = confusion_matrix_pandas(df["Pred label"], df["True label"])
    return df


def signal_exploration(event_idx, dataset, model, plot: bool = False):
    window_indices = dataset.get_indices_event(event_idx)
    conf_matrix = torch.zeros(1, 5, dtype=torch.int64).to(device)
    preds = torch.empty((0, 1)).to(device)
    labels = torch.empty((0, 1)).to(device)
    # plot signal
    if plot:
        signals, label = dataset.get_event(event_idx)
        t = dataset.get_time()
        plt.plot(t, signals)
        plt.show()

    for idx in window_indices:
        signal, label = dataset[idx]
        label = torch.unsqueeze(label, 0).to(device)
        signal = torch.unsqueeze(signal, 0).to(device)
        pred = model(signal)
        preds = torch.cat((preds, pred), 0)
        labels = torch.cat((labels, label), 0)
        conf_matrix = conf_matrix.add(binary_stat_scores(pred, label))
    df = confusion_matrix(preds, labels)
    max_idx, min_idx = window_indices[-1], window_indices[0]
    df.insert(
        loc=0, column="event_idx", value=np.repeat(event_idx, max_idx - min_idx + 1)
    )
    return df, conf_matrix


def plot_confusion_matrix(metrics):
    z = [[metrics["TP"], metrics["FN"]], [metrics["FP"], metrics["TN"]]]
    fig = px.imshow(
        z,
        text_auto=True,
        template="seaborn",
        # labels=dict(x="Predicted Label", y="Real Label", color="Predictions"),
        labels=dict(x="Predicted Label", y="Real Label"),
        color_continuous_scale="gray",
        x=["Positive", "Negative"],
        y=["Positive", "Negative"],
        width=400,
        height=300,
    )
    fig.update_layout(
        font=dict(family="Times New Roman", size=16),  # Set the font family and size
        yaxis=dict(title="Real Label", tickangle=-90),
    )
    fig.show()


def print_metrics(metrics):
    print(f"{'Total windows:':.<30}{metrics['TOTAL']:4}")
    print(f"{'True Positives:':.<30}{metrics['TP']:4}")
    print(f"{'False Positives:':.<30}{metrics['FP']:4}")
    print(f"{'True Negatives:':.<30}{metrics['TN']:4}")
    print(f"{'False Negatives:':.<30}{metrics['FN']:4}")
    print(f"{'Accuracy:':.<30}{metrics['ACC']*100:>6.1f}%")
    try:
        print(f"{'True Positive Rate:':.<30}{metrics['TPR']*100:>6.1f}%")
    except ValueError:
        print("TPR cannot be calculates, division by zero")
    try:
        print(f"{'False Positive Rate:':.<30}{metrics['FPR']*100:>6.1f}%")
        print(f"{'True Negative Rate:':.<30}{metrics['TNR']*100:>6.1f}%")
    except ValueError:
        print("FPR and TNR cannot be calculates, division by zero")

    try:
        print(f"{'Positive Predictive Value:':.<30}{metrics['PPV']*100:>6.1f}%")
    except KeyError:
        print(f"PPV divided by 0. No positive class predicted")


from tqdm import tqdm


def dataframe_creation(dataset, model):
    conf_matrix = torch.zeros(0, 5, dtype=torch.int64).to(device)
    for idx in tqdm(range(dataset.len_events())):
        df, CM = signal_exploration(idx, dataset, model)
        conf_matrix = torch.cat((conf_matrix, CM))
        if idx == 0:
            dataset_df = df
        else:
            dataset_df = pd.concat([dataset_df, df])

    dataset_df = dataset_df.reset_index()
    dataset_df = dataset_df.rename(columns={"index": "window idx"})
    conf_matrix_total = np.sum(conf_matrix.cpu().numpy(), axis=0)
    return dataset_df, conf_matrix_total


def test_result(df: pd.DataFrame) -> tuple[int, int, int, int]:
    p = df[df["True label"] == 1].event_idx.value_counts().shape[0]
    n = df[df["True label"] == 0].event_idx.value_counts().shape[0]
    tp = test_result_count(df, "TP", 1)
    fp = test_result_count(df, "FP", 0)
    tn = n - fp
    fn = p - tp
    print(
        f"True Positives: {tp} \nFalse Positives: {fp} \nTrue Negatives: {tn} \nFalse Negatives: {fn}"
    )
    return tp, fp, tn, fn


def test_result_count(
    df: pd.DataFrame, expected_result: str, Label: int, includes=True
) -> int:
    if includes:
        test_result_df = df[
            (df["Result"] == expected_result) & (df["True label"] == Label)
        ]
    else:
        test_result_df = df[
            (df["Result"] != expected_result) & (df["True label"] == Label)
        ]

    test_result_count_per_event = test_result_df["event_idx"].value_counts()
    return test_result_count_per_event.shape[0]


def false_events_plots(false_events: pd.DataFrame, Title: str):
    """_summary_

    Parameters
    ----------
    false_events : pd.DataFrame
        _description_
    Title : str
        _description_
    """
    # df and plots settings
    pd.set_option("display.float_format", "{:.2%}".format)
    pd.options.plotting.backend = "matplotlib"
    pd.set_option("display.max_rows", 2000)

    false_events_plot = false_events.groupby(["window idx"])["window idx"].count()

    if len(false_events_plot) > 0:
        print(Title.upper())
        print(f"Total ")
        print(false_events)
        false_events.groupby(["event_idx"])["event_idx"].count().plot(kind="bar")
        plt.title(f"{Title}s per Event")
        plt.show()

        false_events_plot.plot(kind="bar", edgecolor="black")
        plt.title(f"{Title} per Window Index")
        plt.show()

        limits_range = (
            range(6) if (false_events["Pred probability"].min() < 0.5) else range(5, 11)
        )
        print(f"LIMITS RANGE {limits_range}")
        false_events["Pred probability"].value_counts(
            bins=[i * 0.1 for i in limits_range], sort=False, normalize=True
        ).plot(kind="bar")
        plt.title(f"{Title} probability per time interval")
        plt.show()


def trip_percentage(df: pd.DataFrame, fs: int = (60 * 64)) -> pd.Series:
    """_summary_

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe with following columns:
            window idx | event_idx | Pred probability | Pred label | True label | Result
    fs : int, optional
        Sampling frequency of the system, by default (60 * 64)

    Returns
    -------
    pd.Series
        Returns count of all trips by window index. (trips vs window_idx)
    """
    df = df[df["True label"] == 1]
    events_amount = len(df.groupby("event_idx"))
    df = df[df["True label"] == 1]
    df = df[df["Pred label"] == df["True label"]]
    df = df.groupby("event_idx").first()
    df = df.groupby("window idx").size()
    df.index = df.index / fs * 1000
    df = df / events_amount
    df = df.cumsum()

    return df
