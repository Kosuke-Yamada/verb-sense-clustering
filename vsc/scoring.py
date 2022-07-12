import numpy as np


def eval_confusion_matrix(df):
    spearman = calc_spearman(df)
    acc = calc_accuracy(df)
    mse = calc_mse(df)
    return {"spearman": spearman, "acc": acc, "mse": mse}


def calc_spearman(df):
    return df.corr("spearman").loc["n_frames", "n_clusters"]


def calc_accuracy(df):
    return len(df[df["n_frames"] == df["n_clusters"]]) / len(df)


def calc_mse(df):
    df["mse"] = (df["n_frames"] - df["n_clusters"]) ** 2
    return np.sqrt(sum(df["mse"]) / len(df))
