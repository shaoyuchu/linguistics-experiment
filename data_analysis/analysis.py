from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CONDITIONS = [
    "Unordered Characters (in-word)",
    "Unordered Characters (cross-word)",
    "Unordered Words",
]


def get_by_paragraph_correctness(data_dir: Path):
    summary_df = pd.DataFrame(columns=list(range(0, 25)))
    for data_path in data_dir.glob("*"):
        df = pd.read_csv(
            data_path,
            dtype={"No": str},
            converters={"correct": lambda x: 1 if x == "true" else 0},
        )
        index = df["No"]
        correct = df["correct"]

        correctness = np.zeros(25)
        for i in range(len(index)):
            if not pd.isna(index[i]):
                correctness[int(index[i])] += correct[i]
        summary_df.loc[len(summary_df)] = correctness / 2
    return summary_df


def get_by_condition_correctness(para_correctness: pd.DataFrame):
    df = pd.DataFrame()
    df[f"{CONDITIONS[0]} *3"] = para_correctness.loc[:, 1:4].mean(axis=1)
    df[f"{CONDITIONS[0]} *6"] = para_correctness.loc[:, 5:8].mean(axis=1)
    df[f"{CONDITIONS[1]} *3"] = para_correctness.loc[:, 9:12].mean(axis=1)
    df[f"{CONDITIONS[1]} *6"] = para_correctness.loc[:, 13:16].mean(axis=1)
    df[f"{CONDITIONS[2]} *3"] = para_correctness.loc[:, 17:20].mean(axis=1)
    df[f"{CONDITIONS[2]} *6"] = para_correctness.loc[:, 21:24].mean(axis=1)
    return df


def plot(cond_correctness: pd.DataFrame, image_dir: Path):
    for cond in CONDITIONS:
        plt.clf()
        plt.title(cond)
        plt.scatter(
            [3] * len(cond_correctness),
            cond_correctness[f"{cond} *3"],
            alpha=0.2,
            c="b",
        )
        plt.scatter(
            [6] * len(cond_correctness),
            cond_correctness[f"{cond} *6"],
            alpha=0.2,
            c="b",
        )
        plt.xlabel("# of swaps")
        plt.ylabel("correct ratio")
        plt.savefig(image_dir / f"{cond}.png", dpi=300)


if __name__ == "__main__":
    data_dir = Path("data")
    image_dir = Path("image")
    para_correctness = get_by_paragraph_correctness(data_dir)
    cond_correctness = get_by_condition_correctness(para_correctness)
    plot(cond_correctness, image_dir)
    for cond in CONDITIONS:
        for num in [" *3", " *6"]:
            data = cond_correctness[cond + num]
            mean = np.mean(data)
            std = np.std(data)
            print(f"{cond + num}:\tmean = {'%.2f'%mean}, std = {'%.2f'%std}")
