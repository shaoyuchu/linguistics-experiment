from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
from scikit_posthocs import posthoc_dunn

CONDITIONS = [
    "Unordered Characters within Words",
    "Unordered Characters between Words",
    "Unordered Words",
]

SWAP_NUM = [3, 6]


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

    for cond_id in range(3):
        for swap_id in range(2):
            i = (cond_id * 2 + swap_id) * 4 + 1
            col_name = f"{CONDITIONS[cond_id]} *{SWAP_NUM[swap_id]}"
            df[col_name] = para_correctness.loc[:, i : i + 3].mean(axis=1)
    return df


def scatter_plot(cond_correctness: pd.DataFrame, image_dir: Path):
    for cond in CONDITIONS:
        plt.clf()
        plt.title(cond)
        for i in SWAP_NUM:
            plt.scatter(
                [i] * len(cond_correctness),
                cond_correctness[f"{cond} *{i}"],
                alpha=0.02,
                linewidths=0,
                c="b",
            )
        mean = [np.mean(cond_correctness[f"{cond} *{i}"]) for i in SWAP_NUM]
        std = [np.std(cond_correctness[f"{cond} *{i}"]) for i in SWAP_NUM]
        plt.errorbar(
            SWAP_NUM,
            mean,
            yerr=std,
            # error bar
            elinewidth=1,
            ecolor="#b3b3b3",
            capsize=2,
            # mean trend
            c="#b3b3b3",
            linestyle="--",
            linewidth="1",
            marker="o",
            markerfacecolor="none",
        )
        plt.xlabel("# of swaps")
        plt.ylabel("Accuracy")
        plt.ylim([0, 1.1])
        plt.savefig(image_dir / f"Scatter - {cond}.png", dpi=300)


def histogram_by_swap_type(cond_correctness: pd.DataFrame, image_dir: Path):
    plt.style.use("ggplot")
    bins = np.linspace(0, 1, num=8)
    n_user = len(cond_correctness)
    for cond in CONDITIONS:
        plt.clf()
        plt.title(cond, pad=20)
        for i in range(2):
            plt.hist(
                cond_correctness[f"{cond} *{SWAP_NUM[i]}"],
                alpha=0.5,
                label=f"{SWAP_NUM[i]} swaps",
                bins=bins,
                weights=np.ones(n_user) * 100 / n_user,
            )
        plt.legend(loc="upper left", facecolor="white", edgecolor="none")
        plt.xlabel("Accuracy")
        plt.ylabel("Percentage (%)")
        plt.ylim([0, 100])
        plt.savefig(image_dir / f"Histogram - {cond}.png", dpi=300)


def bar_plot_by_swap_type(cond_correctness: pd.DataFrame, image_dir: Path):
    plt.style.use("ggplot")
    n_user = len(cond_correctness)
    accuracy = np.arange(0, 1.125, 0.125)
    for cond in CONDITIONS:
        plt.clf()
        data = pd.DataFrame(index=accuracy)
        for i in range(2):
            count = Counter(cond_correctness[f"{cond} *{SWAP_NUM[i]}"])
            total = np.array(list(count[acc] for acc in accuracy))
            percentage = total * 100 / n_user
            data[f"{SWAP_NUM[i]} Swaps"] = percentage
        data.plot(kind="bar", width=0.8)
        plt.legend(loc="upper left", facecolor="white", edgecolor="none")
        plt.title(cond, pad=20)
        # plt.xlabel("Accuracy")
        # plt.ylabel("Percentage (%)")
        plt.ylim([0, 50])
        plt.xticks(rotation=45)
        plt.savefig(image_dir / f"Bar Plot - {cond}.png", dpi=300)


def line_plot_by_num_swap(cond_correctness: pd.DataFrame, image_dir: Path):
    plt.style.use("ggplot")
    n_user = len(cond_correctness)
    accuracy = np.arange(0, 1.125, 0.125)
    for swap_num in SWAP_NUM:
        plt.clf()
        plt.title(
            f"Accuracy Distribution of Different Swap Types ({swap_num} swaps)",
            pad=20,
        )
        for i in range(3):
            count = Counter(cond_correctness[f"{CONDITIONS[i]} *{swap_num}"])
            total = np.array(list(count[acc] for acc in accuracy))
            percentage = total * 100 / n_user
            plt.plot(
                accuracy,
                percentage,
                label=CONDITIONS[i],
            )
        plt.legend(loc="upper left", facecolor="white", edgecolor="none")
        plt.xlabel("Accuracy")
        plt.ylabel("Percentage (%)")
        plt.ylim([0, 100])
        plt.savefig(image_dir / f"Line - {swap_num} swaps.png", dpi=300)


def histogram_by_num_swap(cond_correctness: pd.DataFrame, image_dir: Path):
    plt.style.use("ggplot")
    bins = np.linspace(0, 1, num=8)
    n_user = len(cond_correctness)
    for swap_num in SWAP_NUM:
        plt.clf()
        plt.title(
            f"Accuracy Distribution of Different Swap Types ({swap_num} swaps)",
            pad=20,
        )
        for i in range(3):
            plt.hist(
                cond_correctness[f"{CONDITIONS[i]} *{swap_num}"],
                alpha=0.5,
                label=CONDITIONS[i],
                bins=bins,
                weights=np.ones(n_user) * 100 / n_user,
            )
        plt.legend(loc="upper left", facecolor="white", edgecolor="none")
        plt.xlabel("Accuracy")
        plt.ylabel("Percentage (%)")
        plt.ylim([0, 100])
        plt.savefig(image_dir / f"Histogram - {swap_num} swaps.png", dpi=300)


def bar_plot_by_num_swap(cond_correctness: pd.DataFrame, image_dir: Path):
    plt.style.use("ggplot")
    n_user = len(cond_correctness)
    accuracy = np.arange(0, 1.125, 0.125)
    for swap_num in SWAP_NUM:
        plt.clf()
        data = pd.DataFrame(index=accuracy)
        for i in range(3):
            count = Counter(cond_correctness[f"{CONDITIONS[i]} *{swap_num}"])
            total = np.array(list(count[acc] for acc in accuracy))
            percentage = total * 100 / n_user
            data[CONDITIONS[i]] = percentage
        data.plot(kind="bar", width=0.8)
        plt.legend(loc="upper left", facecolor="white", edgecolor="none")
        plt.title(
            f"Accuracy Distribution of Different Swap Types",
            pad=20,
        )
        # plt.xlabel("Accuracy")
        # plt.ylabel("Percentage (%)")
        plt.ylim([0, 50])
        plt.xticks(rotation=45)
        plt.savefig(image_dir / f"Bar Plot - {swap_num} swaps.png", dpi=300)


def print_statistics(cond_correctness: pd.DataFrame):
    for cond in CONDITIONS:
        for swap_num in SWAP_NUM:
            col_name = f"{cond} *{swap_num}"
            data = cond_correctness[col_name]
            mean = np.mean(data)
            std = np.std(data)
            print(f"{col_name}:\tmean = {'%.2f'%mean}, std = {'%.2f'%std}")


def compare_swap_types(cond_correctness: pd.DataFrame):
    """Run Kruskal-Wallis test and Dunn's test for post-hoc test.

    Test if the median of the correctness is the same for different swap types.
    Note that the statistics are not normally distributed, so we use
    non-parametric tests.
    """
    for swap_num in SWAP_NUM:
        result = stats.kruskal(
            *[
                cond_correctness[f"{CONDITIONS[i]} *{swap_num}"]
                for i in range(3)
            ],
        )
        print(
            f"\nKruskal-Wallis test for {swap_num} swaps: "
            f"p-value = {'%.2e'%result.pvalue}"
        )

        # post hoc Dunn's test with Bonferroni correction
        if result.pvalue < 0.05:
            post_hoc = posthoc_dunn(
                [
                    cond_correctness[f"{CONDITIONS[i]} *{swap_num}"]
                    for i in range(3)
                ],
                p_adjust="bonferroni",
            )
            # print("post-hoc Dunn's test:\n", post_hoc < 0.05)
            print("post-hoc Dunn's test:\n", post_hoc)


def compare_swap_num(cond_correctness: pd.DataFrame):
    """Run Mann-Whitney U test.

    Test if the median of the accuracy is the same for different swap types.
    Note that the statistics are not normally distributed, so we use
    non-parametric tests.
    """
    for cond in CONDITIONS:
        result = stats.mannwhitneyu(
            *[cond_correctness[f"{cond} *{SWAP_NUM[i]}"] for i in range(2)]
        )
        print(
            f"Mann-Whitney U test for {cond}: "
            f"p-value = {'%.2f'%result.pvalue}"
        )


if __name__ == "__main__":
    data_dir = Path("data")
    image_dir = Path("image")
    image_dir.mkdir(exist_ok=True)
    para_correctness = get_by_paragraph_correctness(data_dir)
    cond_correctness = get_by_condition_correctness(para_correctness)
    scatter_plot(cond_correctness, image_dir)
    line_plot_by_num_swap(cond_correctness, image_dir)
    histogram_by_swap_type(cond_correctness, image_dir)
    histogram_by_num_swap(cond_correctness, image_dir)
    bar_plot_by_num_swap(cond_correctness, image_dir)
    bar_plot_by_swap_type(cond_correctness, image_dir)
    compare_swap_types(cond_correctness)
    compare_swap_num(cond_correctness)
    print_statistics(cond_correctness)
