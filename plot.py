import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def get_top_k_words(subreddit, data, k):
    freqs = data.loc[data.Sub == subreddit].iloc[0, 1:]
    freqs = freqs.sort_values(ascending=False)[:k]
    freqs = freqs/freqs.max()  # relative
    return freqs


def plot_subreddit(subreddit, words):
    plt.figure(figsize=(8, 6))
    data = pd.DataFrame({"Word": words.index, "TF-IDF Frequencies": words.values})
    ax = sns.barplot(data, x="Word", y="TF-IDF Frequencies")
    ax.set_title("/r/" + subreddit, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right", fontsize=14, horizontalalignment="center")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(f"fig/{subreddit}.png", transparent=False, facecolor="white")


def main():
    data = pd.read_csv("data_freq.csv")
    for sub in [x[0] for (i, x) in data.iterrows()]:
        plot_subreddit(sub, get_top_k_words(sub, data, 10))

if __name__ == "__main__":
    main()

