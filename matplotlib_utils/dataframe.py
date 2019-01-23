from collections import Counter

import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from numpy import ceil, sqrt, sum


plt.rcParams.update({'figure.autolayout': True})


def plot_binary_classification(df, target):
    y = df[target]
    assert np.issubdtype(y.dtype, np.bool), "target must be boolean"
    y = y.astype(np.bool)
    df = df[[col for col in df.columns if col != target]]
    n_obs, n_features = df.shape
    nrows, ncols = _choose_subplot_dimensions(n_features)
    fig, axes_array = plt.subplots(nrows, ncols)

    features = list(df.columns) + [None] * ((nrows * ncols) - len(df.columns))

    for feature, ax in zip(features, axes_array.ravel()):
        if feature:
            x = df[feature]

            if len(set(x)) == 2:
                _plot_feature_barplot(x, y, ax)
            else:
                _plot_feature_histogram(x, y, ax)

            ax.set_title(feature, fontsize=5)
        _clean_up(ax)

    plt.show()


def _clean_up(ax):
    ax.get_yaxis().set_visible(False)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)


def _plot_feature_barplot(x, y, ax):
    def heights(items):
        counts = np.array([n for k, n in items])
        return counts / sum(counts)

    x_True = sorted(Counter(x[y]).items())
    x_False = sorted(Counter(x[~y]).items())
    bars1 = heights(x_True)
    bars2 = heights(x_False)
    bar_width = 0.15
    r1 = np.arange(len(bars1))
    r2 = [r + bar_width for r in r1]
    ax.bar(r1, bars1, alpha=0.6, width=bar_width, edgecolor='white', label='var1')
    ax.bar(r2, bars2, color='green', alpha=0.6, width=bar_width, edgecolor='white', label='var2')


def _plot_feature_histogram(x, y, ax):
    if np.issubdtype(x.dtype, np.bool):
        x = x.astype(np.int)
    range = tuple(x.quantile([0.05, 0.95]))
    ax.hist(x[y], range=range, density=True, alpha=0.6)
    ax.hist(x[~y], range=range, density=True, color="green", alpha=0.6)


def _choose_subplot_dimensions(n):
    """
    1 -> 1, 1
    2 -> 1, 2
    3 -> 2, 2
    4 -> 2, 2
    5 -> 2, 3
    6 -> 2, 3
    7 -> 3, 3
    8 -> 3, 3
    9 -> 3, 3
    """
    width = int(ceil(sqrt(n)))
    height = int(ceil(n / width))
    return height, width


if __name__ == '__main__':
    DATA_FILE = '/tmp/creditcard.csv'
    df = pd.read_csv(DATA_FILE)
    plot_binary_classification(df, 'Class')
