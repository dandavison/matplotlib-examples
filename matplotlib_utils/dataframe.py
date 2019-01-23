import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
from matplotlib import pyplot as plt
from numpy import ceil, sqrt


plt.rcParams.update({'figure.autolayout': True})


def plot_binary_classification(df, target):
    y = df[target]
    assert np.issubdtype(y.dtype, np.bool), "target must be boolean"
    y = y.astype(np.bool)
    df = df[[col for col in df.columns if col != target]]
    n_obs, n_features = df.shape
    nrows, ncols = _choose_subplot_dimensions(n_features)
    fig, axes_array = plt.subplots(nrows, ncols)
    for feature, ax in zip(df.columns, axes_array.ravel()):
        _plot_feature(feature, ax, df, y)
    plt.show()


def _plot_feature(feature, ax, df, y):
    x = df[feature]
    # TODO: Handle categorical features.
    if np.issubdtype(x.dtype, np.bool):
        x = x.astype(np.int)

    range = tuple(x.quantile([0.05, 0.95]))
    ax.hist(x[y], range=range, density=True)
    ax.hist(x[~y], range=range, density=True, alpha=0.6, color="green")

    ax.get_yaxis().set_visible(False)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.set_title(feature, fontsize=5)


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
