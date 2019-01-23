import matplotlib.pyplot as plt
import numpy as np


def plot_direction_field(first_derivative_fn, xlim=(-1, 1), ylim=(-1, 1)):
    Y, X = np.mgrid[ylim[0]:ylim[1]:20j, xlim[0]:xlim[1]:20j]
    first_deriv = first_derivative_fn(X, Y)
    angle = np.arctan(first_deriv)
    U = np.cos(angle)
    V = np.sin(angle)
    plt.quiver(X, Y, U, V)
    plt.axhline(color="lightgrey")
    plt.axvline(color="lightgrey")
