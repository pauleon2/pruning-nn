import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a


def signum(x):
    a = []
    for item in x:
        if item < 0:
            a.append(0)
        else:
            a.append(1)
    return a


def plot_signum():
    Xaxis = np.arange(-10., 10., 0.2)
    sig = signum(Xaxis)
    plt.plot(Xaxis, sig)
    plt.show()


def plot_sigmoid():
    Xaxis = np.arange(-10., 10., 0.2)
    sig = sigmoid(Xaxis)
    plt.plot(Xaxis, sig)
    plt.show()


if __name__ == '__main__':
    # sns.set_context('paper')
    plot_sigmoid()
    # plot_signum()
