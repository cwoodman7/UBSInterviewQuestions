import numpy as np


def runs(N):
    """Returns a vector of length N containing ones and zeros"""
    x = np.round(np.random.random(N))
    return x


def max_runs(x):
    """Returns the length of the longest run of consecutive heads"""
    count_vec = np.zeros(len(x))
    count_vec[0] = x[0]
    for i in range(1, len(x)):
        if x[i] == 1:
            count_vec[i] = count_vec[i-1] + 1
    return int(np.max(count_vec))


def sequence(run):
    """Converts sequence of ones and zeros to heads and tails"""
    letters = ['H' if run[i] == 1 else 'T' for i in range(len(run))]
    s = ''
    for l in letters:
        s += l
    return s