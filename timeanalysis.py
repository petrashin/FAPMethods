import numpy as np
# from pywavelets import wavedec
from pywt import wavedec
from typing import List

def fft_impl(x_array: List):
    x_array = np.array(x_array)
    N = x_array.shape[0]
    if N % 2 > 0:
        raise ValueError("must be a power of 2")
    elif N <= 2:
        x_array = np.asarray(x_array, dtype=float)
        NN = x_array.shape[0]
        nn = np.arange(NN)
        k = nn.reshape((NN, 1))
        M = np.exp(-2j * np.pi * k * nn / N)
        return np.dot(M, x_array)
    else:
        X_even = fft_impl(x_array[::2])
        X_odd = fft_impl(x_array[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N/2)] * X_odd,
                               X_even + terms[int(N/2):] * X_odd])


def fft(arr: List):
    t1 = arr + [0]*(2**int(np.ceil(np.math.log2(len(arr))))-len(arr))
    return fft_impl(t1)


def non_sinusoidal_coefficient(y: List, t: str):
    y8 = round(len(y)*0.08)
    y = y[y8:-y8]
    d1 = np.std(wavedec(y, t, level=1)[1])
    d2 = np.std(wavedec(y, t, level=2)[1])
    d3 = np.std(wavedec(y, t, level=3)[1])
    d4 = np.std(wavedec(y, t, level=4)[1])
    a4 = np.std(wavedec(y, t, level=4)[0])
    return ((d1 ** 2 + d2 ** 2 + d3 ** 2 + d4 ** 2) ** 0.5) / a4

def nonsinFurie(data: List):
    y = fft(data)
    y_max = max(y)
    y_sum = sum(y) - y_max

    return y_max/y_sum