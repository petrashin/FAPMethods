def fft_impl(x_array):
    #x_array += [0]*(2**int(numpy.ceil(numpy.math.log2(len(x_array))))-len(x_array))
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
        X_even = self.fft_impl(x_array[::2])
        X_odd = self.fft_impl(x_array[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + terms[:int(N/2)] * X_odd,
                               X_even + terms[int(N/2):] * X_odd])


def fft(self, arr):
    t1 = arr + [0]*(2**int(numpy.ceil(numpy.math.log2(len(arr))))-len(arr))
    return self.fft_impl(t1)


def non_sinusoidal_coefficient(self, t):
    y8 = round(len(self.y)*0.08)
    y = self.y[y8:-y8]
    d1 = numpy.std(wavedec(y, t, level=1)[1])
    d2 = numpy.std(wavedec(y, t, level=2)[1])
    d3 = numpy.std(wavedec(y, t, level=3)[1])
    d4 = numpy.std(wavedec(y, t, level=4)[1])
    a4 = numpy.std(wavedec(y, t, level=4)[0])
    return ((d1 ** 2 + d2 ** 2 + d3 ** 2 + d4 ** 2) ** 0.5) / a4

def nonsinFurie(self, data=None):
    if data is not None:
      y = self.fft(data)
    y = self.fft(self.y)
    y_max = max(y)
    y_sum = sum(y) - y_max

    return y_max/y_sum