import numpy
import sympy
import numpy as np
import scipy.optimize


def chebishevPoints(x_values, y_values, n):
    a = min(x_values)
    b = max(y_values)
    result = []
    for i in range(1, n+1):
        result.append(0.5*(a+b) + 0.5*(b-a)*numpy.cos((2*i-1)/2/n*numpy.pi))
    return result


def lagrx(x_values, y_values):
    """
    функция, вызывающая интерполяцию методом Лагранжа
    :params: массив точек x
    :params: массив точек y
    :return: массив точек и интерполяционный член Лагранжа
    """
    ratio = (max(x_values)-min(x_values))/len(x_values)
    if len(x_values) >= 60 and ratio < 1:
        result = lagr_cheb(x_values, y_values)
        if result is None:
            return lagrx_standart(x_values, y_values)
        return result
    else:
        return lagrx_standart(x_values, y_values)


def lagrx_standart(x_values, y_values):
    """
    Функция, выполняющая интерполяцию методом Лагранжа (в том случае, если длина массива x < 60 и среднее > 1 или нет
    результата от lagr_cheb)
    :params: массив точек x
    :params: массив точек y
    :return: массив точек x,
    y и интерполяционный член Лагранжа
    """
    x0 = sympy.Symbol('x')
    s = 0
    for i in range(len(x_values)):
        t1 = 1
        t2 = 1
        for j in range(len(x_values)):
            t1 *= x0-x_values[j] if i != j else 1
            t2 *= x_values[i] - x_values[j] if i != j else 1
        s += y_values[i]*(t1/t2)
    return x_values, y_values, [sympy.simplify(s).subs(x0, x_values[i]) for i in x_values]


def lagr_cheb(x_values, y_values):
    """
    Функция,выполняющая интерполяцию методом Лагранжа (?) (в том случае, если длина массива x > 60 и среднее < 1)
    :params: массив точек x
    :params: массив точек y
    """
    x0 = sympy.Symbol('x')

    rounded_x = [round(i, 1) for i in x_values]
    last_poly = None
    last_error = None

    for power in range(3, 50):
        xx = list(set(rounded_x) & set(round(i, 1)
                  for i in chebishevPoints(x_values, y_values, power)))
        s = 0
        if len(xx) != power:
            continue

        for i in range(len(xx)):
            t1 = 1
            t2 = 1
            for j in range(len(xx)):
                t1 *= x0-xx[j] if i != j else 1
                t2 *= xx[i] - xx[j] if i != j else 1
            s += y_values[rounded_x.index(xx[i])]*(t1/t2)

        error = lagr_error(x_values, y_values, sympy.lambdify(x0, s))
        if last_error is not None and error > last_error*2:
            return x_values, y_values, [last_poly.subs(x0, x_values[i]) for i in x_values]
        if last_error is None or last_error > error:
            last_error = error
            last_poly = s

    if last_poly is not None:
        return x_values, y_values, [last_poly.subs(x0, x_values[i]) for i in x_values]
    return last_poly


def lagr_error(x_values, y_values, poly):
    maxx = 0
    for i, x in enumerate(x_values):
        m = abs(y_values[i]-poly(x))
        if m > maxx:
            maxx = m
    return maxx


def newtons_interpolation(x_values, y_values, is_forward=True):
    x0 = sympy.symbols('x')
    n_res = len(x_values)
    interpol = []

    def poly_newton_coefficient(x, y):
        m = len(x)
        x = np.copy(x)
        a = np.copy(y)
        h = x[1] - x[0]
        for k in range(1, m):
            a[k:m] = (a[k:m] - a[k - 1]) / h
        return a

    def newton_first(x, y):
        x_res = sympy.symbols('x')
        a = poly_newton_coefficient(x, y)
        result = 0
        for i in range(len(a)):
            factor = a[i]
            for j in range(i):
                factor *= (x_res - x[j])
            result += factor
        return sympy.simplify(result)

    def newton_second(x, y):
        x_res = sympy.symbols('x')
        a = poly_newton_coefficient(x[::-1], y[::-1])
        n = len(a)
        result = 0
        for i in range(n):
            factor = a[i]
            for j in range(n - 1, n - 1 - i, -1):
                factor *= (x_res - x[j])
            result += factor
        return sympy.simplify(result)

    if is_forward:
        res = newton_first(x_values, y_values)
        interpol = [res.subs(x0, x_values[i]) for i in range(n_res)]

    elif is_forward is False:
        res = newton_second(x_values, y_values)
        interpol = [res.subs(x0, x_values[i]) for i in range(n_res)]

    return x_values, y_values, interpol


def linear_function_approximation(x_values, y_values):
    n = len(x_values)
    xx = sympy.Symbol('x')
    s1, s2, s3, s4 = 0, 0, 0, 0
    for i in range(n):
        s1 += x_values[i] ** 2
        s2 += x_values[i]
        s3 += x_values[i] * y_values[i]
        s4 += y_values[i]

    c1 = (s3 * n - s2 * s4) / (s1 * n - s2 ** 2)
    c0 = (s1 * s4 - s2 * s3) / (s1 * n - s2 ** 2)
    yy = sympy.simplify(c0 + c1 * xx)

    f_values = [yy.subs(xx, x_values[i]) for i in range(n)]

    return x_values, y_values, f_values


def quadratic_function_approximation(x_values, y_values):
    n = len(x_values)
    xx = sympy.Symbol('x')
    s1, s2, s3, s4, s5, s6, s7 = 0, 0, 0, 0, 0, 0, 0
    for i in range(n):
        s1 += x_values[i] ** 4
        s2 += x_values[i] ** 3
        s3 += x_values[i] ** 2
        s4 += x_values[i]
        s5 += x_values[i] ** 2 * y_values[i]
        s6 += x_values[i] * y_values[i]
        s7 += y_values[i]

    a = [[0] * 3 for _ in range(3)]
    b = [0] * 3
    a[0][0], a[0][1], a[0][2] = s1, s2, s3
    a[1][0], a[1][1], a[1][2] = s2, s3, s4
    a[2][0], a[2][1], a[2][2] = s3, s4, n
    b[0], b[1], b[2] = s5, s6, s7

    c2, c1, c0 = list(np.linalg.solve(a, b))
    yy = sympy.simplify(c0 + c1 * xx + c2 * xx ** 2)

    f_values = [yy.subs(xx, x_values[i]) for i in range(n)]

    return x_values, y_values, f_values


def normal_distribution_approximation(x_values, y_values):
    n = len(x_values)

    def t3(value):
        return s.subs({a: value[0], b: value[1], c: value[2]})

    x, y, a, b, c = sympy.symbols('x y a b c')
    expr = (y - a * sympy.exp(-(x - b) ** 2 / (2 * c ** 2))) ** 2

    s = sympy.sympify(0)
    my = max(y_values)
    mx = x_values[y_values.index(my)]
    length = abs(max(x_values) - min(x_values))

    for i in range(len(x_values)):
        s += expr.subs({x: x_values[i], y: y_values[i]})

    bnds = ((-np.inf, np.inf), (-np.inf, np.inf), (0, None))
    res = scipy.optimize.minimize(t3, [my, mx, length], bounds=bnds)

    yy = sympy.simplify(
        res.x[0] * sympy.exp(-(x - res.x[1]) ** 2 / (2 * res.x[2] ** 2)))
    f_values = [yy.subs(x, x_values[i]) for i in range(n)]

    return x_values, y_values, f_values


def interpolate(x_values, y_values, method="lagr"):
    if method == "lagr":
        return lagrx(x_values=x_values, y_values=y_values)
    elif method == "newton1":
        return newtons_interpolation(x_values=x_values, y_values=y_values, is_forward=True)
    elif method == "newton2":
        return newtons_interpolation(x_values=x_values, y_values=y_values, is_forward=False)
    else:
        raise "Неверный метод"


def approximate(x_values, y_values, method='quadratic'):
    if method == "linear":
        return linear_function_approximation(x_values=x_values, y_values=y_values)
    elif method == "quadratic":
        return quadratic_function_approximation(x_values=x_values, y_values=y_values)
    elif method == "newton2":
        return normal_distribution_approximation(x_values=x_values, y_values=y_values)
    else:
        raise "Неверный метод"
