from numpy import arange, isinf, isnan


def func(function):
    """
    Функция, которая выполняет переданную функцию
    :params function: передается строка с функцией
    :return: возвращает выполненную строку с функцией
    """
    return eval(function)


def findDerivative(function, left_bound, right_bound, step):
    """
    Функция - генератор поиска производной
    :params function: передается строка с функцией
    :params left_bound, right_bound: левая и правая границы
    :params step: шаг
    :return: иттерационно возвращает точки в указанном интервале
    """
    for i in arange(left_bound, right_bound, step):
        yield (function(i + step) - function(i - step)) / (2 * step)


def findIntegration(function, left_bound, right_bound, step, except_points):
    """
    Функция поиска интеграла
    :params function: передается строка с функцией
    :params left_bound, right_bound: левая и правая границы
    :params step: шаг
    :params except_points: точки, где функция не существует
    :return: иттерационно возвращает точки в указанном интервале
    """
    summ = 0
    for i in arange(left_bound, right_bound, step):
        if round(i, 5) not in except_points:
            summ += function(i)
    summ += (function(left_bound) + function(right_bound)) / 2
    summ *= step
    return summ


def findExceptions(function, left_bound, right_bound):
    """
    Функция поиска точек, где функция не существует
    :params function: передается строка с функцией
    :params left_bound, right_bound: левая и правая границы
    :return: возвращает точки, где функция не существует
    """
    results = []
    for i in arange(round(left_bound, 2), round(right_bound, 2), 1e-2):
        fx = function(round(i, 2))
        if isnan(fx) or isinf(fx):
            results.append(round(i, 5))
    return results


def my_differentiation(left_bound, right_bound, step):
    """
    Функция - генератор поиска производной
    :params function: передается строка с функцией
    :params left_bound, right_bound: левая и правая границы
    :params step: шаг
    :return: списки точек в указанном интервле
    """
    list_of_iterations, list_of_x_values, list_of_diff_values = [], [], []
    counter = 0
    for x in arange(left_bound, right_bound + step, step):
        counter += 1
        try:
            diff = (func(x + step) - func(x - step)) / (2 * step)
        except ZeroDivisionError:
            continue
        list_of_iterations.append(counter)
        list_of_x_values.append(x)
        list_of_diff_values.append(diff)

    return list_of_iterations, list_of_x_values, list_of_diff_values
