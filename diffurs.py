def parse_function(s: str):
    if "z'" in s:
        variable = 'z'
    else:
        variable = 'y'

    signs = ["+", "-"]

    # перенесем все из правой части в левую
    ind_of_equation_sign = s.index("=")
    result = s[:ind_of_equation_sign] + "-({})".format(s[ind_of_equation_sign + 1:]) + "=0" 

    # найдем левый и правый индексы слагаемого с '
    right_index = result.index("'")
    left_index = right_index
    for i in range(right_index, -1, -1):
        left_index = i
        if result[i] in signs:
            break
    
    # вырежем слагаемое с ' из строки
    y_derivative = result[left_index:right_index + 1]
    result = result[:left_index] + result[right_index + 1:]

    # перенесем слагаемое с ' направо
    ind_of_equation_sign = result.index("=")
    result = result[:ind_of_equation_sign + 1] + "-" + y_derivative

    # получим коэффициент при ' из строки
    ind_of_equation_sign = result.index("=")
    ind_of_y_derivative = result.index(f"{variable}'")
    coeff = result[ind_of_equation_sign + 1:ind_of_y_derivative]
    if coeff in signs or coeff == '--' or coeff == '++' or coeff == '-+' or coeff == '+-':
        coeff += '1'
    if '*' in coeff:
        coeff = coeff.replace("*", "")
    result = result[:ind_of_equation_sign + 1] + result[ind_of_y_derivative:]

    # разделим левую часть уравнения на найденный коэффициент
    ind_of_equation_sign = result.index("=")
    result = "({})/({})".format(result[:ind_of_equation_sign], coeff) + "=" + result[ind_of_equation_sign + 1:]

    # запишем уравнение в правильной форме
    ind_of_equation_sign = result.index("=")
    result = result[ind_of_equation_sign + 1:] + result[ind_of_equation_sign] + result[:ind_of_equation_sign]

    return result[result.index("=") + 1:]

def calculate(s, x, y, z=0):
    return eval(s)


def eulers_method(f, x0, y0, n, h):
    i_r, x_r, y_r = [], [], []
    for i in range(n):
        y = y0 + h * calculate(f, x0, y0)
        x = x0 + h
        i_r.append(i)
        x_r.append(x)
        y_r.append(y)
        x0 = x
        y0 = y
    return i_r, x_r, y_r

def euler_cauchy_method(f, x0, y0, n, h):
    i_r, x_r, y_r = [], [], []
    for i in range(n):
        y1 = y0 + h * calculate(f, x0, y0)
        x = x0 + h
        y = y0 + h / 2 * (calculate(f, x0, y0) + calculate(f, x, y1))
        i_r.append(i)
        x_r.append(x)
        y_r.append(y)
        x0 = x
        y0 = y
    return i_r, x_r, y_r

def runge_kutta_method(f, x0, y0, n, h):
    i_r, x_r, y_r = [], [], []
    for i in range(n):
        k1 = h * calculate(f, x0, y0)
        k2 = h * calculate(f, x0 + h / 2, y0 + k1 / 2)
        k3 = h * calculate(f, x0 + h / 2, y0 + k2 / 2)
        k4 = h * calculate(f, x0 + h, y0 + k3)
        y = y0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x = x0 + h
        i_r.append(i)
        x_r.append(x)
        y_r.append(y)
        x0 = x
        y0 = y
    return i_r, x_r, y_r


def runge_kutta_method_for_system(f1, f2, x0, y0, z0, n, h):
    i_r, x_r, y_r, z_r = [], [], [], []
    for i in range(n):
        k1 = h * calculate(f1, x0, y0, z0)
        l1 = h * calculate(f2, x0, y0, z0)
        k2 = h * calculate(f1, x0 + h / 2, y0 + k1 / 2, z0 + l1 / 2)
        l2 = h * calculate(f2, x0 + h / 2, y0 + k1 / 2, z0 + l1 / 2)
        k3 = h * calculate(f1, x0 + h / 2, y0 + k2 / 2, z0 + l2 / 2)
        l3 = h * calculate(f2, x0 + h / 2, y0 + k2 / 2, z0 + l2 / 2)
        k4 = h * calculate(f1, x0 + h, y0 + k3, z0 + l3)
        l4 = h * calculate(f2, x0 + h, y0 + k3, z0 + l3)
        y = y0 + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        z = z0 + 1 / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        x = x0 + h
        i_r.append(i)
        x_r.append(x)
        y_r.append(y)
        z_r.append(z)
        x0 = x
        y0 = y
        z0 = z
    return i_r, x_r, y_r, z_r

def get_inputs():
    isSystem = True if input("Хотите ли вы решать систему уравнений? y/n: ") == 'y' else False
    needTable = True if input("Вывести решение в виде таблицы? y/n: ") == 'y' else False
    if not(isSystem):
        function = input("Введите дифференциальное уравнение: ")
        x0, y0 = map(float, input("Введите через пробел начальные условия: ").split())
        a, b = map(float, input("Введите через пробел исследуемый интервал: ").split())
        n = 10000
        h = (b - a) / n
        methods = {'1': eulers_method, '2': euler_cauchy_method, '3': runge_kutta_method}
        choose_methods = input("Введите метод, которым хотите решить дифференциальное уравнение (1 - метод Эйлера, 2 - метод Эйлера-Коши, 3 - метод Рунге-Кутты): ")
        method = methods.get(choose_methods)
        print()
        return {'needTable': needTable, 'function_1': function, 'x0': x0, 'y0': y0, 'n': n, 'h': h, 'method': method, 'a': a, 'b': b, 'num_of_method': choose_methods}
    else:
        function_1 = input("Введите певрое дифференциальное уравнение: ")
        function_2 = input("Введите второе дифференциальное уравнение: ")
        x0, y0, z0 = map(float, input("Введите через пробел начальные условия: ").split())
        a, b = map(float, input("Введите через пробел исследуемый интервал: ").split())
        n = 10000
        h = (b - a) / n
        return {'needTable': needTable, 'function_1': function_1, 'function_2': function_2, 'x0': x0, 'y0': y0, 'z0': z0, 'n': n, 'h': h, 'a': a, 'b': b}

def main(needTable=False, function_1=None, function_2=None, x0=None, y0=None, z0=None, n=None, h=None, method=None, a=None, b=None, coeff=1, num_of_method=None):
    h /= coeff
    if function_2:
        i_lst, x_lst, y_lst, z_lst = runge_kutta_method_for_system(parse_function(function_1), parse_function(function_2), x0, y0, z0, n, h)
        makeApr_y = IntApr(x=x_lst, y=y_lst)
        makeApr_z = IntApr(x=x_lst, y=z_lst)
        print("Апроксимирующая функция МНК для y:", makeApr_y.quadratic_function_approximation())
        print("Апроксимирующая функция МНК для z:", makeApr_z.quadratic_function_approximation())
    else:
        i_lst, x_lst, y_lst = method(parse_function(function_1), x0, y0, n, h)

    return x_lst, y_lst, method


def accuracy_assessment(first_list, second_list, method='1'):
    if method in ['1', '2']:
        p = 2
    else:
        p = 4
    result = [abs(first_list[i] - second_list[i]) / (2 ** p - 1) for i in range(min(len(first_list), len(second_list)))]
    return statistics.mean(result)