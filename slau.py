from numpy import linalg
from random import uniform, randint
from cmath import sqrt as complex_sqrt
from math import sqrt as default_sqrt
import csv
import fractions

DELTA = 0.0001  # возможная погрешность при вычислении определителя


def get_matrix_from_keyboard():
    """
    Функция, обеспечивающая ввод матрицы с клавиатуры любых чисел (int, float, complex)
    :return: возвращается матрица
    """
    new_matrix = []
    while True:
        try:
            rows = int(input("Введите количество строк матрицы: "))
            cols = int(input("Введите количество столбцов матрицы: "))
            if rows > 0 and cols > 0:
                break
            else:
                print("Вводите только положительные числа!")
        except ValueError:
            print("Вводите только целые числа!")
    for i in range(rows):
        new_row = []
        for j in range(cols):
            while True:
                new_el = input("Введите новый элемент: ")
                if 'j' in new_el:
                    try:
                        new_row.append(complex(new_el))
                        break
                    except ValueError:
                        try:
                            pos_of_j = new_el.index('j')
                            j_part = new_el[:pos_of_j + 1]
                            num_part = new_el[pos_of_j + 1:]
                            if j_part.startswith('-'):
                                result = num_part + j_part
                            else:
                                result = num_part + '+' + j_part
                            new_row.append(complex(result))
                            break
                        except ValueError:
                            print("Вы неправильно ввели число!")
                elif '.' in new_el:
                    try:
                        new_row.append(float(new_el))
                        break
                    except ValueError:
                        print("Вы неправильно ввели число!")
                else:
                    try:
                        new_row.append(int(new_el))
                        break
                    except ValueError:
                        print("Вы неправильно ввели число!")
        new_matrix.append(new_row)
    return new_matrix


def generate_csv(n, type_of_matrix_elements):
    """
    Функция для генерации csv файла с числами
    :param n: передаётся размерность матрицы
    :param type_of_matrix_elements: передаётся тип данных, которые будут в матрице
    :return: None
    """
    with open('data.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        if isinstance(type_of_matrix_elements, int):
            data = [[randint(-100, 100) for _ in range(n)] for _ in range(n)]
        elif isinstance(type_of_matrix_elements, float):
            data = [[round(uniform(-100, 100), 2) for _ in range(n)] for _ in range(n)]
        filewriter.writerows(data)


def get_data_from_csv(file_name):
    """
    Функция для получения матрицы из csv файла с названием file_name
    :param file_name: передаётся название csv файла
    :return: возвращается матрица, как список списков
    """
    result_matrix = []
    with open(file_name, 'r') as csvfile:
        data = csv.reader(csvfile)
        for row in data:
            result_matrix.append(row)
    return result_matrix


def generate_matrix(rows, cols, type_of_matrix_elements):
    """
    Функция, генерирующая матрицу размера rows x cols с элементами типа type_of_matrix_elements
    :param rows: передаётся целое число строк
    :param cols: передаётся целое число столбцов
    :param type_of_matrix_elements: передаётся тип данных, которые будут в матрице
    :return: возвращается матрица, как список списков
    """
    if isinstance(type_of_matrix_elements, float):
        return [[round(uniform(-100, 100), 2) for _ in range(cols)] for _ in range(rows)]
    elif isinstance(type_of_matrix_elements, int):
        return [[randint(-100, 100) for _ in range(cols)] for _ in range(rows)]


def print_matrix(matrix):
    """
    Функция, выводящая матрицу построчно для более удобного восприятия
    :param matrix: Матрица чисел, заданная как список списков
    :return: None
    """
    for row in matrix:
        print(row)


def matrix_of_coefficients_is_square(matrix):
    """
    Функция, проверяющая является ли введённая матрица квадратной
    :param matrix: передаётся матрица чисел, заданная как список списков
    :return: возвращается True или False
    """
    return len(matrix) == len(matrix[0]) - 1


def minor(matrix, i, j):
    """
    Функция, возвращающая минор матрицы с вычеркнутым элементом matrix[i][j]
    :param matrix: передаётся матрица чисел, заданная как список списков
    :param i: номер вычеркиваемой строки
    :param j: номер вычеркиваемого столбца
    :return: возвращается матрица
    """
    return [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]


def det(matrix):
    """
    Функция, считающая определитель матрицы
    :param matrix: передаётся матрица чисел, заданная как список списков
    :return: возвращается число, равное определителю матрицы
    """
    if len(matrix) == 1:
        return matrix[0][0]
    elif len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]
    else:
        result = 0
        for row in range(len(matrix)):
            result += ((-1) ** row) * matrix[0][row] * det(minor(matrix, 0, row))
        return result


def get_matrix_of_coefficients(matrix):
    """
    Функция, возвращающая матрицу коэффициентов по заданной матрице системы уравнений
    :param matrix: передается матрица чисел, заданная как список списков
    :return: возвращается квадратная матрица
    """
    return [matrix[i][:-1] for i in range(len(matrix))]


def get_matrix_of_free_terms(matrix):
    """
    Функция, возвращающая список свободных членов по заданной матрице системы уравнений
    :param matrix: передается матрица чисел, заданная как список списков
    :return: возвращается список
    """
    return [matrix[i][-1] for i in range(len(matrix))]


def jacobis_method_of_simple_iterations(matrix):
    """
    Функция, возвращающая прямую матрицу коэффициентов (изначальную матрицу),
    обратную матрицу матрице коэффициентов (A - 1),
    решение СЛАУ (X) методом простых итераций Якоби
    :param matrix: передаётся матрица чисел, заданная как список списков
    :return: исходная матрица; матрица обратная матрице коэффициентов (А - 1); матрица решений СЛАУ
    """
    a = get_matrix_of_coefficients(matrix)
    b = get_matrix_of_free_terms(matrix)

    tempx = [0 for _ in range(len(a))]
    e = 0.01
    x = [0 for _ in range(len(a))]
    flag = True
    count = 0
    while flag or count > 100:
        for i in range(len(a)):
            if a[i][i] == 0:
                for j in range(i, len(a)):
                    if a[j][i] != 0:
                        a[j], a[i] = a[i], a[j]
                        b[j], b[i] = b[i], b[j]
                        break
            tempx[i] = b[i]
            count += 1
            for j in range(len(a[i])):
                if i != j:
                    count += 1
                    tempx[i] -= a[i][j] * x[j]
            tempx[i] /= a[i][i]
            count += 1
        for k in range(len(a)):
            if abs(x[k] - tempx[k]) < e:
                flag = False
            x[k] = tempx[k]

    return a, linalg.inv(a), x


def jordan_gauss_direct_algorithm(matrix):
    """
    Функция, возвращающая прямую матрицу коэффициентов (изначальную матрицу),
    обратную матрицу матрице коэффициентов (A - 1),
    решение СЛАУ (X) прямым алгоритмом Гаусса-Жордана
    :param matrix: передаётся матрица чисел, заданная как список списков
    :return: исходная матрица; матрица обратная матрице коэффициентов (А - 1); матрица решений СЛАУ
    """
    a = get_matrix_of_coefficients(matrix)
    x = [[0] for _ in range(len(matrix))]

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j:
                ratio = matrix[j][i] / matrix[i][i]
                for k in range(len(matrix) + 1):
                    matrix[j][k] = matrix[j][k] - ratio * matrix[i][k]

    for i in range(len(matrix)):
        x[i][0] = matrix[i][len(matrix)] / matrix[i][i]

    return a, linalg.inv(a), x


def direct_gauss_Jordan_algorithm_with_calculations_of_correct_fractions(matrix):
    """
    Функция, возвращающая прямую матрицу коэффициентов (изначальную матрицу),
    обратную матрицу матрице коэффициентов (A - 1),
    решение СЛАУ (X) прямым алгоритмом Гаусса-Жордана с вычислениями правильных дробей
    :param matrix: передаётся матрица чисел, заданная как список списков
    :return: исходная матрица; матрица обратная матрице коэффициентов (А - 1); матрица решений СЛАУ
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = fractions.Fraction(matrix[i][j])

    a = get_matrix_of_coefficients(matrix)
    x = [[0] for _ in range(len(matrix))]

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j:
                ratio = matrix[j][i] / matrix[i][i]
                for k in range(len(matrix) + 1):
                    matrix[j][k] = matrix[j][k] - ratio * matrix[i][k]

    for i in range(len(matrix)):
        x[i][0] = matrix[i][len(matrix)] / matrix[i][i]

    return a, linalg.inv(a), convert_from_fraction_to_float(x)


def convert_from_fraction_to_float(matrix):
    """Функция, конвертирующая элементы матрицы типа Fraction в тип float"""
    for i in range(len(matrix)):
        matrix[i] = float(matrix[i])
    return matrix


def infinite_norm(matrix):
    """
    Функция вычисления бесконечной нормы матрицы
    :param matrix: передаётся матрица чисел, заданная как список списков
    :return: возвращается число, равное бесконечной норме матрицы
    """
    total = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            total += matrix[i][j] ** 2
    return complex_sqrt(total) if isinstance(matrix[0][0], complex) else default_sqrt(total)


def transpose(m):
    """Функция транспонирования матрицы"""
    return map(list, zip(*m))


def inverse_matrix(m):
    """
    Функция вычисления обратной матрицы
    :param m: передается матрица чисел
    :return: возвращается матрица чисел, равная обратной матрице
    """
    determinant = det(m)
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    cofactors = []
    for r in range(len(m)):
        cofactorrow = []
        for c in range(len(m)):
            minor_of_matrix = minor(m, r, c)
            cofactorrow.append(((-1)**(r+c)) * det(minor_of_matrix))
        cofactors.append(cofactorrow)
    cofactors = transpose(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant

    return cofactors
