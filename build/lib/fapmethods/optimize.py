import numpy as np
from random import randint
from collections import namedtuple
import pandas as pd
from numpy.random import choice as np_choice
import time


def timer(func):
    """
    Декоратор, выводящий время которое заняло
    выполнение декорируемой функции
    """

    def wrapped_func(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        ended_at = time.time()
        run_time = round(ended_at - started_at, 4)
        return *result, run_time
    return wrapped_func


class TSP:
    """
    Класс методов решения задачи комивояжора
    """
    def __init__(self, matrix):
        self.__matrix = matrix

    def __repr__(self):
        return repr(self.matrix)

    @property
    def n(self):
        """
        Возвращает число - количество вершин.
        :params: self
        :return: количество вершин графа.
        """
        return self.matrix.shape[0]

    @property
    def matrix(self):
        """
        Матрица весов
        :params: self
        :return: копия матрицы весов
        """
        return self.__matrix.copy()

    @classmethod
    def random(cls, n, bottom_limit=1, upper_limit=10, orientated=False):
        """
        Метод в классе, генерирующий связанный граф со случайными весами
        :params: n - размерность
        :params: bottom_limit - минимальный вес ребра
        :params: upper_limit - максимальный вес ребра
        :params: orientated - ориентированный ли граф
        :return: Возвращается созданный объект класса
        """
        # method = uniform
        method = randint
        matrix = np.empty((n, n))

        if orientated:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        matrix[i][j] = np.Inf
                    else:
                        matrix[i][j] = method(bottom_limit, upper_limit)

        else:
            for i in range(n):
                for j in range(i, n):
                    if i == j:
                        matrix[i][j] = np.Inf
                    else:
                        matrix[i][j] = method(bottom_limit, upper_limit)
                        matrix[j][i] = matrix[i][j]

        return cls(matrix)

    @classmethod
    def keybord_input(cls, n):
        """
        Метод в классе, позволяющий ввести матрицу с клавиатуры
        :params: n - размерность
        :return: Возвращается созданный объект класса
        """
        matrix = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = np.Inf
                else:
                    matrix[i][j] = float(input())

        return cls(matrix)

    @timer
    def branchNBound(self):
        """
        Функция, реализующая метод ветвей и границ
        :params: self
        :return: вычисленный итоговый путь + его длина
        """
        n_rows = self.matrix.shape[0]
        n_cols = self.matrix.shape[1]
        table = pd.DataFrame(self.matrix,
                             columns=list(range(n_cols)),
                             index=list(range(n_rows)))
        leaf = namedtuple('Leaf', ['table',
                                   'bottom_bound',
                                   'included',
                                   'edge'
                                   ]
                          )

        def path(edges):
            """
            Функция нахождения пути
            :params: edges like list (?)
            :return: найденный list
            """
            edges = dict(edges)
            v_set = set(edges.keys())
            lst = []
            loop_flag = False
            while len(v_set) > 0:
                lst.append([v_set.pop()])
                while True:
                    try:
                        v = edges[lst[-1][-1]]
                        v_set -= {v}
                        lst[-1].append(v)
                        if lst[-1][0] == lst[-1][-1]:
                            loop_flag = True
                            break
                    except KeyError:
                        break
            if loop_flag and len(lst) > 1:
                return []
            return lst[0]

        def score_func(table_):
            """
            Целевая функция
            :params: table_ таблица весов
            :return: приведенная таблица весов, нижняя границу
            """
            assert table_.shape[0] > 0
            table = table_.copy()

            matrix = np.array(table)

            bottom_bound = 0
            for i in range(matrix.shape[0]):
                min_ = min(matrix[i])
                if min_ == np.inf:
                    return None, np.inf
                bottom_bound += min_
                matrix[i] -= min_
            for i in range(matrix.shape[1]):
                min_ = min(matrix[:, i])
                if min_ == np.inf:
                    return None, np.inf
                bottom_bound += min_
                matrix[:, i] -= min_

            table[:][:] = matrix
            assert table.shape[0] > 0
            return table, bottom_bound

        def max_penalty(table):
            """
            функция нахождения самого "тяжелого" нуля
            :params: table (см. выше)
            :return: ребра с самым тежелым ребром
            """
            assert table.shape[0] > 0
            max_zero = 0  # самый тяжелый ноль
            max_zero_edge = (0, 0)
            matrix = np.array(table.copy())

            for i, j in zip(*np.where(matrix == 0)):  # находим индексы нулей
                matrix[i, j] = np.Inf
                assert min(matrix[i]) >= 0 and min(matrix[j]) >= 0
                d = min(matrix[i]) + min(matrix[j])
                if d >= max_zero:
                    max_zero = d
                    max_zero_edge = (i, j)
                matrix[i, j] = 0

            max_zero_edge = (table.index[max_zero_edge[0]],
                             table.columns[max_zero_edge[1]])
            return max_zero_edge

        def include(leaf_):
            """
            функция включения ребра в путь
            :params: объект содержащий матрицу весов, ребро для включения, включенние ребра, нижнюю границу
            :return: объект с включенним ребром
            """
            assert leaf_.table.shape[0] > 0
            table = leaf_.table.copy()
            i, j = leaf_.edge
            table.drop(i, 'index', inplace=True)  # вычеркиваем строку
            table.drop(j, 'columns', inplace=True)  # и столбец
            table, bottom_bound = score_func(table)  # приведение матрицы
            # сложение нижней границы и константы приведения
            bottom_bound += leaf_.bottom_bound
            # добавляем старое ребро в путь
            included = leaf_.included + [leaf_.edge]
            if bottom_bound == np.inf:
                return None
            edge = max_penalty(table)[0]  # ищем следующее ребро
            # путь замкнулся раньше времени
            if len(path(included + [edge])) == 0:
                return None
            return leaf(table, bottom_bound, included, edge)

        def exclude(leaf_):
            """
            функция исключения ребра из пути
            :params: объект содержащий матрицу весов, ребро для исключения, включенные ребра, нижнюю границу
            :return: объект с включенним ребром
            """

            assert leaf_.table.shape[0] > 0
            table = leaf_.table.copy()
            i, j = leaf_.edge
            table.loc[i, j] = np.Inf
            table, bottom_bound = score_func(table)  # приведение матрицы
            # сложение нижней границы и константы приведения
            bottom_bound += leaf_.bottom_bound
            if bottom_bound == np.inf:
                return None
            included = leaf_.included  # оставляем путь таким же
            edge = max_penalty(table)[0]  # ищем следующее ребро
            return leaf(table, bottom_bound, included, edge)

        table_, bottom_bound_ = score_func(table)
        edge = max_penalty(table_)[0]
        tree_leaves = [leaf(table_, bottom_bound_, [], edge)]
        ii = 0
        while True:
            ii += 1
            # эвристика выбора следующего ребра
            tree_leaves = sorted(tree_leaves, key=lambda x: x.bottom_bound)
            min_leaf_check = min_leaf = tree_leaves[0]
            min_leaf_index = 0
            for i, l in enumerate(tree_leaves):
                if min_leaf.bottom_bound != l.bottom_bound:
                    break
                if l.table.shape[0] < min_leaf.table.shape[0]:
                    min_leaf_index = i

            # деление на включенный и запрещенный
            min_leaf = tree_leaves.pop(min_leaf_index)

            # условие остановки
            if min_leaf.table.shape[0] == 1:
                break
            temp = include(min_leaf)
            if temp is not None:
                tree_leaves.append(temp)
            temp = exclude(min_leaf)
            if temp is not None:
                tree_leaves.append(temp)

        return path(min_leaf.included + [min_leaf.edge]), min_leaf.bottom_bound, ii

    def countPathCost(self, path):
        """
        функция поиска "стоимости" пути
        :params: путь, найденный выше
        :return: путь, стоимость пути
        """
        c = 0
        i = 0
        try:
            for i in range(1, len(path)):
                # print(path[i-1], path[i])
                c += self.matrix[path[i - 1]][path[i]]
        except:
            print(path[i - 1], [path[i]])
        return c

    @timer
    def Boltsman(self, t0=1000):
        """
        Реализация метода имитации Больцмановского отжига
        :params: t0 начальная температура
        :return: стоимость пути (число)
        """
        def t(k):
            return t0 / (k**0.8)

        def isTransit(x0, x, ct):
            df = self.countPathCost(x) - self.countPathCost(x0)
            if df <= 0:
                return True
            else:
                if np.exp(-df / ct) >= np.random.uniform():
                    return True

                return False

        def getNewState(x):
            x = x.copy()
            r1 = np.random.randint(0, len(x) - 2)
            r2 = np.random.randint(0, len(x) - 2)
            while r1 == r2:
                r2 = np.random.randint(0, len(x) - 1)
            x[r1], x[r2] = x[r2], x[r1]
            x[-1] = x[0]
            return x

        ct = t(1)
        x = [i for i in range(0, len(self.matrix))] + [0]

        counter = 1

        while counter <= 10 ** 4:
            x_new = getNewState(x)
            if isTransit(x, x_new, ct):
                x = x_new
            counter += 1
            ct = t(counter)
        return x, self.countPathCost(x)

    @timer
    def AntColonyMethod(self, n_ants=None, n_best=5, n_iterations=None):
        """
        Реализация Муравьиного алгоритма
        :params n_ants: кол-во муравьев
        :params n_best: кол-во путей сохраняемых на каждой итерации
        :params n_iterations: кол-во итераций

        :return: кратчайший цикл, длина цикла
        """
        if n_ants:
            n_ants = self.n
        if n_iterations:
            n_iterations = self.n * 4
        decay = 0.95
        alpha = 1
        beta = 1

        class AntColony:
            """
            Класс описывающий муравьиную колонию
            """
            def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha, beta):

                self.distances = distances
                self.pheromone = np.ones(
                    self.distances.shape) / len(self.distances)
                self.all_inds = range(len(self.distances))
                self.n_ants = n_ants
                self.n_best = n_best
                self.n_iterations = n_iterations
                self.decay = decay
                self.alpha = alpha
                self.beta = beta

            def run(self):
                """
                Функция вычисления кратчайшего цикла в графе
                :returns: кратчайший из найденных путей и его длина

                """
                shortest_path = None
                all_time_shortest_path = ("placeholder", np.inf)
                for i in range(self.n_iterations):
                    all_paths = self.gen_all_paths()
                    self.spread_pheronome(
                        all_paths, self.n_best, shortest_path=shortest_path)
                    shortest_path = min(all_paths, key=lambda x: x[1])
                    if shortest_path[1] < all_time_shortest_path[1]:
                        all_time_shortest_path = shortest_path
                    self.pheromone *= self.decay
                return self.route_conversion(all_time_shortest_path[0]), all_time_shortest_path[1]

            def route_conversion(self, lst):
                """
                Преобразует набор ребер в путь
                :params lst: набор ребер
                :returns: последовательность из вершин
                """
                result = [*lst[0]]
                for i in range(1, len(lst)):
                    result.append(lst[i][1])
                return result

            def spread_pheronome(self, all_paths, n_best, shortest_path):
                """
                Считает значения феромона на ребрах
                :params all_paths: сгенерированные пути
                :params n_best: количесво лучших путей
                """
                sorted_paths = sorted(all_paths, key=lambda x: x[1])
                for path, dist in sorted_paths[:n_best]:
                    for move in path:
                        self.pheromone[move] += 1.0 / self.distances[move]

            def gen_path_dist(self, path):
                """
                :params path: Путь
                :returns: длина пути
                """
                total_dist = 0
                for ele in path:
                    total_dist += self.distances[ele]
                return total_dist

            def gen_all_paths(self):
                """
                Создает n_ants путей
                :returns: список путей
                """
                all_paths = []
                for i in range(self.n_ants):
                    path = self.gen_path(0)
                    all_paths.append((path, self.gen_path_dist(path)))
                return all_paths

            def gen_path(self, start):
                """
                Сгенерировать путь
                :params start: точка начала пути
                :returns: замнкутый поуть по всем вершинам
                """
                path = []
                visited = set()
                visited.add(start)
                prev = start
                for i in range(len(self.distances) - 1):
                    move = self.pick_move(
                        self.pheromone[prev], self.distances[prev], visited)
                    path.append((prev, move))
                    prev = move
                    visited.add(move)
                path.append((prev, start))
                return path

            def pick_move(self, pheromone, dist, visited):
                """
                Выбор хода
                :returns: следующая вершина
                """
                pheromone = np.copy(pheromone)
                pheromone[list(visited)] = 0
                row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
                norm_row = row / row.sum()
                move = np_choice(self.all_inds, 1, p=norm_row)[0]
                return move

        return AntColony(self.matrix, n_ants, n_best, n_iterations, decay, alpha, beta).run()
