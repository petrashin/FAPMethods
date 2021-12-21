import numpy
import math
import re
from typing import List


class DIYMatrix:
    def __init__(self, ncols: int, nrows: int, elements: List[List[int]]):
        self.matrix = elements
        self._ncols = ncols
        self._nrows = nrows

    def transpose(self):
        t_matrix = []
        for i in range(self._ncols):
            row = []
            for j in range(self._nrows):
                row.append(self.matrix[j][i])
            t_matrix.append(row)
        return DIYMatrix(self._nrows, self._ncols, t_matrix)

    def parse_complex(self, s: str):
        s = s.replace(" ", "")
        pattern_j = re.compile(r"[-]*\d*.\d*j")
        pattern_int = re.compile(r"[+-]\d*$")
        try:
            return complex(s)
        except:
            i = re.search(pattern_int, s)[0]
            j = re.search(pattern_j, s)[0]
            sign = '' if j[0] == '-' else "+"
            print(f"{i}{sign}{j}")
            return complex(f"{i}{sign}{j}")

    def matrix_to_type(self, newType: type):
        if isinstance(self.matrix[0][0], str) and newType != complex:
            return DIYMatrix(self._ncols, self._nrows, [list(map(lambda x: newType(x.replace(" ", "")), row)) for row in self.matrix])
        elif isinstance(self.matrix[0][0], str) and newType == complex:
            return DIYMatrix(self._ncols, self._nrows, [list(map(lambda x: self.parse_complex(x), row)) for row in self.matrix])
        return DIYMatrix(self._ncols, self._nrows, [list(map(newType, row)) for row in self.matrix])

    def to_min_type(self):
        """
            Сonverts matrix to minimal type according to the following gradation: int < float < complex < str
        """
        try:
            res_f = self.matrix_to_type(float)
            try:
                return self.matrix_to_type(int)
            except Exception as e:
                return res_f
        except:
            try:
                return self.matrix_to_type(complex)
            except Exception as e:
                return self

    def read_from_csv(filename: str):
        """
            Read matrix from csv
            Input: filename or fullpath to file
            Output: matrix of 'str' type
        """
        with open(filename) as csvf:
            l = csvf.readline()
            if l == '':
                raise ValueError(
                    "Количество элементов в матрице не может быть 0")

            elements = [list(l.rstrip("\n").split(";"))]
            n_cols = len(elements[0])

            for line in csvf:
                line = line.rstrip("\n")
                row = list(line.split(";"))
                assert len(
                    row) == n_cols, "Кол-во элементов в строках матрицы не совпадает"
                elements.append(row)

        res = DIYMatrix(n_cols, len(elements), elements)
        res.to_min_type()
        return res

    def write_to_csv(self, filename: str):
        """
            Writes matrix to file in csv format
        """
        assert isinstance(self, DIYMatrix)
        with open(filename, "w") as f:
            for row in self.matrix:
                f.write(";".join(map(str, row))+"\n")

    def det(self) -> int:
        if isinstance(self.matrix[0][0], str):
            raise ValueError(
                "Определитель можно найти только для численных матриц")
        else:
            assert self._ncols == self._nrows, "Матрица не квадратная"
            if self._ncols == 1:
                return self.matrix[0][0]
            elif self._ncols == 2:
                m = self.matrix
                return m[0][0]*m[1][1] - m[0][1]*m[1][0]
            else:
                m = self.matrix
                temp = list(zip(*m[1:]))
                result = 0
                for i in range(self._ncols):
                    minor = DIYMatrix(self._ncols-1, self._nrows-1,
                                      list(zip(*temp[:i], *temp[i+1:])))
#                 print(f"minor{i}: \n{minor}")
                    t = ((-1)**i)*m[0][i]*minor.det()
#                 print(t)
                    result += t
                return round(result)

    def first_norm(self) -> int:
        maxCol = sum(self.matrix[0])
        for j in range(self._ncols):
            s = 0
            for i in range(self._nrows):
                s += self.matrix[i][j]
            if s > maxCol:
                maxCol = s
        return maxCol

    def second_norm(self) -> int:
        s = 0
        for i in range(self._ncols):
            for j in range(self._nrows):
                s += self.matrix[i][j] ** 2
        return s**0.5

    def third_norm(self) -> int:
        mS = sum(self.matrix[0])
        for i in self.matrix:
            s = sum(i)
            if s > mS:
                mS = s
        return mS

    def __add__(self, another):
        assert self._ncols == another._ncols, "Кол-во столбцов не совпадает"
        assert self._nrows == another._nrows, "Кол-во строк не совпадает"
        assert isinstance(
            self.matrix[0][0], (int, float, complex)), "Левая матрица не числового типа"
        assert isinstance(
            another.matrix[0][0], (int, float, complex)), "Правая матрица не числового типа"

        result = []
        for i in range(self._nrows):
            temp = []
            for j in range(self._ncols):
                temp.append(self.matrix[i][j] + another.matrix[i][j])
            result.append(temp)
        return DIYMatrix(self._nrows, self._ncols, result)

    def __sub__(self, another):
        assert self._ncols == another._ncols, "Кол-во столбцов не совпадает"
        assert self._nrows == another._nrows, "Кол-во строк не совпадает"
        assert isinstance(
            self.matrix[0][0], (int, float, complex)), "Левая матрица не числового типа"
        assert isinstance(
            another.matrix[0][0], (int, float, complex)), "Правая матрица не числового типа"

        result = []
        for i in range(self._nrows):
            temp = []
            for j in range(self._ncols):
                temp.append(self.matrix[i][j] - another.matrix[i][j])
            result.append(temp)
        return DIYMatrix(self._ncols, self._nrows, result)

    def __mul__(self, another):
        if isinstance(another, DIYMatrix):
            assert self._ncols == another._nrows, "Матрицы несовместимы, кол-во столбцов первой матрицы и кол-во строк второй не совпадают"
            assert isinstance(
                self.matrix[0][0], (int, float, complex)), "Левая матрица не числового типа"
            assert isinstance(
                another.matrix[0][0], (int, float, complex)), "Правая матрица не числового типа"

            return DIYMatrix(self._nrows, another._ncols, [[sum(k*m for (k, m) in zip(i, j)) for i in another.transpose().matrix] for j in self.matrix])
        elif isinstance(another, (int, float, complex)):
            result = []
            for i in range(self._nrows):
                t = []
                for j in range(self._ncols):
                    t.append(self.matrix[i][j] * another)
                result.append(t)
            return DIYMatrix(self._nrows, self._ncols, result)

    @classmethod
    def keyboard_input(self):
        while True:
            try:
                ncols = int(input("Введите количество столбцов матрицы:"))
                assert ncols > 0, "Кол-во столбцов должно быть больше 0"

                nrows = int(input("Введите количество строк матрицы:"))
                assert nrows > 0, "Кол-во строк должно быть больше 0"
                break
            except Exception as err:
                #           print("Введите корректное количество столбцов и строк")
                print(err)

        res = []
        for i in range(nrows):
            t = []
            for j in range(ncols):
                temp = input("Введите новый элемент: ")
                t.append(temp)
            print(t)
            res.append(t)
        return DIYMatrix(ncols, nrows, res).to_min_type()

    def __str__(self):
        return "\n".join(list(map(str, self.matrix)))

    __repr__ = __str__
    __rmul__ = __mul__

    def minor(self, i, j):
        """Функция, возвращающая минор матрицы с вычеркнутым элементов matrix[i][j]"""
        return DIYMatrix(ncols=self._ncols-1, nrows=self._nrows-1, elements=[row[:j] + row[j + 1:] for row in (self.matrix[:i] + self.matrix[i + 1:])])
