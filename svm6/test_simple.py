#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/6/12
# Desc:

from svm6.svm_core_simple import Svm
import numpy as np


def load_data_set(file_path):
    data_matrix = []
    label_matrix = []
    with open(file_path) as f:
        for line in f.readlines():
            line_array = line.strip().split('\t')
            data_matrix.append(
                [float(line_array[0]), float(line_array[1])])
            label_matrix.append(float(line_array[2]))
    return data_matrix, label_matrix


def t1():
    data_matrix, label_matrix = load_data_set('./testSet.txt')
    svm = Svm().build(data_matrix, label_matrix, 0.6, 0.001, 40)
    print(svm.b)
    print(svm.alphas[svm.alphas > 0])


def t2():
    svm = Svm()
    data_matrix, label_matrix = load_data_set('./testSet.txt')
    b, alphas = svm.smo_simple(data_matrix, label_matrix, 0.6, 0.001, 40)

    for i in range(100):
        if alphas[i] > 0:
            print('{}, {}'.format(data_matrix[i], label_matrix[i]))


if __name__ == '__main__':
    t1()
    # a = np.mat([[2, 3]])
    # b = np.mat([[1], [1]])
    # c = np.mat([[1, 1]])
    # print(np.multiply(a, b))
    # print(np.multiply(a, c))

    # t2()
