#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/6/5
# Desc:

from svm6.svm_core import SVM


def load_data_set(file_path):
    data_matrix = []
    label_matrix = []
    with open(file_path) as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            data_matrix.append([float(line[0]), float(line[1])])
            label_matrix.append(float(line[2]))
    return data_matrix, label_matrix


def t1():
    svm = SVM()
    data_matrix, label_matrix = load_data_set('./testSet.txt')
    b, alphas = svm.smo_simple(data_matrix, label_matrix, 0.6, 0.001, 40)

    for i in range(100):
        if alphas[i] > 0:
            print('{}, {}'.format(data_matrix[i], label_matrix[i]))


def t2():
    svm = SVM()
    data_matrix, label_matrix = load_data_set('./testSet.txt')
    b, alphas = svm.smo_full(data_matrix, label_matrix, 0.6, 0.001, 40)

    for i in range(100):
        if alphas[i] > 0:
            print('{}, {}'.format(data_matrix[i], label_matrix[i]))

    print('---w----')
    wt = SVM.calc_Wt(alphas, data_matrix, label_matrix)


def t3():
    data_matrix, label_matrix = load_data_set('./testSet.txt')
    svm = SVM.build(data_matrix, label_matrix, 0.6, 0.001, 40)
    a = svm.classify([3.542485, 1.977398])
    print(a)
    print(svm.classify([8.398012, 1.584918]))


def t4():
    import numpy as np

    data_matrix, label_matrix = load_data_set('./testSetRBF.txt')
    svm = SVM.build(data_matrix, label_matrix, 200, 0.0001, 10000, ('rbf', 1.3))
    m, n = np.shape(data_matrix)
    error_count = 0
    for i in range(m):
        a = svm.classify(data_matrix[i])
        b = np.sign(label_matrix[i])
        if np.sign(label_matrix[i]) != svm.classify(data_matrix[i]):
            error_count += 1
    print('the training error rate is {}'.format(float(error_count) / m))








if __name__ == '__main__':
    # t1()
    # t2()
    # t3()
    t4()
