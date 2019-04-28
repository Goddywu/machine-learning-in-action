#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/28
# Desc: 


def load_data_set():
    data_matrix = []
    label_matrix = []
    with open('./testSet.txt') as f:
        lines = f.readlines()
        for line in lines:
            _list = line.strip().split()
            data_matrix.append([1.0, _list[0], _list[1]])
            label_matrix.append(int(_list[2]))
        return data_matrix, label_matrix


def t1():
    from logisticRegress5 import LogisticRegress
    data_matrix, label_matrix = load_data_set()
    a = LogisticRegress.gradient_ascent(data_matrix, label_matrix)
    print(a)


def t2():
    from logisticRegress5 import LogisticRegress
    data_matrix, label_matrix = load_data_set()
    a = LogisticRegress.stochastic_gradient_ascent(data_matrix, label_matrix)
    print(a)
    pass


def colic_t():
    from logisticRegress5 import LogisticRegress
    import numpy as np

    training_set = []
    training_labels = []
    with open('./HorseColicTraining.txt') as f:
        lines = f.readlines()
        for line in lines:
            current_line = line.strip().split('\t')
            line_array = []
            for i in range(21):
                line_array.append(float(current_line[i]))
            training_set.append(line_array)
            training_labels.append(float(current_line[-1]))
    model = LogisticRegress(training_set, training_labels)
    error_count = 0
    num_test = 0.0
    with open('./HorseColicTest.txt') as f:
        lines = f.readlines()
        for line in lines:
            num_test += 1
            current_line = line.strip().split('\t')
            line_array = []
            for i in range(21):
                line_array.append(float(current_line[i]))
            if int(int(current_line[-1]) != model.predict(np.array(line_array))):
                error_count += 1
    error_rate = float(error_count) / num_test
    print('the error rate is %f' % error_rate)
    return error_rate


def multi_t():
    num_test = 10
    error_sum = 0.0
    for k in range(num_test):
        error_sum += colic_t()
    print('after %d iterations, the average error rate is %f' % (num_test, error_sum / float(num_test)))


if __name__ == '__main__':
    # t1()
    # t2()
    multi_t()
