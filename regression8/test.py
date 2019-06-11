#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/6/10
# Desc:

from regression8.regression_core import Regression


def load_data_set(file_name):
    num_feature = len(open(file_name).readline().split('\t')) - 1
    data_matrix = []
    label_matrix = []
    with open(file_name) as f:
        for line in f.readlines():
            line_array = []
            current_line = line.strip().split('\t')
            for i in range(num_feature):
                line_array.append(float(current_line[i]))
            data_matrix.append(line_array)
            label_matrix.append(float(current_line[-1]))
    return data_matrix, label_matrix


def t1():
    x_array, y_array = load_data_set('ex0.txt')
    ws = Regression.stand_regress(x_array, y_array)
    print(ws)


if __name__ == '__main__':
    t1()
