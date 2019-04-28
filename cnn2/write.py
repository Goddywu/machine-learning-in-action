#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/23
# Desc: 

from cnn2.knn import *
import numpy as np
import os


def img2vector(file_path):
    """
    32 x 32 => 1 x 1024
    """
    result_vector = np.zeros((1, 1024))
    with open(file_path, 'r') as f:
        for i in range(32):
            line = f.readline()
            for j in range(32):
                result_vector[0, 32 * i + j] = int(line[j])
        return result_vector


def guess_write_number():
    k=3
    labels = []
    training_file_name_list = os.listdir('./trainingDigits')
    train_num = len(training_file_name_list)
    train_matrix = np.zeros((train_num, 1024))
    for i in range(train_num):
        file_name = training_file_name_list[i]
        class_num = file_name.split('.')[0].split('_')[0]
        labels.append(class_num)
        train_matrix[i, :] = img2vector('./trainingDigits/{}'.format(file_name))
    test_file_name_list = os.listdir('./testDigits')
    test_num = len(test_file_name_list)
    error_count = 0
    for i in range(test_num):
        file_name = test_file_name_list[i]
        class_num = file_name.split('.')[0].split('_')[0]
        test_vector = img2vector('./testDigits/{}'.format(file_name))
        classifier_result = classify0(in_x=test_vector,
                                      data_set=train_matrix,
                                      labels=labels,
                                      k=k)
        print('predict {} while true is {}'.format(classifier_result, class_num))
        if classifier_result != class_num:
            error_count += 1
    print('total error rate is {}'.format(error_count / test_num))


if __name__ == '__main__':
    # a = img2vector('./testDigits/0_13.txt')
    guess_write_number()
    pass
