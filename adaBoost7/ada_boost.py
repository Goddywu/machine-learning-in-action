#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/6/11
# Desc: 

import numpy as np


class AdaBoost:
    def __init__(self):
        pass

    @staticmethod
    def stump_classify(data_matrix, dimension, thresh_value, thresh_in_equal):
        ret_array = np.ones((np.shape(data_matrix)[0], 1))
        if thresh_in_equal == 'lt':
            ret_array[data_matrix[:, dimension] <= thresh_value] = -1.0
        else:
            ret_array[data_matrix[:, dimension] > thresh_value] = 1.0
        return ret_array

    @staticmethod
    def build_stump(data_matrix, class_labels, D):
        data_matrix = np.mat(data_matrix)
        label_matrix = np.mat(class_labels).T
        m, n = np.shape(data_matrix)
        num_steps = 10.0
        best_stump = {}
        best_classify_est = np.mat(np.zeros((m, 1)))
        min_error = np.inf
        for i in range(n):
            range_min = data_matrix[:, i].min()
            range_max = data_matrix[:, i].max()
            step_size = (range_max -  range_min) / num_steps
            for j in range(-1, int(num_steps) + 1):
                for in_equal in ['lt', 'gt']:
                    thresh_value = range_min + float(j) * step_size
                    predicted_values = AdaBoost.stump_classify(data_matrix, i, thresh_value, in_equal)
                    error_array = np.mat(np.ones((m, 1)))
                    error_array[predicted_values == label_matrix] = 0


