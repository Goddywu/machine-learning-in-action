#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/24
# Desc:

import numpy as np
import operator


class Knn:

    def __init__(self):
        pass

    @staticmethod
    def classify(classify_vector, data_matrix, label_matrix, k):
        """
        利用欧式距离(各维度平方差的和)，开根号
        :param classify_vector: 待分类的向量 [1, 2]
        :param data_matrix: 基础数据 [ [1, 2], [1, 2] ]
        :param label_matrix: 基础数据的标注 [1, 0]
        :param k: 多少点中取频率最高
        :return:
        """
        diff_matrix = np.tile(classify_vector, (1, np.shape(data_matrix)[0]))
        square_diff_matrix = diff_matrix**2
        square_distance = square_diff_matrix.sum(axis=1)
        distance = square_distance**0.5
        sorted_distance = distance.argsort()
        class_count = {}
        for i in range(k):
            current_label = label_matrix[sorted_distance[i]]
            class_count[current_label] = class_count.get(current_label, 0) + 1
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    @staticmethod
    def auto_norm(data_set):
        """
        归一化特征值到：0-1   new = (old - min) / max
        """
        min_values = data_set.min(0)
        max_values = data_set.max(0)
        ranges = max_values - min_values
        # norm_data_set = np.zeros(shape=np.shape(data_set))
        m = data_set.shape[0]
        norm_data_set = (data_set - np.tile(min_values, (m, 1))) / np.tile(ranges, (m, 1))
        return norm_data_set, ranges, min_values


