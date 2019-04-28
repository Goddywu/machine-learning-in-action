#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/23
# Desc: 


import numpy as np
import operator


def classify0(in_x, data_set, labels, k):
    """
    k-近邻算法，欧式距离(平方差)
    :param in_x: 输入向量
    :param data_set: 训练样本的特征向量
    :param labels: 训练样本的目标向量
    :param k: 最近邻居的数量
    :return: 预测的类
    """
    data_set_size = data_set.shape[0]
    # tail：扩展矩阵 https://blog.csdn.net/weixin_38656890/article/details/80198749
    diff_matrix = np.tile(in_x, (data_set_size, 1)) - data_set
    square_diff_matrix = diff_matrix ** 2
    square_distances = square_diff_matrix.sum(axis=1)
    distances = square_distances ** 0.5
    sorted_distances_index = distances.argsort()
    class_count = {}
    for i in range(k):
        current_label = labels[sorted_distances_index[i]]
        class_count[current_label] = class_count.get(current_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(file_path):
    with open(file_path, 'r') as f:
        array0lines = f.readlines()
        data_set_matrix = np.zeros((len(array0lines), 3))
        class_label_vector = []
        for index, line in enumerate(array0lines):
            line = line.strip()
            list0line = line.split('\t')
            data_set_matrix[index, :] = list0line[0: 3]
            class_label_vector.append(int(list0line[-1]))
        return data_set_matrix, class_label_vector


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
