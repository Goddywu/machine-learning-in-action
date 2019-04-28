#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/28
# Desc: 


class LogisticRegress:
    def __init__(self, data_matrix_in, class_labels_in):
        """

        :param data_matrix_in:
            [
                [1, 2, 3, ...],
                []
            ]
        :param class_labels_in:
            [0, 1, 0, 1]
        """
        self.data_matrix = data_matrix_in
        self.class_labels = class_labels_in
        # 梯度上升算法
        self.weights = LogisticRegress.gradient_ascent(data_matrix_in, class_labels_in, alpha=0.001, max_cycles=500)
        # 随机梯度上升算法
        self.weights = LogisticRegress.stochastic_gradient_ascent(data_matrix_in, class_labels_in, max_cycles=500)
        pass

    def predict(self, vector):
        return LogisticRegress.classify_vector(vector, self.weights)

    """ --- utils --- """

    @staticmethod
    def classify_vector(in_x, weights):
        """

        :param in_x: [1, 2, ...]
        :param weights: [[2], ...]
        :return:
        """
        prob = LogisticRegress.sigmoid(sum(in_x * weights))
        if prob > 0.5:
            return 1.0
        else:
            return 0.0

    @staticmethod
    def sigmoid(in_x):
        import numpy as np
        # 1 / (1 + e^(-z))
        return 1.0 / (1 + np.exp(-in_x))

    @staticmethod
    def gradient_ascent(data_matrix_in, class_labels_in, alpha=0.001, max_cycles=500):
        """
        梯度上升，计算最优
        :param data_matrix_in:
            [
                [1, 2, 3, ...],
                []
            ]
        :param class_labels_in:
            [0, 1, 0, 1]
        :param alpha 步长
        :param max_cycles 轮次
        :return:
        """
        import numpy as np

        # 转换为矩阵
        data_matrix = np.mat(data_matrix_in, dtype=np.float64)
        # 转换为列向量
        label_matrix = np.mat(class_labels_in, dtype=np.float64).transpose()
        # m: vector num, n: variable num
        m, n = np.shape(data_matrix)
        weights = np.ones((n, 1))
        for k in range(max_cycles):
            h = LogisticRegress.sigmoid(data_matrix * weights)
            # 列向量， 0：相等 / 1：预期是1，结果是0 / -1：预期是0，结果是1
            error = (label_matrix - h)
            # 往理想状态移动步长
            weights = weights + alpha * data_matrix.transpose() * error
        return weights.getA()

    @staticmethod
    def stochastic_gradient_ascent(data_matrix_in, class_labels_in, max_cycles=500):
        """
        随机梯度上升
        :return:
        """
        import numpy as np
        import random

        data_matrix_in = np.array(data_matrix_in, dtype=np.float64)
        m, n = np.shape(data_matrix_in)
        weights = np.ones(n)
        for i in range(max_cycles):
            data_index = list(range(m))
            for j in range(m):
                alpha = 4 / (1.0 + i + j) + 0.01
                random_index = int(random.uniform(0, len(data_index)))
                h = LogisticRegress.sigmoid(sum(data_matrix_in[random_index] * weights))
                error = class_labels_in[random_index] - h
                weights = weights + alpha * error * data_matrix_in[random_index]
                del(data_index[random_index])
        return weights
