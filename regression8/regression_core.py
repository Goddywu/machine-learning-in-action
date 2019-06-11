#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/6/10
# Desc: 

import numpy as np


class Regression:

    @staticmethod
    def rss_error(y_array, y_hat_array):
        return ((y_array - y_hat_array)**2).sum()

    @staticmethod
    def lwlr(test_point, x_array, y_array, k=1.0):
        """
        Locally Weight Linear Regression 局部加权线性回归
        :param test_point:
        :param x_array:
        :param y_array:
        :param k:
        :return:
        """
        x_matrix = np.mat(x_array)
        y_matrix = np.mat(y_array).T
        m = np.shape(x_matrix)[0]
        # 对角矩阵
        weights = np.mat(np.eye(m))
        for j in range(m):
            diff_matrix = test_point - x_matrix[j, :]
            weights[j, j] = np.exp(diff_matrix * diff_matrix.T / (-2.0 * k**2))
        xTx = x_matrix.T * (weights * x_matrix)
        if np.linalg.det(xTx) == 0.0:
            print('this matrix is singular, cannot do inverse')
            return
        ws = xTx.T * (x_matrix.T * (weights * y_matrix))
        return test_point * ws

    @staticmethod
    def lwlr_test(test_array, x_array, y_array, k=1.0):
        m = np.shape(test_array)[0]
        y_hat = np.zeros(m)
        for i in range(m):
            y_hat[i] = Regression.lwlr(test_array[i], x_array, y_array, k)
        return y_hat

    @staticmethod
    def stand_regress(x_array, y_array):
        """
        w最优 = (X^T · X)^-1 · X^T · y
        """
        x_matrix = np.mat(x_array)
        y_matrix = np.mat(y_array).T
        xTx = x_matrix.T * x_matrix
        if np.linalg.det(xTx) == 0.0:
            print('this matrix is singular, cannot do inverse')
            return
        ws = xTx.I * (x_matrix.T * y_matrix)
        return ws

