#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/6/11
# Desc: 这里根据书中写的，实现的是序列最小化SMO算法

import numpy as np
import random


class Svm:

    @staticmethod
    def build(data_matrix_in, label_matrix_in, C, tolerance_rate, max_iter):
        svm = Svm()
        svm.b, svm.alphas = svm.smo_simple(data_matrix_in, label_matrix_in, C, tolerance_rate, max_iter)
        return svm

    @staticmethod
    def smo_simple(data_matrix_in, label_matrix_in, C, tolerance_rate, max_iter):
        """
        简易SMO算法
        :param data_matrix_in: 数据集
        :param label_matrix_in: 类别标签
        :param C: 常数c
        :param tolerance_rate: 容错率
        :param max_iter: 退出前最大循环次数
        :return:
        """
        data_matrix = np.mat(data_matrix_in)
        label_matrix = np.mat(label_matrix_in).transpose()
        m, n = np.shape(data_matrix)
        b = 0
        alphas = np.mat(np.zeros((m, 1)))
        iter = 0
        while iter < max_iter:
            alpha_pairs_changed = 0
            for i in range(m):
                # 相当于 g(x_i)
                fXi = float(np.multiply(alphas, label_matrix).T * (data_matrix * data_matrix[i, :].T)) + b
                Ei = fXi - float(label_matrix[i])
                # 相当于第三步，判断条件是否满足
                if ((label_matrix[i] * Ei < -tolerance_rate) and (alphas[i] < C)) \
                        or ((label_matrix[i] * Ei > tolerance_rate) and (alphas[i] > 0)):
                    j = Svm.select_j_rand(i, m)
                    # 随机选择第二组
                    fXj = float(np.multiply(alphas, label_matrix).T * (data_matrix * data_matrix[j, :].T)) + b
                    Ej = fXj - float(label_matrix[j])
                    alphaIold = alphas[i].copy()
                    alphaJold = alphas[j].copy()
                    # 后边就不懂了，对应《统计学习方法-李航》的P126
                    if label_matrix[i] != label_matrix[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = max(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])
                    if L == H:
                        print('L == H')
                        continue
                    eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T \
                          - data_matrix[i, :] * data_matrix[i, :].T \
                          - data_matrix[j, :] * data_matrix[j, :].T
                    if eta > 0:
                        print(eta > 0)
                        continue
                    alphas[j] -= label_matrix[j] * (Ei - Ej) / eta
                    alphas[j] = Svm.clip_alpha(alphas[j], H, L)
                    if abs(alphas[j] - alphaJold) < 0.00001:
                        print('j not moving enough')
                        continue
                    alphas[i] += label_matrix[j] * label_matrix[i] * (alphaJold - alphas[j])
                    b1 = b - Ei - label_matrix[i] * (alphas[i] - alphaIold) * data_matrix[i, :] * data_matrix[i,:].T \
                        - label_matrix[j] * (alphas[j] - alphaJold) * data_matrix[i, :] * data_matrix[j, :].T
                    b2 = b - Ej - label_matrix[i] * (alphas[i] - alphaIold) * data_matrix[i, :] * data_matrix[j, :].T \
                        - label_matrix[j] * (alphas[j] - alphaJold) * data_matrix[j, :] * data_matrix[j, :].T
                    if 0 < alphas[i] < C:
                        b = b1
                    elif 0 < alphas[j] < C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2.0
                    alpha_pairs_changed += 1
                    print('iter: {} i:{}, pairs changed {}'.format(iter, i, alpha_pairs_changed))
            if alpha_pairs_changed == 0:
                iter += 1
            else:
                iter = 0
            print('iteration number: {}'.format(iter))
        return b, alphas

    @staticmethod
    def select_j_rand(i, m):
        """
        随机在区域范围内选择一个整数
        :param i: 不可以取的整数
        :param m: 范围
        """
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j

    @staticmethod
    def clip_alpha(aj, H, L):
        """
        数值太大时进行调整
        :param aj: 目标值
        :param H: 最大值
        :param L: 最小值
        """
        if aj > H:
            aj = H
        if aj < L:
            aj = L
        return aj
