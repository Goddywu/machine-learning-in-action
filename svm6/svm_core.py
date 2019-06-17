#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/6/17
# Desc: 

import numpy as np
import random


class OptStruct:
    def __init__(self, data_matrix_in, class_labels, C, toler, k_trup):
        self.X = data_matrix_in
        self.label_matrix = class_labels
        self.C = C
        self.toler = toler
        self.m = np.shape(data_matrix_in)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 误差缓存
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = self.kernel_trains(self.X, self.X[i, :], k_trup)

    @staticmethod
    def kernel_trains(X, A, k_tup):
        m, n = np.shape(X)
        K = np.mat(np.zeros((m, 1)))
        if k_tup[0] == 'lin':
            K = X * A.T
        elif k_tup[0] == 'rbf':
            for j in range(m):
                delta_row = X[j, :] -A
                K[j] = delta_row * delta_row.T
            K = np.exp(K / (-1 * k_tup[1] ** 2))
        else:
            raise NameError('That kernel is not recognized!')
        return K


class Svm:
    def __init__(self):
        self.Wt = None
        self.b = None

    @staticmethod
    def build(data_matrix_in, class_labels, C, toler, max_iter, k_tup=('lin', 0)):
        """

        :param data_matrix_in: 数据集
        :param class_labels: 类别标签
        :param C: 常数c
        :param toler: 容错率
        :param max_iter: 退出前最大循环次数
        """
        svm = Svm()
        svm.b, svm.alphas = svm.smo_full(data_matrix_in, class_labels, C, toler, max_iter, k_tup)
        svm.Wt = svm.calc_Wt(svm.alphas, data_matrix_in, class_labels)
        return svm

    def classify(self, data_array):
        data_matrix = np.mat(data_array)
        tmp = data_matrix*np.mat(self.Wt) + self.b
        # if float(tmp) > 0:
        #     return 1
        # elif float(tmp) < 0:
        #     return -1
        # else:
        #     return 0
        return np.sign(float(tmp))

    @staticmethod
    def calc_Wt(alphas, data_array, class_labels):
        X = np.mat(data_array)
        label_matrix = np.mat(class_labels).transpose()
        m, n = np.shape(X)
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(alphas[i] * label_matrix[i], X[i, :].T)
        return w

    # ----- 下面是完整提速版的smo算法 -----

    @staticmethod
    def smo_full(data_matrix_in, class_labels, C, toler, max_iter, k_tup=('lin', 0)):
        oS = OptStruct(np.mat(data_matrix_in), np.mat(class_labels).transpose(), C, toler, k_tup)
        iter = 0
        entire_set = True
        alpha_pairs_changed = 0
        while (iter < max_iter) and ((alpha_pairs_changed > 0) or (entire_set)):
            alpha_pairs_changed = 0
            if entire_set:
                for i in range(oS.m):
                    alpha_pairs_changed += Svm.inner_L(i, oS)
                    print('full set, iter: {} i {}, pairs changed {}'.format(iter, i, alpha_pairs_changed))
                iter += 1
            else:
                non_bound_Is = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
                for i in non_bound_Is:
                    alpha_pairs_changed += Svm.inner_L(i, oS)
                    print('non-bound, iter: {} i {}, pairs changed {}'.format(iter, i, alpha_pairs_changed))
                iter += 1
            if entire_set:
                entire_set = False
            elif alpha_pairs_changed == 0:
                entire_set = True
            print('iteration number: {}'.format(iter))
        return oS.b, oS.alphas

    @staticmethod
    def clac_Ek(oS: OptStruct, k):
        fXk = float(np.multiply(oS.alphas, oS.label_matrix).T * oS.K[:, k] + oS.b)
        Ek = fXk - float(oS.label_matrix[k])
        return Ek

    @staticmethod
    def select_J(i, oS: OptStruct, Ei):
        max_K = -1
        max_delta_e = 0
        Ej = 0
        oS.eCache[i] = [1, Ei]
        valid_eCache_list = np.nonzero(oS.eCache[:,0].A)[0]
        if len(valid_eCache_list) > 1:
            for k in valid_eCache_list:
                if k == i:
                    continue
                Ek = Svm.clac_Ek(oS, k)
                delta_E = abs(Ei - Ek)
                if delta_E > max_delta_e:
                    max_K = k
                    max_delta_e = delta_E
                    Ej = Ek
            return max_K, Ej
        else:
            j = Svm.select_j_rand(i, oS.m)
            Ej = Svm.clac_Ek(oS, j)
            return j, Ej

    @staticmethod
    def update_Ek(oS: OptStruct, k):
        Ek = Svm.clac_Ek(oS, k)
        oS.eCache[k] = [1, Ek]

    @staticmethod
    def inner_L(i, oS: OptStruct):
        Ei = Svm.clac_Ek(oS, i)
        if ((oS.label_matrix[i]*Ei < -oS.toler) and (oS.alphas[i] < oS.C)) \
                or ((oS.label_matrix[i] * Ei > oS.toler) and (oS.alphas[i] > 0)):
            j, Ej = Svm.select_J(i, oS, Ei)
            alphaIold = oS.alphas[i].copy()
            alphaJold = oS.alphas[j].copy()
            if oS.label_matrix[i] != oS.label_matrix[j]:
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L == H:
                print('L == H')
                return 0
            eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
            if eta > 0:
                print(eta > 0)
                return 0
            oS.alphas[j] -= oS.label_matrix[j] * (Ei - Ej) / eta
            oS.alphas[j] = Svm.clip_alpha(oS.alphas[j], H, L)
            Svm.update_Ek(oS, j)
            if abs(oS.alphas[j] - alphaJold) < 0.00001:
                print('j not moving enough')
                return 0
            oS.alphas[i] += oS.label_matrix[j] * oS.label_matrix[i] * (alphaJold - oS.alphas[j])
            Svm.update_Ek(oS, i)
            b1 = oS.b - Ei - oS.label_matrix[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] \
                 - oS.label_matrix[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
            b2 = oS.b - Ej - oS.label_matrix[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] \
                 - oS.label_matrix[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
            if 0 < oS.alphas[i] < oS.C:
                oS.b = b1
            elif 0 < oS.alphas[j] < oS.C:
                oS.b = b2
            else:
                oS.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    @staticmethod
    def select_j_rand(i, m):
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j

    @staticmethod
    def clip_alpha(aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj
