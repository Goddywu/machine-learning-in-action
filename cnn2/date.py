#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/23
# Desc: 

from cnn2.knn import *


def create_date_set():
    group = np.array(
        [
            [1.0, 1.1],
            [1.0, 1.0],
            [0, 0],
            [0, 0.1]
        ]
    )
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def draw():
    group, labels = create_date_set()
    # todo: 画一下


""" ---   scenes   --- """


def ttest1():
    group, labels = create_date_set()
    result = classify0([0, 0], group, labels, 3)
    print(result)


def draw_date_person(data_set, class_label):
    import matplotlib.pyplot as plt
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    # ax.scatter(data_set[:, 1], data_set[:, 2])
    # 后两个是大小、颜色
    # ax.scatter(data_set[:, 1], data_set[:, 2], 15.0 * np.array(class_label), 15.0 * np.array(class_label))
    ax.scatter(data_set[:, 0], data_set[:, 1], 15.0 * np.array(class_label), 15.0 * np.array(class_label))
    plt.show()


def find_date_person():
    # 测试数据占总数据的比例
    ho_ratio = 0.1
    # knn中取前多少的数据
    k = 3

    data_set, class_label = file2matrix('./dating.txt')
    # draw_date_person(data_set, class_label)
    norm_data_set, ranges, min_values = auto_norm(data_set)
    data_set_num = norm_data_set.shape[0]
    test_set_num = int(data_set_num * ho_ratio)

    error_count = 0
    for i in range(test_set_num):
        classifier_result = classify0(in_x=norm_data_set[i, :],
                                      data_set=norm_data_set[test_set_num: data_set_num, :],
                                      labels=class_label[test_set_num: data_set_num],
                                      k=k)
        print("predict is {} while true is {}".format(classifier_result, class_label[i]))
        if classifier_result != class_label[i]:
            error_count += 1
    print("total error rate is {}".format(error_count / float(test_set_num)))
    pass


if __name__ == '__main__':
    # ttest1()
    find_date_person()
