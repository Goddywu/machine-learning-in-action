#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/23
# Desc: ID3算法

from math import log
import operator


def calc_shannon_entropy(data_set):
    """
    计算熵(香农熵)
    """
    label_counts = {}
    for vector in data_set:
        current_label = vector[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_entropy = 0.0
    for key in label_counts:
        # 该分类的概率
        prob = float(label_counts[key]) / len(data_set)
        shannon_entropy -= prob * log(prob, 2)
    return shannon_entropy


def split_data_set(data_set, axis, value):
    """
    按照给定特征划分数据集
    :param data_set:
    :param axis: 划分数据集的特征的index
    :param value: 划分数据集的特征的value
    :return:
    """
    result = []
    for single in data_set:
        if single[axis] == value:
            reduced_vector = single[:axis]
            reduced_vector.extend(single[axis + 1:])
            result.append(reduced_vector)
    return result


def choose_best_feature_index2split(data_set):
    """
    选择最好的划分数据集的特征de index
    :return:
    """
    num0feature = len(data_set[0]) - 1
    base_entropy = calc_shannon_entropy(data_set)
    best_info_gain = 0
    best_feature = -1
    for i in range(num0feature):
        feature_list = [single[i] for single in data_set]
        unique_values = set(feature_list)
        new_entropy = 0
        for value in unique_values:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_entropy(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majority_class(class_list):
    """
    返回数量最多的类
    :param class_list:
    :return:
    """
    class_count = {}
    for _class in class_list:
        if _class not in class_count.keys():
            class_count[_class] = 0
        class_count[_class] += 1
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0]


def create_tree(data_set, feature_labels):
    """
    创建决策树
    """
    class_list = [single[-1] for single in data_set]
    # 如果都属于同一类别，则停止划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 如果没有特征了，选择个数最多的class作为分类
    if len(data_set[0]) == 1:
        return majority_class(class_list)

    best_feature_index = choose_best_feature_index2split(data_set)
    best_feature_label = feature_labels[best_feature_index]
    result_tree = {best_feature_label: {}}
    del(feature_labels[best_feature_index])
    best_feature_values = [single[best_feature_index] for single in data_set]
    unique_best_feature_values = set(best_feature_values)
    for value in unique_best_feature_values:
        sub_labels = feature_labels[:]
        result_tree[best_feature_label][value] = create_tree(
            data_set=split_data_set(data_set, best_feature_index, value),
            feature_labels=sub_labels)
    return result_tree


def classify(input_tree, feature_labels, test_vector):
    """predict"""
    first_key = list(input_tree.keys())[0]
    second_dict = input_tree[first_key]
    feature_index = feature_labels.index(first_key)
    for key in list(second_dict.keys()):
        if test_vector[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feature_labels, test_vector)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(input_tree, file_path):
    import pickle
    with open(file_path, 'w') as f:
        pickle.dump(input_tree, f)


def grab_tree(file_path):
    import pickle
    with open(file_path, 'r') as f:
        return pickle.load(f)
