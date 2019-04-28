#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/26
# Desc:

import numpy as np

"""
P(class_i | words_vector) = P(words_vector | class_i) · P(class_i) / P(words_vector)

P(words_vector | class_i) = P(word_1 | class_i) + P(word_2 | class_i) + ...
"""


def create_vocab_list(data_set) -> list:
    """取data_set里所有的词汇"""
    vocab_set = set([])
    for document in data_set:
        # 取并集
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def set0words_vector(vocab_list: list, input_set):
    """
    :param vocab_list:
    :param input_set: 输入向量
    :return: 标注vocab_list出现的词汇，有为1
    """
    # 创建一个所有元素为0的向量
    result_vector = [0] * len(vocab_list)
    for word in input_set:
        if word in input_set:
            result_vector[vocab_list.index(word)] = 1
        else:
            print('the word: {} is not in my vocabulary! '.format(word))
    return result_vector


def train_naive_bayes(train_matrix, train_category):
    """
    朴素贝叶斯分类器训练函数 (0-1分类)
    :param train_matrix:
        [
            ['my', 'dog'],
            ['maybe', 'not'],
            ['stop', 'stupid']
        ]
    :param train_category:
        [0, 0, 1]
    :return:
    """
    num0train_docs = len(train_matrix)
    num0words = len(train_matrix[0])
    # P(脏话)
    prob_abusive = sum(train_category) / float(num0train_docs)
    prob_0 = np.zeros(num0words)
    prob_1 = np.zeros(num0words)
    # 分母
    prob_0_denom = 0.0
    prob_1_denom = 0.0
    for i in range(num0train_docs):
        if train_category[i] == 1:
            prob_1 += train_matrix[i]
            prob_1_denom += sum(train_matrix[i])
        else:
            prob_0 += train_matrix[i]
            prob_0_denom += sum(train_matrix[i])
    prob_1_vector = prob_1 / prob_1_denom
    prob_0_vector = prob_0 / prob_0_denom
    return prob_0_vector, prob_1_vector, prob_abusive

