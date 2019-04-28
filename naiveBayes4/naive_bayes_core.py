#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/26
# Desc: 

"""
P(class_i | words_vector) = P(words_vector | class_i) · P(class_i) / P(words_vector)
因为比较分类概率 P(class_i | words_vector) 的大小时， 与 P(words_vector) 无关，
    所以' / P(words_vector)'可以忽略掉
P(words_vector | class_i) = P(word_1 | class_i) + P(word_2 | class_i) + ...
因为P(word_1 | class_i)太小了，导致程序向下溢出或者得不到正确答案，
    所以利用 ln(a * b) = ln(a) * ln(b)  & ln(a + b) = ln(a) + ln(b)  公式
最终比较的是    [log(P(word_1 | class_i)) + log(P(word_2 | class_i)) + ...] + log(P(class_i)) 来比较即可
"""


class NaiveBayes:
    def __init__(self, words_matrix, words_labels):
        """

        :param words_matrix:
            [
                ['my', 'dog'],
                ['maybe', 'not'],
                ['stop', 'stupid']
            ]
        :param words_labels:
            [
                1,
                0,
                1
            ]
        """
        self.words_matrix = words_matrix
        self.vocab_list = NaiveBayes.get_vocab_list(words_matrix)
        train_matrix = NaiveBayes.create_train_matrix(words_matrix)
        self.prob0_vector, self.prob1_vector, self.prob_abusive \
            = NaiveBayes.train_naive_bayes(train_matrix, train_class=words_labels)
        pass

    def predict(self, test_words):
        return NaiveBayes.classify(
            self.vocab_list, test_words, self.prob0_vector, self.prob1_vector, self.prob_abusive)

    """ --- utils --- """

    @staticmethod
    def classify(vocab_list, test_words, prob0_vector, prob1_vector, prob_abusive):
        """

        :param vocab_list:
        :param test_words: ['my', 'dog']
        :param prob0_vector: [P(word_1 | class_0), ...]
        :param prob1_vector: [P(word_1 | class_1), ...]
        :param prob_abusive: P(class_1)
        :return:
        """
        import numpy as np
        # [1, 0, 0, 1, ...] 测试语句在词汇表中出现的情况
        test_vocab_marked_list = NaiveBayes.words2vocab_marked_list(vocab_list, test_words)
        # [P(word_1 | class_0), ...]
        prob_1 = sum(test_vocab_marked_list * prob1_vector) + np.log(prob_abusive)
        prob_0 = sum(test_vocab_marked_list * prob0_vector) + np.log(1 - prob_abusive)
        if prob_1 > prob_0:
            return 1
        else:
            return 0

    @staticmethod
    def train_naive_bayes(train_matrix: list, train_class: list):
        """
        朴素贝叶斯分类器训练函数 (0-1分类)
        :param train_matrix:
            [
                [1, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1]
            ]
        :param train_class: 标注分类结果
            [0, 0, 1]
        :return: prob0_vector -> [P(word_1 | class_0), ...]
                 prob1_vector -> [P(word_1 | class_1), ...]
                 prob_abusive -> P(class_1)
        """
        import numpy as np

        words_num0matrix = len(train_matrix)
        word_num0words = len(train_matrix[0])
        # class 1, P(class_1)
        prob_abusive = sum(train_class) / float(len(train_class))
        # 统计出现次数,,这里初始用1，防止相乘变0,
        #   words_count0class1为在class为1的情况下，各个词出现的次数
        words_count0class1 = np.ones(word_num0words)
        words_count0class0 = np.ones(word_num0words)
        # 统计所有词的次数
        word_num0class1 = 2.0
        word_num0class0 = 2.0

        for i in range(words_num0matrix):
            if train_class[i] == 1:
                # list 对应位置相加，
                words_count0class1 += train_matrix[i]
                word_num0class1 += sum(train_matrix[i])
            else:
                words_count0class0 += train_matrix[i]
                word_num0class0 += sum(train_matrix[i])
        # 为了防止向下溢出，添加log(a, b)
        #   [P(word | class_1), ...]
        prob1_vector = np.log(words_count0class1 / word_num0class1)
        prob0_vector = np.log(words_count0class0 / word_num0class0)
        return prob0_vector, prob1_vector, prob_abusive

    @staticmethod
    def create_train_matrix(words_matrix: list) -> list:
        """
        获取训练使用的矩阵，
        :param words_matrix:
            [
                ['my', 'dog'],
                ['maybe', 'not'],
                ['stop', 'stupid']
            ]
        :return:
            [
                [1, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1]
            ]
        """
        train_matrix = []
        vocab_list = NaiveBayes.get_vocab_list(words_matrix)
        for words in words_matrix:
            train_matrix.append(NaiveBayes.words2vocab_marked_list(vocab_list, words))
        return train_matrix

    @staticmethod
    def get_vocab_list(words_matrix: list) -> list:
        """
        获取词汇表
        :param words_matrix:
            [
                ['my', 'dog'],
                ['maybe', 'not'],
                ['stop', 'stupid']
            ]
        :return: ['my', 'dog'...]
        """
        vocab_set = set()
        for words in words_matrix:
            # 取并集
            vocab_set = vocab_set | set(words)
        return list(vocab_set)

    @staticmethod
    def words2vocab_marked_list(vocab_list: list, words: list) -> list:
        """
        标注句子在词典出现的词汇, 1 为出现， 0 未出现
        :param vocab_list:
        :param words:
        :return:
        """
        # [0, 0, 0...]
        result_list = [0] * len(vocab_list)
        for word in words:
            if word in vocab_list:
                # 词集模型 set-of-words-model，只关心出现不出现
                # result_list[vocab_list.index(word)] = 1
                # 词袋模型 bag-of-words-model，关心词出现频率
                result_list[vocab_list.index(word)] += 1
            else:
                print('the word: {} is not in my vocabulary! '.format(word))
        return result_list
