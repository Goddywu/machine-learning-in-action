#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/26
# Desc:

from naiveBayes4.bayes import *


def load_data_set():
    posting_list = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    # 1 代表有侮辱性词语
    class_vector = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vector


def ttest1():
    posting_list, class_vector = load_data_set()
    vocab_list = create_vocab_list(posting_list)
    print(vocab_list)
    print(set0words_vector(vocab_list, posting_list[1]))

    print('------')

    train_matrix = []
    for single_posting in posting_list:
        train_matrix.append(set0words_vector(vocab_list, single_posting))

    prob_0_vector, prob_1_vector, prob_abusive = train_naive_bayes(train_matrix, class_vector)
    print(prob_0_vector)
    print(prob_1_vector)
    print(prob_abusive)

    print('-----')
    print([0] * 3)


def ttest2():
    from naiveBayes4 import NaiveBayes
    posting_list, class_vector = load_data_set()
    bayes = NaiveBayes(posting_list, class_vector)
    print(bayes.predict(['love', 'my', 'dalmation']))
    print(bayes.predict(['stupid', 'garbage']))


def spam_test():
    from naiveBayes4 import NaiveBayes
    from naiveBayes4 import text_util
    import random

    doc_list = []
    class_list = []
    full_text = []
    for i in range(1, 26):
        with open('./email/spam/%d.txt' % i, encoding='ISO-8859-1') as f:
            long_string = f.read()
            word_list = text_util.text_parse(long_string)
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(1)
        with open('./email/ham/%d.txt' % i, encoding='ISO-8859-1') as f:
            word_list = text_util.text_parse(f.read())
            doc_list.append(word_list)
            full_text.extend(word_list)
            class_list.append(0)

    training_set = list(range(50))
    test_set = []
    for i in range(10):
        # 随机生成0到x范围内的实数
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])

    train_words_matrix = []
    train_words_labels = []
    for doc_index in training_set:
        train_words_matrix.append(doc_list[doc_index])
        train_words_labels.append(class_list[doc_index])
    naive_bayes = NaiveBayes(words_matrix=train_words_matrix, words_labels=train_words_labels)

    error_count = 0
    for doc_index in test_set:
        if class_list[doc_index] != naive_bayes.predict(doc_list[doc_index]):
            error_count += 1
    total_error_rate = float(error_count) / len(test_set)
    print('total error rate is {}'.format(str(total_error_rate)))


if __name__ == '__main__':
    # ttest1()
    # ttest2()
    spam_test()
