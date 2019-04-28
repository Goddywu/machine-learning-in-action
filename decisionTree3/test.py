#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/23
# Desc:

from decisionTree3.trees import *
from decisionTree3.tree_plotter import *


def create_data_set():
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    feature_labels = ['no surfacing', 'flippers']
    return data_set, feature_labels


def retrieve_tree(i):
    tree_list = [
        {
            'no surfacing': {
                0: 'no',
                1: {
                    'flippers': {
                        0: 'no',
                        1: 'yes'
                    }
                }
            }
        },
        {
            'no surfacing': {
                0: 'no',
                1: {
                    'flippers': {
                        0: {
                            'head': {
                                0: 'no',
                                1: 'yes'
                            }
                        },
                        1: 'no'
                    }
                }
            }
        }
    ]
    return tree_list[i]


def ttest1():
    data_set, feature_labels = create_data_set()
    # print(calc_shannon_entropy(data_set))
    #
    # data_set[0][-1] = 'maybe'
    # print(calc_shannon_entropy(data_set))
    #
    # print(split_data_set(data_set, 0, 1))
    # print(split_data_set(data_set, 0, 0))

    print(choose_best_feature_index2split(data_set))

    data_set, feature_labels = create_data_set()
    tree = create_tree(data_set, feature_labels)
    print(tree)


    print(get_num0leafs(tree))


def ttest2():
    # tree = retrieve_tree(0)
    tree = retrieve_tree(1)
    print(get_num0leafs(tree))
    print(get_depth0tree(tree))
    create_plot(tree)


def ttest3():
    tree = retrieve_tree(0)
    data_set, feature_labels = create_data_set()
    print(classify(tree, feature_labels, [1, 0]))
    print(classify(tree, feature_labels, [1, 1]))


def predict_lenses():
    with open('./lenses.txt', 'r') as f:
        lenses = [line.split('\t') for line in f.readlines()]
        lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
        # tree = create_tree(lenses, lenses_labels)

        from decisionTree3 import DecisionTree
        tree = DecisionTree(data_set=lenses, feature_labels=lenses_labels)

        print(tree.get_tree())
        create_plot(tree.get_tree())


def core_test():
    from decisionTree3 import DecisionTree

    data_set, feature_labels = create_data_set()
    # print(DecisionTree.calc_shannon_entropy(data_set))
    # print('---')
    # print(DecisionTree.choose_best_feature_index2split(data_set))
    #
    # print('-----')
    # print(DecisionTree.create_tree(data_set, feature_labels))

    tree = DecisionTree(tree=retrieve_tree(0), feature_labels=feature_labels)

    print(tree.predict([1, 0]))
    print(tree.predict([1, 1]))


if __name__ == '__main__':
    # ttest1()
    # ttest2()
    # ttest3()
    predict_lenses()
    # core_test()
    # print(choose_best_feature_index2split([[1, 'no'],[1, 'no']]))
