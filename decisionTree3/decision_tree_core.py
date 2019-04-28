#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/24
# Desc:

# todo:


class DecisionTree:
    def __init__(self, feature_labels: list, **kwargs):
        """
        决策树 - ID3
        :param data_set: [特征1, ... , 分类]  Nullable
        :param feature_labels: [特征1名称, ...]
               class: data_set最后一列，标注的类
        :param tree: 同级别的英文名称只有存在一个 Nullable
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
            }
        """
        if 'data_set' in kwargs.keys():
            self._data_set = kwargs['data_set']
        self._feature_labels = feature_labels
        if 'tree' in kwargs.keys() and kwargs['tree'] is not None:
            self.tree = kwargs['tree']
        else:
            self.tree = DecisionTree.create_tree(self._data_set, self._feature_labels)

    def predict(self, vector: list) -> str:
        return DecisionTree.classify(self.tree, self._feature_labels, vector)

    def get_tree(self):
        return self.tree

    """ --- utils --- """

    @staticmethod
    def calc_shannon_entropy(data_set: list) -> float:
        """
        计算香农熵
        """
        from math import log

        label_count_dict = {}
        for vector in data_set:
            current_label = vector[-1]
            if current_label not in label_count_dict.keys():
                label_count_dict[current_label] = 0
            label_count_dict[current_label] += 1
        total_entropy = 0.0
        for label in label_count_dict:
            prob = float(label_count_dict[label]) / len(data_set)
            total_entropy -= prob * log(prob, 2)
        return total_entropy

    @staticmethod
    def split_data_set(data_set: list, feature_index: int, value: str) -> list:
        """
        切分数据集
        """
        result_data_set = []
        for vector in data_set:
            if vector[feature_index] == value:
                new_vector = vector[:feature_index]
                new_vector.extend(vector[feature_index + 1:])
                result_data_set.append(new_vector)
        return result_data_set

    @staticmethod
    def choose_best_feature_index2split(data_set: list) -> int:
        """
        选择最好的划分数据集的特征的index
        """
        num0features = len(data_set[0]) - 1
        best_index = -1
        # 信息增益 = 旧熵 - 新熵 : IG(Y|X) = H(Y) - H(Y|X)
        best_info_gain = 0
        for i in range(num0features):
            unique_current_feature_values = set([vector[i] for vector in data_set])
            new_entropy = 0
            for current_feature_value in unique_current_feature_values:
                sub_data_set = DecisionTree.split_data_set(data_set, i, current_feature_value)
                prob = float(len(sub_data_set)) / len(data_set)
                new_entropy += prob * DecisionTree.calc_shannon_entropy(sub_data_set)
            base_entropy = DecisionTree.calc_shannon_entropy(data_set)
            new_info_gain = base_entropy - new_entropy
            if new_info_gain > best_info_gain:
                best_info_gain = new_info_gain
                best_index = i
        return best_index

    @staticmethod
    def choose_major_class(class_list: list) -> str:
        """
        返回数量最多的类
        :param class_list: data_set最后一列，标注结果
        :return:
        """
        import operator

        class_count = {}
        for _class in class_list:
            if _class not in class_count.keys():
                class_count[_class] = 0
            class_count[_class] += 1
        sorted_class_count = sorted(class_count.items(),
                                    key=operator.itemgetter(1),
                                    reverse=True)
        return sorted_class_count[0][0]

    @staticmethod
    def create_tree(data_set: list, feature_labels: list) -> 'DecisionTree.tree':
        """
        创建决策树
        :param data_set:
        :param feature_labels: 特征的名称
        :return:
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
            }
        """
        class_list = [vector[-1] for vector in data_set]
        # 如果都属于同一类别，则停止划分
        if len(set(class_list)) == 1:
            return class_list[0]
        # 如果没有特征了，选择个数最多的class作为分类
        if len(data_set[0]) == 1:
            return DecisionTree.choose_major_class(class_list)

        best_feature_index = DecisionTree.choose_best_feature_index2split(data_set)
        best_feature_label = feature_labels[best_feature_index]
        tree = {
            best_feature_label: {}
        }
        del feature_labels[best_feature_index]
        unique_best_feature_values = set([vector[best_feature_index] for vector in data_set])
        for value in unique_best_feature_values:
            sub_data_set = DecisionTree.split_data_set(data_set, best_feature_index, value)
            sub_labels = feature_labels[:]
            tree[best_feature_label][value] = DecisionTree.create_tree(data_set=sub_data_set,
                                                                       feature_labels=sub_labels)
        return tree

    @staticmethod
    def classify(tree: 'DecisionTree.tree', feature_labels: list, test_vector: list) -> str:
        """
        预测分类
        :param tree: 同级别的英文名称只有存在一个
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
            }
        :param feature_labels:
        :param test_vector:
        :return:
        """
        first_key = list(tree.keys())[0]
        second_dict = tree[first_key]
        feature_index = feature_labels.index(first_key)
        for key in list(second_dict.keys()):
            if test_vector[feature_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    # 这里没必要用feature_labels和test_vector的子集
                    return DecisionTree.classify(second_dict[key], feature_labels, test_vector)
                else:
                    return second_dict[key]

    @staticmethod
    def store_tree(input_tree, file_path):
        import pickle
        with open(file_path, 'w') as f:
            pickle.dump(input_tree, f)

    @staticmethod
    def grab_tree(file_path):
        import pickle
        with open(file_path, 'r') as f:
            return pickle.load(f)
