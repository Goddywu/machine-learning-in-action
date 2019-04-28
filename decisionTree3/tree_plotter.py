#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/24
# Desc: 


import matplotlib.pyplot as plt

decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def create_plot(in_tree):
    """main function"""
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_num0leafs(in_tree))
    plot_tree.totalD = float(get_depth0tree(in_tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


def plot_node(node_text, current_pt, parent_pt, node_type):
    """pt means position"""
    create_plot.ax1.annotate(node_text,
                             xy=parent_pt,
                             xycoords='axes fraction',
                             xytext=current_pt,
                             textcoords='axes fraction',
                             va='center',
                             ha='center',
                             bbox=node_type,
                             arrowprops=arrow_args)


def plot_middle_text(current_pt, parent_pt, link_text):
    x_middle = (parent_pt[0] - current_pt[0]) / 2.0 + current_pt[0]
    y_middle = (parent_pt[1] - current_pt[1]) / 2.0 + current_pt[1]
    create_plot.ax1.text(x_middle, y_middle, link_text)


def plot_tree(tree, parent_pt, node_text):
    num0leafs = get_num0leafs(tree)
    depth = get_depth0tree(tree)
    first_key = list(tree.keys())[0]
    current_pt = (plot_tree.xOff + (1.0 + float(num0leafs)) / 2.0 / plot_tree.totalW,
                  plot_tree.yOff)
    plot_middle_text(current_pt, parent_pt, node_text)
    plot_node(first_key, current_pt, parent_pt, decision_node)
    second_dict = tree[first_key]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in list(second_dict.keys()):
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], current_pt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), current_pt, leaf_node)
            plot_middle_text((plot_tree.xOff, plot_tree.yOff), current_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def get_num0leafs(tree):
    num0leafs = 0
    first_key = list(tree.keys())[0]
    second_dict = tree[first_key]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num0leafs += get_num0leafs(second_dict[key])
        else:
            num0leafs += 1
    return num0leafs


def get_depth0tree(tree):
    max_depth = 0
    first_key = list(tree.keys())[0]
    second_dict = tree[first_key]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_depth0tree(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth
