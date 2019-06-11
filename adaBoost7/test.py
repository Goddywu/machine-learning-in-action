#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/6/11
# Desc: 

import numpy as np


def load_simple_data():
    data_matrix = np.matrix([
        [1, 2.1],
        [2, 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_matrix, class_labels


if __name__ == '__main__':
    pass
