#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Goddy <wuchuansheng@yeah.net> 2019/4/26
# Desc: 


def text_parse(long_string):
    """
    切分string为短句
    :param string:
    :return:
    """
    import re
    tokens = re.split(r'\W* ', long_string)
    return [token.lower() for token in tokens if len(token) > 2]
