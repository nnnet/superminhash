# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:09:52 2018

@author: nnnet
"""

import sys
import collections
from itertools import groupby
import re

if sys.version_info[0] >= 3:
    basestring = str
    unicode = str
    long = int
else:
    range = xrange


def _slide(content, width=4):
    return [content[i:i + width] for i in range(max(len(content) - width + 1, 1))]


def _tokenize(content, reg=r'[\w\u4e00-\u9fcc]+', slide_width=4, words_delimiter=''):
    ''''`reg` : reg expression
               is meaningful only when `value` is basestring and describes
               what is considered to be a letter inside parsed string. Regexp
               object can also be specified (some attempt to handle any letters
               is to specify reg=re.compile(r'\w', re.UNICODE))
    '''

    ret = ('{0}'.format(words_delimiter)).join(re.findall(reg, content.lower()))
    if isinstance(slide_width, int):
        return _slide(ret, width=slide_width)
    return ret


def build_by_text(content, build_by_features, kwargs, reg=r'[\w\u4e00-\u9fcc]+', tokenize_slide_width=4, words_delimiter=''):
    features = _tokenize(content, reg=reg, tokenize_slide_width=tokenize_slide_width, words_delimiter=words_delimiter)
    features = {k:sum(1 for _ in g) for k, g in groupby(sorted(features))}
    return build_by_features(features, **kwargs)


def get_value(value_in, hash_type, build_by_features, reg=r'[\w\u4e00-\u9fcc]+', tokenize_slide_width=4, words_delimiter='', kwargs=None):

    if isinstance(value_in, hash_type):
        value_out = value_in.value
    elif isinstance(value_in, basestring):
        value_out = \
              build_by_features(
                build_by_text(unicode(value_in), reg=reg, tokenize_slide_width=tokenize_slide_width, words_delimiter=words_delimiter)
            , **kwargs)
    elif isinstance(value_in, collections.Iterable):
        value_out = build_by_features(value_in, **kwargs)
    elif isinstance(value_in, long):
        value_out = value_in
    else:
        raise Exception('Bad parameter with type {0}'.format(type(value_in)))

    return value_out

def simhash_push(feature, hash_function, v, masks, length, calc=False):

    if isinstance(feature, basestring):
        h = hash_function(feature.encode('utf-8'))
        w = 1
    else:
        assert isinstance(feature, collections.Iterable)
        h = hash_function(feature[0].encode('utf-8'))
        w = feature[1]

    for i in range(length):
        v[i] += w if h & masks[i] else -w

    if calc:
        value = 0
        for i in range(length):
            if v[i] > 0:
                value |= masks[i]
        return value

    return None


def simhash_build_by_features(features, length, hash_function, push_function):
    """
    `length` : int
               is the dimensions of fingerprints

    `features`
               might be a list of unweighted tokens (a weight of 1
               will be assumed), a list of (token, weight) tuples or
               a token -> weight dict.
    """
    v = [0] * length
    masks = [1 << i for i in range(length)]
    if isinstance(features, dict):
        features = features.items()
    # for f in features:
    #     if isinstance(f, basestring):
    #         h = hash_function(f.encode('utf-8'))
    #         w = 1
    #     else:
    #         assert isinstance(f, collections.Iterable)
    #         h = hash_function(f[0].encode('utf-8'))
    #         w = f[1]
    #     for i in range(length):
    #         v[i] += w if h & masks[i] else -w
    #
    # value = 0
    # for i in range(length):
    #     if v[i] > 0:
    #         value |= masks[i]

    nb_calc = len(features) - 1
    for i, feature in enumerate(features):
        value = push_function(feature, hash_function, v, masks, length, calc=nb_calc == i)

    return value