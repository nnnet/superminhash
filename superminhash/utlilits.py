# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:09:52 2018

@author: nnnet
"""

import numpy as np
import sys
import collections
from itertools import groupby
import re
import copy

if sys.version_info[0] >= 3:
    basestring = str
    unicode = str
    long = int
    dict_items = type({}.items())
else:
    range = xrange

MAX_UINT32 = np.iinfo(np.uint32).max

def _slide(content, width=4):
    return [content[i:i + width] for i in range(max(len(content) - width + 1, 1))]


def _tokenize(content, reg=r'[\w\u4e00-\u9fcc]+', slide_width=4, slide_words_delimiter=' '):
    ''''`reg` : reg expression
               is meaningful only when `value` is basestring and describes
               what is considered to be a letter inside parsed string. Regexp
               object can also be specified (some attempt to handle any letters
               is to specify reg=re.compile(r'\w', re.UNICODE))
    '''

    if sys.version_info[0] >= 3:
        ret = content.lower()
    else:
        ret = content.decode('utf-8').lower()

    if not reg is None:
        ret = re.findall(reg, ret, re.U)
    else:
        ret = ret.replace('    ', ' ').replace('   ', ' ').replace('  ', ' ').split()

    if isinstance(slide_width, int):
        ret = _slide(('{0}'.format(slide_words_delimiter)).join(ret), width=slide_width)

    return ret


def build_by_text(content, reg=r'[\w\u4e00-\u9fcc]+', tokenize_slide_width=4, slide_words_delimiter=''):

    features = _tokenize(content, reg=reg, slide_width=tokenize_slide_width, slide_words_delimiter=slide_words_delimiter)
    return {k:sum(1 for _ in g) for k, g in groupby(sorted(features))}


def get_value(value_in, hash_type, build_by_features,
              tokenize_args,
              kwargs=None):

    if isinstance(value_in, type(hash_type)):
        if type(hash_type).__name__ == 'Simhash':
            value_out = (value_in.value, copy.deepcopy(value_in.v), copy.deepcopy(value_in.masks))
        elif type(hash_type).__name__ == 'Superminhash':
            value_out = (copy.deepcopy(value_in.values), copy.deepcopy(value_in.q), copy.deepcopy(value_in.p)\
                             , copy.deepcopy(value_in.b), value_in.i, value_in.a)
        else:
            raise Exception('Bad parameter with type.__name__ {0}'.format(type(value_in).__name__))
    elif isinstance(value_in, basestring):
        value_out = \
              build_by_features(
                build_by_text(unicode(value_in), **tokenize_args)
            , **kwargs)
    elif isinstance(value_in, collections.Iterable):
        value_out = build_by_features(value_in, **kwargs)
    elif isinstance(value_in, long):
        if type(hash_type).__name__ == 'Simhash':
            value_out = (value_in, None, None)
        else:
            raise Exception('Bad parameter with type.__name__ {0}'.format(type(value_in).__name__))
    else:
        raise Exception('Bad parameter with type {0}'.format(type(value_in)))

    return value_out


def simhash_build_by_features(features, length, hash_function, push_function):
    """
    `features`
               might be a list of unweighted tokens (a weight of 1
               will be assumed), a list of (token, weight) tuples or
               a token -> weight dict.

    `length` : int
               is the dimensions of fingerprints
    """
    v = [0] * length
    masks = [1 << i for i in range(length)]
    if isinstance(features, dict):
        features = features.items()
    elif isinstance(features, zip):
        features = list(features)

    nb_calc = len(features) - 1
    for i, feature in enumerate(features):
        value, v = push_function(feature, hash_function, v, masks, length, calc = nb_calc==i)

    return value, v, masks


def superminhash_build_by_features(features, length, hash_function, push_function):

    values = [MAX_UINT32] * length  # float64
    q = [-1] * length  # int64
    p = list(range(length))  # uint16
    b = [0] * (length - 1) + [np.int64(length)]  # int64
    i = 0  # int64
    a = length - 1  # uint16

    if isinstance(features, dict):
        features = features.items()

    if sys.version_info[0] >= 3 and isinstance(features, dict_items):
        features = (x[0] for x in features.__iter__())
    elif isinstance(features, zip):
        features = (x[0] for x in features)
    elif isinstance(features[0], tuple):
        features = (x[0] for x in features)

    for feature in features:
        values, q, p, b, i, a = push_function(feature, values, q, p, b, i, a, hash_function)

    return values, q, p, b, i, a
