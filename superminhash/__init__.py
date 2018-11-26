from __future__ import division, unicode_literals

__all__ = ['utlilits']

import hashlib
import numpy as np
import logging
import collections
import sys

if sys.version_info[0] >= 3:
    basestring = str
    unicode = str
    long = int
else:
    range = xrange

try:
    from utlilits import get_value, simhash_build_by_features, superminhash_build_by_features, MAX_UINT32
except:
    from superminhash.utlilits import get_value, simhash_build_by_features, superminhash_build_by_features, MAX_UINT32

def _hash_function(x):
    if isinstance(x, str):
        return int(hashlib.md5(x.encode()).hexdigest(), 16)
    else:
        return int(hashlib.md5(x).hexdigest(), 16)
        


class Simhash(object):

    def __init__(self, value, length=64,
                 reg=r'[\w\u4e00-\u9fcc]+', tokenize_slide_width=4, slide_words_delimiter='',
                 hash_function=None, log=None):
        """
        `length` is the dimensions of fingerprints

        `reg` is meaningful only when `value` is basestring and describes
        what is considered to be a letter inside parsed string. Regexp
        object can also be specified (some attempt to handle any letters
        is to specify reg=re.compile(r'\w', re.UNICODE))
        `hash_function` accepts a utf-8 encoded string and returns a unsigned
        integer in at least `f` bits.
        """

        self.length = length
        self.value = None

        if hash_function is None:
            self.hash_function = _hash_function
        else:
            self.hash_function = hash_function

        if log is None:
            self.log = logging.getLogger(type(self).__name__.lower())
        elif isinstance(log, logging.Logger):
            self.log = log

        self.value, self.v, self.masks = get_value(value, self, simhash_build_by_features,
                               tokenize_args={'reg':reg, 'tokenize_slide_width':tokenize_slide_width, 'slide_words_delimiter':slide_words_delimiter},
                               kwargs={'hash_function' : self.hash_function, 'push_function' : self._push, 'length' : self.length})

    def _push(self, feature, hash_function, v, masks, length, calc=False):

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
            return value, v

        return None, v

    def push(self, feature, calc=True):
        self.value, self.v = self._push(feature, self.hash_function, self.v, self.masks, self.length, calc=calc)


    def distance(self, another):
        assert self.length == another.length
        x = (self.value ^ another.value) & ((1 << self.length) - 1)
        ans = 0
        while x:
            ans += 1
            x &= x - 1
        return ans


class SimhashIndex(object):

    def __init__(self, objs, length=64, k=2, log=None):
        """
        `objs` is a list of (obj_id, simhash)
            obj_id is a string, simhash is an instance of Simhash
        `length` is the same with the one for Simhash
        `k` is the tolerance
        """
        self.k = k
        self.length = length
        count = len(objs)

        if log is None:
            self.log = logging.getLogger("simhash")
        else:
            self.log = log

        self.log.info('Initializing %s data.', count)

        self.bucket = collections.defaultdict(set)

        for i, q in enumerate(objs):
            if i % 10000 == 0 or i == count - 1:
                self.log.info('%s/%s', i + 1, count)

            self.add(*q)

    def get_near_dups(self, simhash):
        """
        `simhash` is an instance of Simhash
        return a list of obj_id, which is in type of str
        """
        assert simhash.length == self.length

        ans = set()

        for key in self.get_keys(simhash):
            dups = self.bucket[key]
            self.log.debug('key:%s', key)
            if len(dups) > 200:
                self.log.warning('Big bucket found. key:%s, len:%s', key, len(dups))

            for dup in dups:
                sim2, obj_id = dup.split(',', 1)
                sim2 = Simhash(long(sim2, 16), self.length)

                d = simhash.distance(sim2)
                if d <= self.k:
                    ans.add(obj_id)
        return list(ans)

    def add(self, obj_id, simhash):
        """
        `obj_id` is a string
        `simhash` is an instance of Simhash
        """
        assert simhash.length == self.length

        for key in self.get_keys(simhash):
            v = '%x,%s' % (simhash.value, obj_id)
            self.bucket[key].add(v)

    def delete(self, obj_id, simhash):
        """
        `obj_id` is a string
        `simhash` is an instance of Simhash
        """
        assert simhash.length == self.length

        for key in self.get_keys(simhash):
            v = '%x,%s' % (simhash.value, obj_id)
            if v in self.bucket[key]:
                self.bucket[key].remove(v)

    @property
    def offsets(self):
        """
        You may optimize this method according to <http://www.wwwconference.org/www2007/papers/paper215.pdf>
        """
        return [self.length // (self.k + 1) * i for i in range(self.k + 1)]

    def get_keys(self, simhash):
        for i, offset in enumerate(self.offsets):
            if i == (len(self.offsets) - 1):
                m = 2 ** (self.length - offset) - 1
            else:
                m = 2 ** (self.offsets[i + 1] - offset) - 1
            c = simhash.value >> offset & m
            yield '%x:%x' % (c, i)

    def bucket_size(self):
        return len(self.bucket)


class Superminhash(object):

    def __init__(self, value, length=64,
                 reg=r'[\w\u4e00-\u9fcc]+', tokenize_slide_width=4, slide_words_delimiter='',
                 hash_function=None, log=None):

        self.length = length
        self.values = [MAX_UINT32] * length  # float64
        self.q = [-1] * length  # int64
        self.p = list(range(length))  # uint16
        self.b = [0] * (length - 1) + [np.int64(length)]  # int64
        self.i = 0  # int64
        self.a = length - 1  # uint16

        if hash_function is None:
            self.hash_function = _hash_function
        else:
            self.hash_function = hash_function

        if log is None:
            self.log = logging.getLogger(type(self).__name__.lower())
        elif isinstance(log, logging.Logger):
            self.log = log

        self.values, self.q, self.p, self.b, self.i, self.a = get_value(value, self, superminhash_build_by_features,
                                                   tokenize_args={'reg': reg,
                                                                  'tokenize_slide_width': tokenize_slide_width,
                                                                  'slide_words_delimiter': slide_words_delimiter},
                                                   kwargs={'hash_function': self.hash_function,
                                                           'push_function': self._push, 'length': self.length})

    def _push(self, feature, values, q, p, b, i, a, hash_function=None):

        np.random.seed(seed=(hash(feature) if hash_function is None else hash_function(feature)) % MAX_UINT32)

        for j in range(a):
            r = np.float64(np.random.randint(MAX_UINT32)) / MAX_UINT32
            offset = np.random.randint(MAX_UINT32) % np.uint32(np.uint16(len(values)) - j)
            k = np.uint32(j) + offset

            if q[j] != i:
                q[j] = i
                p[j] = np.uint16(j)

            if q[k] != i:
                q[k] = i
                p[k] = np.uint16(k)

            p[j], p[k] = p[k], p[j]
            rj = r + np.float64(j)
            if rj < values[p[j]]:

                jc = np.uint16(min(values[p[j]], np.float64(len(values) - 1)))
                values[p[j]] = rj
                if j < jc:
                    b[jc] -= 1
                    b[j] += 1
                    while b[a] == 0:
                        a -= 1

        i += 1

        return values, q, p, b, i, a

    # // Push ...
    def push(self, feature):
        self.values, self.q, self.p, self.b, self.i, self.a = \
            self._push(feature, self.values, self.q, self.p, self.b, self.i, self.a, self.hash_function)


    # // Similarity ...
    def similarity(self, other):

        if self.length != other.length:
            raise ValueError("signatures not of same length, sign has length %d, while other has length %d" \
                             , len(self.values), len(other.values))

        sim = 0.0
        for i, element in enumerate(self.values):
            if element == other.values[i]:
                sim += 1

        return sim / np.float64(self.length)

    # // Distance ...
    def distance(self, other):

        return 1 - self.similarity(other)