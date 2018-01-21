__all__ = ['utlilits']

import hashlib
import numpy as np
import logging
from utlilits import get_value, simhash_build_by_features 

MAX_UINT32 = np.iinfo(np.uint32).max

def _hash_function(x):
    return int(hashlib.md5(x).hexdigest(), 16)


class Simhash(object):

    def __init__(self, value, length=64, reg=r'[\w\u4e00-\u9fcc]+', tokenize_slide_width=4, hash_function=None, log=None):
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
        self.reg = reg
        self.value = None

        if hash_function is None:
            self.hash_function = _hash_function
        else:
            self.hash_function = hash_function

        if log is None:
            self.log = logging.getLogger("simhash")
        elif isinstance(log, logging.Logger):
            self.log = log

        self.value = get_value(value, self, simhash_build_by_features,
                               reg=reg, tokenize_slide_width=tokenize_slide_width,
                               kwargs={'hash_function' : self.hash_function, 'length' :self.length, 'reg' : reg})


class SuperMinHash(object):

    def __init__(self, value, length=64, reg=r'[\w\u4e00-\u9fcc]+', hash_function=None, log=None):

        self.values = [MAX_UINT32] * length  # float64
        self.q = [-1] * length  # int64
        self.p = list(range(length))  # uint16
        self.b = [0] * (length - 1) + [np.int64(length)]  # int64
        self.i = 0  # int64
        self.a = length - 1  # uint16

        self.length = length
        self.reg = reg
        self.value = None

        if hash_function is None:
            self.hash_function = _hash_function
        else:
            self.hash_function = hash_function

        if log is None:
            self.log = logging.getLogger("superminhash")
        elif isinstance(log, logging.Logger):
            self.log = log


        self.value = get_value(value, self, simhash_build_by_features, {'hash_function' : self.hash_function, 'length' :self.length})



    # // Push ...
    def push(self, b, hash_function=None):

        #    	// initialize pseudo-random generator with seed d
        #    	d = metro.Hash64(b, 42)
        #    	rnd := pcgr.New(int64(d), 0)
        np.random.seed(seed=(hash(b) if hash_function is None else hash_function(b)) % MAX_UINT32)

        for j in range(self.a):
            r = np.float64(np.random.randint(MAX_UINT32)) / MAX_UINT32
            offset = np.random.randint(MAX_UINT32) % np.uint32(np.uint16(len(self.values)) - j)
            k = np.uint32(j) + offset

            if self.q[j] != self.i:
                self.q[j] = self.i
                self.p[j] = np.uint16(j)

            if self.q[k] != self.i:
                self.q[k] = self.i
                self.p[k] = np.uint16(k)

            self.p[j], self.p[k] = self.p[k], self.p[j]
            rj = r + np.float64(j)
            if rj < self.values[self.p[j]]:

                jc = np.uint16(min(self.values[self.p[j]], np.float64(len(self.values) - 1)))
                self.values[self.p[j]] = rj
                if j < jc:
                    self.b[jc] -= 1
                    self.b[j] += 1
                    while self.b[self.a] == 0:
                        self.a -= 1

        self.i += 1

    # // Similarity ...
    def similarity(self, other):

        if self.length() != other.length():
            raise ValueError("signatures not of same length, sign has length %d, while other has length %d" \
                             , len(self.values), len(other.values))

        sim = 0.0
        for i, element in enumerate(self.values):
            if element == other.values[i]:
                sim += 1

        return sim / np.float64(len(self.values))


if __name__ == '__main__':
    mhash1 = lambda b: int(hashlib.sha1(b).hexdigest(), 32)
    mhash2 = lambda b: int(hashlib.sha256(b).hexdigest(), 32)


    def TestComplete():
        lenght = 10000
        t1 = ["hello", "world", "foo", "baz", "bar", "zomg."]
        t2 = ["hello", "world", "foo", "baz", "bar", "zomg", 'baz', 'baz', 'baz', 'baz']

        hash_function = None

        def _create_hash(s, lenght=10, hash_function=None, minhash=None):
            sh = SuperMinHash(lenght)
            for i in s:
                sh.push(i, hash_function)
            return sh, None if minhash is None else minhash(s)

        s1, _ = _create_hash(t1, lenght=lenght, hash_function=hash_function, minhash=None)
        s2, _ = _create_hash(t2, lenght=lenght, hash_function=hash_function, minhash=None)

        sim1 = s1.similarity(s2)
        #		t.Log(sim1)
        #		sim2 := m1.Similarity(m2)
        #		t.Log(sim2)
        #
        #		fmt.Println(sim1, sim2)
        print(sim1)


    #        print(s1.values)
    #        print(s2.values)

    TestComplete()
