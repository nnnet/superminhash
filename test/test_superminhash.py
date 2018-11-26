# -*- coding: utf-8 -*-
from unittest import main, TestCase

from superminhash import Simhash, SimhashIndex, Superminhash

from sklearn.feature_extraction.text import TfidfVectorizer

class TestSimhash(TestCase):

    def test_value(self):
        self.assertEqual(Simhash(['aaa', 'bbb']).value, 57087923692560392)

    def test_distance(self):
        sh = Simhash('How are you? I AM fine. Thanks. And you?')
        sh2 = Simhash('How old are you ? :-) i am fine. Thanks. And you?')
        self.assertTrue(sh.distance(sh2) > 0)

        sh3 = Simhash(sh2)
        self.assertEqual(sh2.distance(sh3), 0)

        self.assertNotEqual(Simhash('1').distance(Simhash('2')), 0)

    def test_chinese(self):
        self.maxDiff = None

        sh1 = Simhash(u'你好　世界！　　呼噜。')
        sh2 = Simhash(u'你好，世界　呼噜')

        sh4 = Simhash(u'How are you? I Am fine. ablar ablar xyz blar blar blar blar blar blar blar Thanks.')
        sh5 = Simhash(u'How are you i am fine.ablar ablar xyz blar blar blar blar blar blar blar than')
        sh6 = Simhash(u'How are you i am fine.ablar ablar xyz blar blar blar blar blar blar blar thank')

        self.assertEqual(sh1.distance(sh2), 0)

        self.assertTrue(sh4.distance(sh6) < 3)
        self.assertTrue(sh5.distance(sh6) < 3)

    def test_short(self):
        shs = [Simhash(s).value for s in ('aa', 'aaa', 'aaaa', 'aaaab', 'aaaaabb', 'aaaaabbb')]

        for i, sh1 in enumerate(shs):
            for j, sh2 in enumerate(shs):
                if i != j:
                    self.assertNotEqual(sh1, sh2)

    def test_sparse_features(self):
        data = [
            u'How are you? I Am fine. blar blar blar blar blar Thanks.',
            u'How are you i am fine. blar blar blar blar blar than',
            u'This is simhash test.',
            u'How are you i am fine. blar blar blar blar blar thank1'
        ]
        vec = TfidfVectorizer()
        D = vec.fit_transform(data)
        voc = dict((i, w) for w, i in vec.vocabulary_.items())

        # Verify that distance between data[0] and data[1] is < than
        # data[2] and data[3]
        shs = []
        for i in range(D.shape[0]):
            Di = D.getrow(i)
            # features as list of (token, weight) tuples)
            features = zip([voc[j] for j in Di.indices], Di.data)
            shs.append(Simhash(features))
        self.assertNotEqual(shs[0].distance(shs[1]), 0)
        self.assertNotEqual(shs[2].distance(shs[3]), 0)
        self.assertLess(shs[0].distance(shs[1]), shs[2].distance(shs[3]))

        # features as token -> weight dicts
        D0 = D.getrow(0)
        dict_features = dict(zip([voc[j] for j in D0.indices], D0.data))
        self.assertEqual(Simhash(dict_features).value, 17583409636488780916)

        # the sparse and non-sparse features should obviously yield
        # different results
        self.assertNotEqual(Simhash(dict_features).value,
                            Simhash(data[0]).value)


class TestSimhashIndex(TestCase):
    data = {
        1: u'How are you? I Am fine. blar blar blar blar blar Thanks.',
        2: u'How are you i am fine. blar blar blar blar blar than',
        3: u'This is simhash test.',
        4: u'How are you i am fine. blar blar blar blar blar thank1',
    }

    def setUp(self):
        objs = [(str(k), Simhash(v)) for k, v in self.data.items()]
        self.index = SimhashIndex(objs, k=10)

    def test_get_near_dup(self):
        s1 = Simhash(u'How are you i am fine.ablar ablar xyz blar blar blar blar blar blar blar thank')
        dups = self.index.get_near_dups(s1)
        self.assertEqual(len(dups), 3)

        self.index.delete('1', Simhash(self.data[1]))
        dups = self.index.get_near_dups(s1)
        self.assertEqual(len(dups), 2)

        self.index.delete('1', Simhash(self.data[1]))
        dups = self.index.get_near_dups(s1)
        self.assertEqual(len(dups), 2)

        self.index.add('1', Simhash(self.data[1]))
        dups = self.index.get_near_dups(s1)
        self.assertEqual(len(dups), 3)

        self.index.add('1', Simhash(self.data[1]))
        dups = self.index.get_near_dups(s1)
        self.assertEqual(len(dups), 3)


class TestSuperminhash(TestCase):

    def test_value(self):
        self.assertEqual(Superminhash(['aaa', 'bbb'], length=4).values, [0.65485953228894145, 2.0632418997267359, 1.7271384144497892, 1.6527771937318092])

    def test_distance(self):
        sh = Superminhash('How are you? I AM fine. Thanks. And you?')
        sh2 = Superminhash('How old are you ? :-) i am fine. Thanks. And you?')
        self.assertTrue(sh.similarity(sh2) < 1)

        sh3 = Superminhash(sh2)
        self.assertEqual(sh2.similarity(sh3), 1)

        self.assertEqual(Superminhash('1').similarity(Superminhash('2')), 0.)

    def test_chinese(self):
        self.maxDiff = None

        # sh1 = Superminhash(u'你好　世界！　　呼噜。')
        # sh2 = Superminhash(u'你好，世界　呼噜')

        sh4 = Superminhash(u'How are you? I Am fine. ablar ablar xyz blar blar blar blar blar blar blar Thanks.')
        sh5 = Superminhash(u'How are you i am fine.ablar ablar xyz blar blar blar blar blar blar blar than')
        sh6 = Superminhash(u'How are you i am fine.ablar ablar xyz blar blar blar blar blar blar blar thank')

        # self.assertEqual(sh1.similarity(sh2), 0)

        self.assertTrue(sh4.similarity(sh6) < 1)
        self.assertTrue(sh5.similarity(sh6) < 1)

    def test_short(self):
        shs = [Superminhash(s) for s in ('aa', 'aaa', 'aaaa', 'aaaab', 'aaaaabb', 'aaaaabbb')]

        for i, sh1 in enumerate(shs):
            for j, sh2 in enumerate(shs):
                if i != j:
                    self.assertNotEqual(sh1.values, sh2.values)
                    self.assertNotEqual(sh1.similarity(sh2), 1)

    def test_sparse_features(self):
        data = [
            u'How are you? I Am fine. blar blar blar blar blar Thanks.',
            u'How are you i am fine. blar blar blar blar blar than',
            u'This is Superminhash test.',
            u'How are you i am fine. blar blar blar blar blar thank1'
        ]
        vec = TfidfVectorizer()
        D = vec.fit_transform(data)
        voc = dict((i, w) for w, i in vec.vocabulary_.items())

        # Verify that distance between data[0] and data[1] is < than
        # data[2] and data[3]
        shs = []
        for i in range(D.shape[0]):
            Di = D.getrow(i)
            # features as list of (token, weight) tuples)
            features = zip([voc[j] for j in Di.indices], Di.data)
            shs.append(Superminhash(features))

        self.assertNotEqual(shs[0].similarity(shs[1]), 1)
        self.assertNotEqual(shs[2].similarity(shs[3]), 1)
        self.assertGreater(shs[0].similarity(shs[1]), shs[2].similarity(shs[3]))

        # features as token -> weight dicts
        D0 = D.getrow(0)
        dict_features = dict(zip([voc[j] for j in D0.indices], D0.data))
        # self.assertEqual(Superminhash(dict_features).values, 17583409636488780916)

        # the sparse and non-sparse features should obviously yield
        # different results
        self.assertNotEqual(Superminhash(dict_features).values,
                            Superminhash(data[0]).values)


if __name__ == '__main__':
    main()