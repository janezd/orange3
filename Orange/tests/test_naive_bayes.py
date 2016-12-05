# Test methods with long descriptive names can omit docstrings
# pylint: disable=missing-docstring

import unittest

from Orange.classification import NaiveBayesLearner
from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.evaluation import CrossValidation, CA


class TestNaiveBayesLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = Table('titanic')
        cls.learner = NaiveBayesLearner()
        cls.model = cls.learner(data)
        cls.table = data[::20]

    def test_NaiveBayes(self):
        results = CrossValidation(self.table, [self.learner], k=10)
        ca = CA(results)
        self.assertGreater(ca, 0.7)
        self.assertLess(ca, 0.9)

    def test_predict_single_instance(self):
        for ins in self.table:
            self.model(ins)
            val, prob = self.model(ins, self.model.ValueProbs)

    def test_predict_table(self):
        self.model(self.table)
        vals, probs = self.model(self.table, self.model.ValueProbs)

    def test_predict_numpy(self):
        X = self.table.X[::20]
        self.model(X)
        vals, probs = self.model(X, self.model.ValueProbs)

    def test_degenerate(self):
        d = Domain((ContinuousVariable(name="A"), ContinuousVariable(name="B"), ContinuousVariable(name="C")),
                    DiscreteVariable(name="CLASS", values=["M", "F"]))
        t = Table(d, [[0,1,0,0], [0,1,0,1], [0,1,0,1]])
        nb = NaiveBayesLearner()
        model = nb(t)
        self.assertEqual(model.domain.attributes, ())
        self.assertEqual(model(t[0]), 1)
        self.assertTrue(all(model(t) == 1))

    def test_new_bayes(self):
        import numpy as np
        from Orange.statistics import contingency

        X = np.array([[0, 0, 0],
                      [1, 1, 1],
                      [0, 1, 0],
                      [2, 0, 1],
                      [0, 0, 1],
                      [1, 0, 1]])
        Y = np.array([0, 0, 0, 1, 1, 1])
        attrs = [DiscreteVariable("barva", values=("crna", "bela", "rdeca")),
                 DiscreteVariable("velikost", values=("velika", "mala")),
                 DiscreteVariable("cas", values=("dan", "noc"))]
        domain = Domain(attrs, DiscreteVariable("ujel", values=("da", "ne")))
        table = Table(domain, X, Y)
        table = Table("lenses")
        instance = Table(domain, np.array([[1, np.nan, 1]]), np.array([0]))
        # instance = Table(domain, np.array([[0, 0, 0]]), np.array([0]))
        i = 1
        instance = table[i: i+1]
        cont = contingency.get_contingencies(table)
        class_freq = np.array(np.diag(
            contingency.get_contingency(table, table.domain.class_var)))


        p = (class_freq + 1) / (np.sum(class_freq) + len(class_freq))
        #p = (class_freq) / (np.sum(class_freq))
        res = p.copy()
        for c, x in zip(cont, instance.X[0]):
            #print(c)
            if not np.isnan(x):
                #print(m, c[:, int(x)], m * p, c[:, int(x)] + m * p)
                # m-estimate
                #m = 2
                #p_cv = (c[:, int(x)] + m * p) / (np.sum(c[:, int(x)]) + m)
                # laplace
                p_cv = (c[:, int(x)] + 1) / (np.sum(c[:, int(x)]) + c.shape[0])
                #p_cv = (c[:, int(x)]) / (np.sum(c[:, int(x)]))
                res *= (p_cv) / (p)
        print()
        print(res / np.sum(res), table.domain.class_var.values)

        class_prob = (class_freq + 1) / (np.sum(class_freq) + len(class_freq))
        cont_prob = [np.log((np.array(c) + 1) / (np.sum(np.array(c), axis=0)[None, :] + c.shape[0]) /
                     class_prob[:, None]) for c in cont]
        #res = np.log(p.copy())
        #for p_cv, x in zip(cont_prob, instance.X[0]):
        #    if not np.isnan(x):
        #        res += p_cv[:, int(x)]
        #res = np.exp(res)
        res1 = np.exp(np.sum([p_cv[:, int(x)] for p_cv, x
                             in zip(cont_prob, instance.X[0])
                             if not np.isnan(x)], axis=0) + np.log(class_prob))
        #print(res / np.sum(res))
        print(res1 / np.sum(res1))

        log_cont_prob = []
        # [ 0.01364492  0.91071478  0.07564031] ['hard', 'none', 'soft']
        # [ 0.04358755  0.82671726  0.12969519] [young, myope, no, reduced | none]

        # [ 0.07050399  0.74589186  0.18360415] ['hard', 'none', 'soft']
        # [ 0.04358755  0.82671726  0.12969519] [young, myope, no, reduced | none]

        # laplace (instance 1)
        # [0.20937841  0.13631407  0.65430752]['hard', 'none', 'soft']
        # [0.17428279  0.20342097  0.62229625][young, myope, no, normal | soft]



        m = NaiveBayesLearner()(table)
        res = m(instance, 1)
        print(res[0], instance[0])
