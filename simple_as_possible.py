# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:24:54 2019

@author: JJ5JXT
"""

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB


classifier = BinaryRelevance(GaussianNB())


classifier.fit(x_train, y_train)
