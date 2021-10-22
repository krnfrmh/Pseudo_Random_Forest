import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class PseudoRandomForest:
  
  def __init__(self, n_estimators):
    self.B = n_estimators

  def fit(self, X, Y, M=None):
    N, D = X.shape
    if M is None:
      # Number of hidden units as per RandomForest creator
      M = int(np.sqrt(D))

    self.models = []
    self.features = []
    for b in range(self.B):
      tree = DecisionTreeClassifier()

      # Features sampling
      features = np.random.choice(D, size=M, replace=False)

      # Row sampling
      idx = np.random.choice(N, size=N, replace=True)
      Xb = X[idx]
      Yb = Y[idx]
