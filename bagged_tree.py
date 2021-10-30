import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class BaggedTree:
  def __init__(self, B):
    self.B = B
    
  def fit(self, X, Y):
    N = len(X)
    self.models = []
    for b in range(self.B):
      idx = np.random.choice(N, size=N, replace=True)
      Xb = X[idx]
      Yb = Y[idx]
      # Let DecisionTree grow arbitrary deep - high variance
      model = DecisionTreeClassifier()
      model.fit(Xb, Yb)
      self.models.append(model)
