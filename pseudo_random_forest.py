import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class PseudoRandomForest:
  
  def __init__(self, n_estimators):
    self.B = n_estimators
