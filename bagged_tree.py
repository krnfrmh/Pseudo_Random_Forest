import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

class BaggedTree:
  def __init__(self, B):
    self.B = B
    
