import random
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.datasets import  make_blobs

if __name__ == '__main__':
    iris = load_iris()
    print(iris.__class__.__name__)
