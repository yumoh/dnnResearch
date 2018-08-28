import sklearn
import sklearn.datasets
import numpy as np

def binary_classification(size=1024):
    x, y = sklearn.datasets.make_classification(n_samples=size, n_features=10, n_classes=2, n_clusters_per_class=2)
    s=np.random.randn(x.shape[-1])
    x*=s
    return x, y
