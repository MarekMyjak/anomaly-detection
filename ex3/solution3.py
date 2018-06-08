from sklearn import svm
from utils import binary2neg_boolean
import numpy as np

def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    class_svm = svm.OneClassSVM(nu=0.1)
    fit = class_svm.fit(train_data)
    return binary2neg_boolean(fit.predict(test_data))