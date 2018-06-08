import numpy as np
from sklearn.covariance import MinCovDet


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    estimated_covarianvce = MinCovDet().fit(train_data)
    train_dist = estimated_covarianvce.mahalanobis(train_data)
    np_max = np.max(train_dist)

    return [1 for data in test_data]

