import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    mean = np.mean(train_data)
    std = np.std(train_data)
    mean_min = mean - 3 * std
    mean_max = mean + 3 * std
    return [1 if data < mean_min or data > mean_max else 0 for data in test_data]
