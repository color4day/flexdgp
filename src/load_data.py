import numpy as np
from sklearn.model_selection import train_test_split

# 加载数据函数
def load_data(datafile=None, test_ratio=0.2, seed=42):
    if datafile is None:
        raise ValueError("datafile must be specified")
    if not isinstance(datafile, str):
        raise TypeError("datafile must be a string representing the file path")
    data = np.loadtxt(datafile)
    x = data[:, 0:1]
    y = data[:, 1:2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=seed)
    return x_train, x_test, y_train, y_test

