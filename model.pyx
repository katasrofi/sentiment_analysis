cimport numpy as np
cimport pandas as pd

cdef pd.DataFrame dataset = pd.read_csv("text.csv")
cdef pd.DataFrame X = dataset.iloc[:, 1:-1]
cdef pd.DataFrame y = dataset.iloc[: -1]

X_train, y_train, X_test, y_test = split_train_test_set(X, y)
print("Success")

