import numpy as np
from sklearn.linear_model import LinearRegression
from venn_abers import VennAbersRegressor

X_train = np.random.rand(100, 5)
y_train = np.random.rand(100)
X_test = np.random.rand(10, 5)

model = LinearRegression()
model.fit(X_train, y_train)

va = VennAbersRegressor(estimator=model, inductive=False, n_splits=5)
va.fit(X_train, y_train, m=1)
preds, intervals = va.predict(X_test)

print(f"Preds shape: {preds.shape}")
print(f"Intervals type: {type(intervals)}")
if isinstance(intervals, np.ndarray):
    print(f"Intervals shape: {intervals.shape}")
    print(f"Intervals content snippet:\n{intervals[:2]}")
else:
    print(f"Intervals: {intervals}")
