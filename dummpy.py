import numpy as np


test = np.random.RandomState(1)

w = test.normal(0.0, 0.01, 10)
x = np.arange(9.).reshape(3, 3)
print(x)
print(np.where(x > 5, x, -1))
