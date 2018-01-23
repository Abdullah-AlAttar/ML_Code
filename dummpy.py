import numpy as np

yes = np.array([25.2, 19.3, 18.5, 21.7, 20.1, 24.3, 22.8, 23.1, 19.8])

no = np.array([27.3, 30.1, 17.4, 29.5, 15.1])

sum = yes - yes.mean()
print(sum)
sum = sum**2
print(sum)
sum = sum.sum()
print(sum)
print(sum / len(yes))
