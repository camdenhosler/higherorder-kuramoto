import matplotlib.pyplot as plt
import numpy as np

N, num_trials = 3, 2
state_arr = np.array([[1, 10],
                       [2, 20],
                       [3, 30]])  # shape (3, 2)
k_values = np.array([100, 200])

k_flat = np.repeat(k_values, N)
val_flat = state_arr.ravel(order="F")

print(k_flat)    # [100 100 100 200 200 200]
print(val_flat)  # [1 2 3 10 20 30]