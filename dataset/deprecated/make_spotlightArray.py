import numpy as np

options = [ [-4.5, -10, 1, 95, 0, -25],
            [4.5, -10, 1, 95, 0, 25],
            [0, -10, 1, 95, 0, 0]]

num_images = 40000
len_params = len(options[0])

arr = np.zeros((num_images, len_params))
for ind in range(num_images):
    arr[ind] = options[ind%len(options)]

np.save('arrays/spot1.npy', arr)