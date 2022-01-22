import numpy as np

a = np.random.randint(8, size = (8,4))
b = []
# np.append(a[0], [10])
for i in a:
    # print(a[i])
    b.append(10)
b = np.asarray(b).reshape((np.shape(b)[0],1))
print(b)
a = np.hstack((a, b))
print(a, '\n\n', )
