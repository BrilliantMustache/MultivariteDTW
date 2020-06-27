import numpy as np
import time

x = np.random.random((100,144))
times = []
s = time.time()
for i in range(100):
    x1 = np.argsort(x[i,:])
e = time.time()
times.append(e-s)


s = time.time()
for i in range(100):
    y = x[i,:]
    x2 = sorted(range(len(y)), key=lambda x: y[x])

e = time.time()
times.append(e-s)

for j in range(144):
    if (x1[j] != x2[j]):
        print('wrong:'+j)

print(times)