import pickle
import numpy as np
import time

'''
pos_x = 
'''

x = pickle.load(open('x.dset', 'rb'))
y = pickle.load(open('y.dset', 'rb'))

avgs = []

print('norm x')
for i in range(len(x[0])):
    pos_x = x[:, i]
    pos_x.sort()
    print(pos_x[-1])
    if pos_x[-1] == 0:
        x[:, i] = x[:, i] / 1
    else:
        x[:, i] = x[:, i] / pos_x[-1]
print('norm y')
for i in range(len(y[0])):
    pos_y = y[:, i]
    pos_y.sort()
    print(pos_y[-1])
    if pos_y[-1] == 0:
        y[:, i] = y[:, i] / 1
    else:
        y[:, i] = y[:, i] / pos_y[-1]

print(x[10])
print(y[10])

pickle.dump(x, open('x2.dset', 'wb'))
pickle.dump(y, open('y2.dset', 'wb'))