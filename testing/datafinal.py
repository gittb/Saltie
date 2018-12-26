import pickle
import numpy as np
import time

'''
simplying the horrid data structure
'''
all = []
total = 0


for i in range(0, 8):
    data = pickle.load(open('final_' + str(i) + '_.pset', 'rb'))
    all.extend(data)


all = np.array(all)
#print(all[50][2].shape)

x = []
y = []
count = 0

for game in all:
    print(len(x))
    print('#############')
    print(game[0].shape[0])
    print('#############')
    for frame in range(game[0].shape[0]):
        xtemp = [x for x in game[0][frame]]
        xtemp.extend([x for x in game[2][frame]])
        if len(xtemp) != 177:
            pass
        else:
            y.append(game[1][frame])
            x.append(xtemp)
            count += 1

x = np.array(x)
y = np.array(y)
time.sleep(5)

x = x.astype(np.float32)
y = y.astype(np.float32)
time.sleep(5)
print(count)
print(x.shape)
print(y.shape)

pickle.dump(x, open('x.dset', 'wb'))
pickle.dump(y, open('y.dset', 'wb'))
