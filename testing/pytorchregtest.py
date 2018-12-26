import pickle
import numpy as np
import time

x = pickle.load(open('x.dset', 'rb'))
y = pickle.load(open('y.dset', 'rb'))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


numepochs = 5


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

net = Net(n_feature=177, n_hidden=600, n_output=105)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

losses = []
stamp = []
plt.ion()   # something about plotting


for epoch in range(numepochs):
    x, y = unison_shuffled_copies(x, y)


    for row in range(x.shape[0]):
        prediction = net(x[row])     # input x and predict based on x

        loss = loss_func(prediction, y[row])     # must be (1. nn output, 2. target)

        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        if row % 50 == 0:
            losses.append(loss)
            stamp.append(((epoch + 1) * x.shape[0]) + row)


plt.cla()
plt.plot(loss, stamp, 'r-', lw=5)
plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
plt.pause(0.1)
plt.ioff()
plt.show()