import pickle
import numpy as np
import time

x = pickle.load(open('x.dset', 'rb'))
y = pickle.load(open('y.dset', 'rb'))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data


numepochs = 5


torch.manual_seed(1)    # reproducible

BATCH_SIZE = 512
# BATCH_SIZE = 8

x = torch.tensor(x)      # this is x data (torch tensor)
y = torch.tensor(y)  # this is y data (torch tensor)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=3,              # subprocesses for loading data
    pin_memory=True
)


def show_batch():
    for epoch in range(3):   # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())




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
net.cuda()
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

losses = []
stamp = []
plt.ion()   # something about plotting


if __name__ == '__main__':
    for epoch in range(numepochs):

        for step, (batch_x, batch_y) in enumerate(loader):
            b_x = batch_x.cuda()
            b_y = batch_y.cuda()

            prediction = net(b_x)     # input x and predict based on x

            loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            if step % 50 == 0:
                print(step, loss)
                losses.append(loss)
                stamp.append(((epoch + 1) * x.shape[0]) + step)


    plt.cla()
    plt.plot(loss.numpy(), stamp, 'r-', lw=5)
    plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
    plt.pause(0.1)
    plt.ioff()
    plt.show()