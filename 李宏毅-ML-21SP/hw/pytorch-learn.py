import torch

#
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#
# print(device)
#
# x = torch.ones(2, 2, requires_grad=True)
# print(x)
# print(x.grad_fn)
#
# y = x+2
# print(y)
# print(y.grad_fn)
# print(x.is_leaf, y.is_leaf)
#
# z = y * y * 3
# z_ = z.mean()
# print('z', z)
# print('z_', z_)

#
# torch.manual_seed(10)
#
# w = torch.tensor([1.], requires_grad=True)
# x = torch.tensor([2.], requires_grad=True)
#
# a = torch.add(w, x)
# b = torch.add(w, 1)
# y = torch.mul(a, b)
#
# y.backward(retain_graph=True)
# print(w.grad)


use_basic_example = False
mnist = True

# import os
# path = './learn-pytorch/ckpts/model/ckpt'
# if os.path.exists(path):
#     print("true")
# else:
#     print("false")
#     os.makedirs(path)
#
# print("-----------")
# if os.path.exists(path):
#     print("true")
# else:
#     print("false")


if use_basic_example:
    import torch
    import numpy as np
    from torch.autograd import Variable
    # 画图象
    import matplotlib.pyplot as plt

    x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                        [9.779], [6.182], [7.59], [2.167], [7.042],
                        [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
    y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                        [3.366], [2.596], [2.53], [1.221], [2.827],
                        [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
    # 转换为 Tensor
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    w = Variable(torch.randn(1), requires_grad=True)
    b = Variable(torch.zeros(1), requires_grad=True)

    x_train = Variable(x_train)
    y_train = Variable(y_train)


    def linear_model(x):
        return w * x + b


    y_ = linear_model(x_train)


    def get_loss(y_, y):
        return torch.mean((y_ - y_train) ** 2)


    loss = get_loss(y_, y_train)

    loss.backward()

    for e in range(10):
        y_ = linear_model(x_train)
        loss = get_loss(y_, y_train)
        w.grad.zero_()
        b.grad.zero_()
        loss.backward()

        w.data = w.data - 1e-2 * w.grad.data
        b.data = b.data - 1e-2 * b.grad.data
        print('epoch: {}, loss: {}'.format(e, loss.item()))

        print(f"{type(loss.data)}, {loss.data}, {type(loss.item())}, {loss.item()}")

if mnist:
    import torch
    import numpy as np
    import torch.nn as nn
    import torch.functional as F
    from torch import optim as optim
    from torch.autograd import Variable
    from torch.utils.data import DataLoader
    from torchvision.datasets import mnist
    from torchvision import transforms
    import matplotlib.pyplot as plt

    device = "cuda"

    # hyper-parameters
    epoches = 3
    batch_size = 64
    learning_rate = 0.0003
    renew = True
    path = './learn-pytorch/ckpts/model.ckpt'


    class Net(nn.Module):
        def __init__(self, in_c=784, out_c=10):
            super(Net, self).__init__()

            self.fc1 = nn.Linear(in_c, 1024)
            self.bn1 = nn.BatchNorm1d(1024)
            self.ac1 = nn.ReLU(inplace=True)

            self.fc2 = nn.Linear(1024, 512)
            self.bn2 = nn.BatchNorm1d(512)
            self.ac2 = nn.ReLU(inplace=True)

            self.fc3 = nn.Linear(512, 256)
            self.bn3 = nn.BatchNorm1d(256)
            self.ac3 = nn.ReLU(inplace=True)

            self.fc4 = nn.Linear(256, 128)
            self.bn4 = nn.BatchNorm1d(128)
            self.ac4 = nn.ReLU(inplace=True)

            self.fc5 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.bn1(x)
            x = self.ac1(x)

            x = self.fc2(x)
            x = self.bn2(x)
            x = self.ac2(x)

            x = self.fc3(x)
            x = self.bn3(x)
            x = self.ac3(x)

            x = self.fc4(x)
            x = self.bn4(x)
            x = self.ac4(x)

            x = self.fc5(x)
            return x


    train_set = mnist.MNIST('./learn-pytorch/data', train=True, transform=transforms.ToTensor(), download=False)
    test_set = mnist.MNIST('./learn-pytorch/data', train=False, transform=transforms.ToTensor(), download=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    net = Net()

    if renew:
        ckpt = torch.load(path)
        net.load_state_dict(ckpt)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)

    losses = []
    acces = []
    eval_losses = []
    eval_acces = []

    for epoch in range(epoches):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for batch, (img, label) in enumerate(train_loader):
            img = img.reshape(img.size(0), -1)
            img = Variable(img)
            label = Variable(label)

            img = img.to(device)
            label = label.to(device)

            out = net(img)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]

            if (batch + 1) % 200 == 0:
                print('[INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f}'.format(epoch + 1,
                                                                                             batch + 1,
                                                                                             loss.item(),
                                                                                             acc))

            train_acc += acc

        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))

        eval_loss = 0
        eval_acc = 0

        net.eval()

        for img, label in test_loader:
            img = img.reshape(img.size(0), -1)
            img = Variable(img)

            img = img.to(device)
            label = label.to(device)

            out = net(img)
            loss = criterion(out, label)
            eval_loss += loss.item()

            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / img.shape[0]

            eval_acc += acc

        eval_losses.append(eval_loss / len(test_loader))
        eval_acces.append(eval_acc / len(test_loader))

        print('[INFO] Epoch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f} | Test: Loss-{:.4f}, Accuracy-{:.4f}'.format(
            epoch + 1, train_loss / len(train_loader), train_acc / len(train_loader), eval_loss / len(test_loader),
            eval_acc / len(test_loader)))

    torch.save(net.state_dict(), path)

    plt.figure()
    plt.suptitle('Test', fontsize=12)
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(eval_losses, color='r')
    ax1.plot(losses, color='b')
    ax1.set_title('Loss', fontsize=10, color='black')
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(eval_acces, color='r')
    ax2.plot(acces, color='b')
    ax2.set_title('Acc', fontsize=10, color='black')
    plt.show()
