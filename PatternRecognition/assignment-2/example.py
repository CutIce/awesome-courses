import torch
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, Dropout
import torchvision.datasets.mnist as mnist


def download_mnist():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root="./data/", transform=transform, train=True, download=True)
    test_dataset = datasets.MNIST(
        root="./data/", transform=transform, train=False, download=True)

    return train_dataset, test_dataset


def one_hot(y):
    y_ = torch.zeros((y.shape[0], 10))
    y_[torch.arange(y.shape[0], dtype=torch.long), y] = 1
    return y_


def mini_batch(dataset, batch_size=128):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def torch_run():
    train_dataset, test_dataset = download_mnist()

    epoch_number = 1
    learning_rate = 0.1

    for epoch in range(epoch_number):
        for x, y in mini_batch(train_dataset):
            y = one_hot(y)
            print(x.shape)
            print(y)
            break

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(Conv2d(1, 64, 3, 1, 1),
                                         ReLU(),
                                         Conv2d(64, 128, 3, 1, 1),
                                         ReLU(),
                                         MaxPool2d(2, 2)
                                        )
        self.dense = torch.nn.Sequential(Linear(14*14*128, 1024),
                                         ReLU(),
                                         Dropout(p=0.5),
                                         Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x




if __name__ == "__main__":
    train_dataset = datasets.MNIST(root='data/', train=True,
                                   transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='data/', train=False,
                                  transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

    device = torch.device('cuda')
    model = Model().to(device)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.load_state_dict(torch.load('parameter.pkl'))

    print('Training Begins')
    epochs = 5
    for epoch in range(epochs):
        # train
        sum_loss = 0.0
        train_correct = 0
        for data in train_loader:
            inputs, lables = data
            inputs, lables = Variable(inputs).cuda(), Variable(lables).cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = cost(outputs, lables)
            loss.backward()
            optimizer.step()

            _, id = torch.max(outputs.data, 1)
            sum_loss += loss.data
            train_correct += torch.sum(id == lables.data)

        print('%d epoch in %d epochs loss:%.03f' % (epoch + 1, epochs, sum_loss / len(train_loader)))
        print('                 correct:%.03f%%' % (100 * train_correct / len(train_dataset)))
    print('Training Ends')
    print()

    print('#################   Save Model   #########################')
    torch.save(model.state_dict(), "parameter.pkl")  # save
    model.load_state_dict(torch.load('parameter.pkl'))  # load
    print('#################   Save Over    #########################')

    model.eval()
    test_correct = 0
    for data in test_loader:
        inputs, lables = data
        inputs, lables = Variable(inputs).cuda(), Variable(lables).cuda()
        outputs = model(inputs)
        _, id = torch.max(outputs.data, 1)
        test_correct += torch.sum(id == lables.data)
    print("correct in test set:%.3f%%" % (100 * test_correct / len(test_dataset)))