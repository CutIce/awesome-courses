import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


train_path = 'covid.train.csv'
test_path = 'covid.test.csv'

config = {
    # 'ckpt': False,
    'ckpt': True,
    '1': {
        'epoch': 1000,
        'batch_size': 4,
        'learning_rate': 0.0003,
        'save_path': 'multi_models/model1.pth',

    },
    '2': {
        'epoch': 1000,
        'batch_size': 4,
        'learning_rate': 0.0003,
        'save_path': 'multi_models/model2.pth',
    },
    '3': {
        'epoch': 1000,
        'batch_size': 4,
        'learning_rate': 0.0003,
        'save_path': 'multi_models/model3.pth',
    },



    'early_stop_cnt': 500,


}



# Hyper-Parameters
randomSeed = 1000
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
np.random.seed(randomSeed)
torch.manual_seed(randomSeed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(randomSeed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title='', kind='1'):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()

    save_path = './multi_models/model' + kind + '_learning_curve.png'
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None, kind='1'):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    save_path = './multi_models/model' + kind + '_pred_record.png'
    plt.savefig(save_path, bbox_inches='tight')
    # plt.show()



class Covid19DataSet(Dataset):
    
    def __init__(self, path, mode='train', kind='1'):

        with open(path, 'r') as fr:
            row_data = list(csv.reader(fr))
            data = np.array(row_data[1:])[:, 1:].astype('float')
        
        id = []
    
        if mode == 'train' or mode == 'dev':
            if kind == '1':
                col1 = list(range(58))
                col2 = list(range(40)) + list(range(58, 76))
                col3 = list(range(40)) + list(range(76, 94))
    
                id.append(col1)
                id.append(col2)
                id.append(col3)
            elif kind == '2':
                col1 = list(range(76))
                col2 = list(range(40)) + list(range(58, 94))
    
                id.append(col1)
                id.append(col2)
            else:
                col1 = list(range(94))
                id.append(col1)
            for i in id:
                if i == id[0]:
                    tmpdata = data[:, i]
                    self.data = tmpdata[:, 0:-1]
                    self.target = tmpdata[:, -1]
                    # print('self.data', type(self.data), 'self.target', type(self.target))
                    # print('self.data', self.data.shape, 'self.target', self.target.shape)
                else:
                    tmpdata = data[:, i]
                    # print('new', type(tmptrain), 'new ', type(tmptest))
                    # print('new', tmptrain.shape, 'new', tmptest.shape)
                    self.data = np.concatenate([self.data, tmpdata[:, 0:-1]], axis=0)
                    self.target = np.concatenate([self.target, tmpdata[:, -1]], axis=0)
                    # print('self.data', self.data.shape, 'self.target', self.target.shape)
    
            if mode == 'train':
                indices = [i for i in range(len(self.data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(self.data)) if i % 10 == 0]
    
            self.data = torch.FloatTensor(self.data[indices])
            self.target = torch.FloatTensor(self.target[indices])
    
            print(mode, kind, '  self.data: ', type(self.data), self.data.shape)
            print(mode, kind, '  self.target: ', type(self.target), self.target.shape)
    
        elif mode == 'test':
            if kind == '1':
                col = list(range(40)) + list(range(76, 93))
                id.append(col)
            elif kind == '2':
                col = list(range(40)) + list(range(58, 93))
                id.append(col)
            else:
                col = list(range(93))
                id.append(col)
    
            self.data = data[:, id[0]]
            self.data = torch.FloatTensor(self.data)
            print(mode, kind, 'self.data: ', type(self.data), self.data.shape)
        
        # Normalize Features
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)
        
        
        self.mode = mode
        self.kind = kind
        self.dim = self.data.shape[1]
    
    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]
        
    def __len__(self):
        return len(self.data)

    def getKind(self):
        return self.kind

    def getMode(self):
        return self.mode


def prep_dataloader(path, mode, kind, batch_size, n_jobs=0):
    dataset = Covid19DataSet(path, mode, kind)
    dateloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True
    )
    return dateloader



class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(32, 1),
        )

        self.criterion = nn.MSELoss(reduce='mean')

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target, print_flag):
        loss_w = 0
        for name, para in self.named_parameters():
            if 'weight' in name:
                loss_w += torch.sum(torch.pow(para, 2))

        loss_e = self.criterion(pred, target)

        if print_flag:
            print('Loss:   ', loss_e)
            print('L2  :   ', loss_w)

        return loss_e + 0.0015 * loss_w

def train(tr_set, dv_set, model, config, device='cuda'):
    n_epochs = config[tr_set.dataset.getKind()]['epoch']

    optimizer = torch.optim.Adam(model.parameters(), lr=config[tr_set.dataset.getKind()]['learning_rate'],  betas=(0.9, 0.99), eps=1e-08, weight_decay=0.005)

    min_mse = 10000.0

    loss_record = {'train': [], 'dev':[]}
    early_stop_cnt = 0

    epoch = 0

    while epoch < n_epochs:
        print('------------------------   ', epoch, '   -------------------------')
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            mse_loss = model.cal_loss(pred, y, print_flag=False)
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())

        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('save model (epoch = {:4d}, loss = {:.4f})'.format(epoch+1, min_mse))
            torch.save(model.state_dict(), config[tr_set.dataset.getKind()]['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop_cnt']:
            print('Early Stop !')
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            mse_loss = model.cal_loss(pred, y, print_flag=False)
        total_loss += mse_loss.detach().cpu().item() * len(x)
    total_loss /= len(dv_set.dataset)
    print('loss:  ', total_loss)
    return total_loss


def test(ts_set, model, device):
    model.eval()
    preds = []
    for x in ts_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    print('Model ', ts_set.dataset.getKind(), ' --------------- Testing End')
    return preds


device = get_device()
os.makedirs('multi_models', exist_ok=True)

# tr_set_1 = Covid19DataSet(train_path, 'train', '1')
# vl_set_1 = Covid19DataSet(train_path, 'dev', '1')
# ts_set_1 = Covid19DataSet(test_path, 'test', '1')
# 
# tr_set_2 = Covid19DataSet(train_path, 'train', '2')
# vl_set_2 = Covid19DataSet(train_path, 'dev', '2')
# ts_set_2 = Covid19DataSet(test_path, 'test', '2')
# 
# tr_set_3 = Covid19DataSet(train_path, 'train', '3')
# vl_set_3 = Covid19DataSet(train_path, 'dev', '3')
# ts_set_3 = Covid19DataSet(test_path, 'test', '3')

tr_set1 = prep_dataloader(train_path, 'train', '1', config['1']['batch_size'])
dv_set1 = prep_dataloader(train_path, 'dev', '1', config['1']['batch_size'])
ts_set1 = prep_dataloader(test_path, 'test', '1', config['1']['batch_size'])

tr_set2 = prep_dataloader(train_path, 'train', '2', config['2']['batch_size'])
dv_set2 = prep_dataloader(train_path, 'dev', '2', config['2']['batch_size'])
ts_set2 = prep_dataloader(test_path, 'test', '2', config['2']['batch_size'])

tr_set3 = prep_dataloader(train_path, 'train', '3', config['3']['batch_size'])
dv_set3 = prep_dataloader(train_path, 'dev', '3', config['3']['batch_size'])
ts_set3 = prep_dataloader(test_path, 'test', '3', config['3']['batch_size'])

model1 = DNN(tr_set1.dataset.dim).to(device)
model2 = DNN(tr_set2.dataset.dim).to(device)
model3 = DNN(tr_set3.dataset.dim).to(device)

if config['ckpt']:
    ckpt1 = torch.load(config['1']['save_path'], map_location='cuda')
    model1.load_state_dict(ckpt1)
    
    ckpt2 = torch.load(config['2']['save_path'], map_location='cuda')
    model2.load_state_dict(ckpt2)
    
    ckpt3 = torch.load(config['3']['save_path'], map_location='cuda')
    model3.load_state_dict(ckpt3)

def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


# ------------  Model 1 -------------------
model1_loss, model1_loss_record = train(tr_set1, dv_set1, model1, config, device)
plot_pred(dv_set1, model1, device, kind='1')

pred1 = test(ts_set1, model1, device)
print('1', type(pred1), len(pred1))

save_pred(pred1, './multi_models/pred1.csv')

# ------------  Model 2 -------------------
model2_loss, model2_loss_record = train(tr_set2, dv_set2, model2, config, device)
plot_pred(dv_set2, model2, device, kind='2')

pred2 = test(ts_set2, model2, device)
print('2', type(pred2), len(pred2))

save_pred(pred2, './multi_models/pred2.csv')

# ------------  Model 3 -------------------
model3_loss, model3_loss_record = train(tr_set3, dv_set3, model3, config, device)
plot_pred(dv_set3, model3, device, kind='3')

pred3 = test(ts_set3, model3, device)
print('3', type(pred3), len(pred3))

save_pred(pred3, './multi_models/pred3.csv')


p1 = list(pred1)
p2 = list(pred2)
p3 = list(pred3)
res = []
print(len(p3))
for i in range(len(p3)):
    tmp = 0.1 * p1[i] + 0.3 * p2[i] + 0.6 * p3[i]
    res.append(tmp)

save_pred(res, './multi_models/milti.csv')
