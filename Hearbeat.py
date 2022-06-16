import torch.nn as nn
import sys
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from collections import OrderedDict
import torch.nn.functional as F
import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 64, 48),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=4, memory_efficient=False):
 
        super(DenseNet, self).__init__()
 
        # 首层卷积层
        self.Net = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv1d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm1d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
        ]))
 
        # 构建DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config): #构建4个DenseBlock
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.Net.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,  #每个DenseBlock后跟一个TransitionLayer
                                    num_output_features=num_features // 2)
                self.Net.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
 
        # Final batch norm
        self.Net.add_module('norm5', nn.BatchNorm1d(num_features))
 
        # Linear layer
        self.Classifier = nn.Linear(num_features, num_classes) #构建分类器
 
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
 
    def forward(self, x):
        features = self.Net(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool1d(out, 1)
#         print(out.shape)
        out = torch.flatten(out, 1)
#         print(out.shape)
        out = self.Classifier(out)
        return out
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
 
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1) #将之前的层拼接在一起，并且按行展开
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output
    return bn_function
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm1d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv1d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient
 
    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features
class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2)) #尺寸减少一半
class TrainDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return torch.Tensor(data),label
    
    def __len__(self):
        return len(self.data)

def train_dataset(train_path):
    train = pd.read_csv(train_path)
    # 处理训练数据
    train_data = []
    train_label = []
    for item in train.values:
        # train_data
        arr = np.array([float(i) for i in item[1].split(',')])
        arr.resize((1,205))
        train_data.append(arr)
        #train_label
#         arr = np.zeros((4))
#         arr[int(item[2])]=1.0
#         train_label.append(arr)
        train_label.append(item[2])
    
    # 分割训练集和验证集
    data = TrainDataset(train_data, train_label)
    train_size = int(len(data) * 0.8)
    validdate_size = int(len(data)) - train_size
    traindata, validdata = torch.utils.data.random_split(data, [train_size, validdate_size])
    return traindata, validdata
class TestDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        data = self.data[index]
        return torch.Tensor(data)
    
    def __len__(self):
        return len(self.data)
    
def test_dataset(test_path):
    test = pd.read_csv(test_path)
    # 处理训练数据
    test_data = []
    for item in test.values:
        # train_data
        arr = np.array([float(i) for i in item[1].split(',')])
        arr.resize((1,205))
        test_data.append(arr)
    
    testdata = TestDataset(test_data)
    return testdata
def get_acc(out, label):
    total = out.shape[0]
    _, pred_label = out.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total
def train(net, trainiter, validiter, num_epochs, optimizer, criterion):
    device = torch.device('cuda:1' if torch.cuda.is_available else 'cpu')
    net = net.to(device)
    print("train start!")
    for epoch in range(num_epochs):
        net = net.train()
        train_loss = 0
        train_acc = 0
        for data, label in trainiter:
#             # 将装有tensor的list转换为tensor
#             data = torch.stack(data,1)
            
            data = data.to(device)
            label = label.to(device, dtype=torch.int64)
            # 前向传播
            out = net(data)
            loss = criterion(out, label)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算loss和accuracy
            train_loss += loss.item()
            train_acc += get_acc(out, label)
        
        if validiter is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for data, label in validiter:
                data = data.to(device)
                label = label.to(device, dtype=torch.int64)
                out = net(data)
                loss = criterion(out, label)
                
                valid_loss += loss.item()
                valid_acc += get_acc(out, label)
                
            print("Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                    % (epoch, train_loss / len(trainiter),train_acc / len(trainiter), 
                       valid_loss / len(validiter),valid_acc / len(validiter)))
        else:
            print("Epoch %d. Train Loss: %f, Train Acc: %f, "
                   %(epoch, train_loss / len(trainiter),train_acc / len(trainiter)))
# train1(学习率为0.1)
if __name__ == '__main__':
    net = DenseNet()   
    batch_size = 200 
    traindata, validdata = train_dataset('train.csv')
    trainiter = DataLoader(traindata, batch_size=batch_size,shuffle=True)
    validiter = DataLoader(validdata, batch_size=batch_size,shuffle=True)
    num_epochs = 50
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-1) #随机梯度下降
    criterion = nn.CrossEntropyLoss() #loss为交叉熵
    
    train(net, trainiter, validiter, num_epochs, optimizer, criterion)
    torch.save(net, 'Densenet264.pth')

# 计算验证集分数
if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = torch.load('Densenet264.pth')
    model = model.to(device)
    model.eval()  # 转为test模式
    batch_size = 200
#     print(iter(testiter).next().shape)
    result = []
    result_label = []
    for data, label in validiter:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pre = F.softmax(out, 1)
        pre = pre.to('cpu')
#         print(pre)
        result.append(pre)
        result_label.append(label)
    
#     print(result_label)
    result = torch.stack(result, 0) #按照轴0将list转换为tensor
    result = np.array(result)
    result = result.reshape((20000,4))
    result_label = torch.stack(result_label, 0) #按照轴0将list转换为tensor
    result_label = np.array(result_label)
    result_label = result_label.reshape((20000))
    thr = [0.8, 0.45, 0.8, 0.8]
    for x in result:
        for i in [1, 2, 3, 0]:
            if x[i] > thr[i]:
                x[0:i] = 0
                x[i+1:4] = 0
                x[i] = 1

    num = 0
    i = 0
    for x in result:
        ans = 0
        x[int(result_label[i])] -= 1.0
        i += 1
        ans = sum(abs(x))
        num += ans
    print(num)
295.90491063204126
# test
if __name__ == '__main__':
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = torch.load('Densenet264.pth')
    model = model.to(device)
    model.eval()  # 转为test模式
    batch_size = 200
    testdata = test_dataset('testA.csv')
    testiter = DataLoader(testdata, batch_size=batch_size,shuffle=False) # 一定要定义为False!
#     print(iter(testiter).next().shape)
    result = []
    for data in testiter:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        pre = F.softmax(out, 1)
        pre = pre.to('cpu')
#         print(pre)
        result.append(pre)
    result = torch.stack(result, 0) #按照轴0将list转换为tensor
    # 进行数据的后处理，准备提交数据(设置阈值)
    result = np.array(result)
    result = result.reshape((20000,4))
    thr = [0.8, 0.45, 0.8, 0.8]
    for x in result:
        for i in [1, 2, 3, 0]:
            if x[i] > thr[i]:
                x[0:i] = 0
                x[i+1:4] = 0
                x[i] = 1

    id =np.arange(100000,120000)
    df = DataFrame(result, columns=['label_0','label_1','label_2','label_3'])
    df.insert(loc=0, column='id', value=id, allow_duplicates=False) 
    df.to_csv("submit.csv", index_label="id", index = False)
    print(df)
      
