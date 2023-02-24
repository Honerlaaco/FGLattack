import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import warnings
warnings.filterwarnings(action='ignore')
import time
import sklearn
import argparse
import numpy as np
import random
import scipy.sparse as sp
import torch.nn as nn
import torch.optim as optim
from defense_model import LocalGuard
import torch
import matplotlib.pyplot as plt
import sys
import  pickle as pkl
import  networkx as nx
import  scipy.sparse as sp
from    scipy.sparse.linalg.eigen.arpack import eigsh
import math

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F
# from defense_model import Adv_training_yale

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device', device)

# 加载原始数据

with open('./{}_feats.pkl'.format('yale'), 'rb') as f1:
    feature = pkl.load(f1)
    feature = torch.from_numpy(feature.toarray())
with open('./{}_adj.pkl'.format('yale'), 'rb') as f2:
    adj = pkl.load(f2, encoding='latin1')
    adj = torch.from_numpy(adj.todense()).long()
label = np.load('./{}_labels.npy'.format('yale'))

"""yale数据集 主任务为教育水平推断，隐私属性为性别"""
utility_label = torch.LongTensor(label[:, 5]).cuda()
privacy_property = torch.LongTensor(label[:, 0]).cuda()
print(privacy_property)


train_privacy_property = privacy_property
test_privacy_property = privacy_property

# adj = torch.LongTensor(adj).cuda()


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

idx_train = range(6000)
idx_test = range(6000,privacy_property.shape[0])

train_mask = torch.from_numpy(sample_mask(idx_train, privacy_property.shape[0])).cuda()
test_mask = torch.from_numpy(sample_mask(idx_test, privacy_property.shape[0])).cuda()

# train_utility_label = np.zeros(utility_label.shape)
# test_utility_label = np.zeros(utility_label.shape)

# train_utility_label[train_mask, :] = utility_label[train_mask, :]
# test_utility_label[test_mask, :] = utility_label[test_mask, :]
# 特征进行分割
feature_1 = torch.split(feature, feature.size()[1] // 2, dim=1)[0]
feature_2 = torch.split(feature, feature.size()[1] // 2, dim=1)[1]

#是否注入噪声
# feature_2 = feature_2+1*torch.randn(feature_2.shape)
# print(train_mask)

# 构建VFL模型

class SplitNN(nn.Module):
    def __init__(self, models, optimizers, partition):
        super().__init__()
        self.models = models
        self.optimizers = optimizers
        self.output = [None] * (partition)

    #         self.output.to(device)

    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()

    # Here x is a list having a batch of diffent partitioned datasets.
    def forward(self, x, stage):

        for i in range(len(x)):
            self.output[i] = self.models[i](x[i][0], x[i][1])

            # print(stage)
            if i == 1 and stage == 'train' and epoch == 299:
                noise = LocalGuard(self.output[i], epoch, stage, privacy_property)
                # print(noise)
                # noise = torch.rand(self.output[1].shape).cuda()
                self.output[i] = self.output[i] + 1 * noise.detach()

            if epoch == 299 and stage == 'train':
                torch.save(self.output[1], 'yale_embedding_train.pth')

        # Concatenating the output of various structures in bottom part (alice's location)
        total_out = torch.cat(tuple(self.output[i] for i in range(len(self.output))), dim=1)
        second_layer_inp = total_out.detach().requires_grad_()

        self.second_layer_inp = second_layer_inp
        pred = self.models[-1](second_layer_inp)
        return pred, self.output[1]

    def backward(self):

        second_layer_inp = self.second_layer_inp
        grad = second_layer_inp.grad

        i = 0
        while i < partition - 1:
            self.output[i].backward(grad[:, hidden_sizes[1] * i: hidden_sizes[1] * (i + 1)])
            i += 1

        # This is implemented because it is not necessary that last batch is of exact same size as partitioned.
        self.output[i].backward(grad[:, hidden_sizes[1] * i:])

    def step(self):
        for opt in self.optimizers:
            opt.step()


class GraphConvolution(Module):
    """
    Simple pygGCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeatn, adj):
        support = torch.spmm(infeatn, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN(Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = 0.5

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = torch.relu(x)
        # x = F.dropout(x,p=0.8)
        x = self.gc2(x, adj)
        # x = F.dropout(x,p=0.8)
        return x

def create_models(partition, input_size, hidden_sizes, output_size):
    models = list()
    for _ in range(1, partition):
        models.append(GCN(nfeat=94, nhid=hidden_sizes[0], nclass=hidden_sizes[1]).cuda())

    models.append(GCN(nfeat=94, nhid=hidden_sizes[0], nclass=hidden_sizes[1]).cuda())

    models.append(nn.Sequential(nn.Linear(hidden_sizes[1] * partition, hidden_sizes[2]),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(hidden_sizes[2], output_size),
                                nn.LogSoftmax(dim=1)
                                ).cuda())
    return models

input_size = 7
hidden_sizes = [128, 128, 64]
output_size = 6

partition = 2
models = create_models(partition, input_size, hidden_sizes, output_size)

optimizers = [optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) for model in
              models]

splitNN = SplitNN(models, optimizers, partition).cuda()

# 训练模型

def masked_loss(out, label, mask):
    # print(out)
    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    #     pred = out
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    precision = precision_score(label.cpu(), pred.detach().cpu(),average='macro')
    recall = recall_score(label.cpu(), pred.detach().cpu(),average='macro')
    f1 = f1_score(label.cpu(), pred.detach().cpu(),average='macro')

    return acc, precision, recall, f1


def train(epoch, x, target, splitnn):
    splitnn.zero_grads()
    pred, total_ = splitnn.forward(x, 'train')

    # if epoch == 280:
    # # if epoch%60==0:
    #     torch.save(total_, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/yale_embedding_train.pth')
    loss = masked_loss(pred, target, train_mask)
    loss.backward()
    splitnn.backward()
    splitnn.step()
    return loss.item()


def test_cda(epoch, x, target, splitnn):
    splitnn.eval()
    pred, total_ = splitnn.forward(x,'test')
    # if epoch == 280:
    # # if epoch%60==0:
    #     torch.save(total_, '/data0/BigPlatform/FL/lirongchang/localguard/tmp/yale_embedding_test.pth')
    acc_train, precision, recall, f1 = masked_acc(pred, target, test_mask)
    return acc_train, precision, recall, f1

def adversary():

    # 攻击者的背景知识：考虑训练集中训练攻击模型，在测试集中测试攻击模型
    adv_train = torch.load('yale_embedding_train.pth').cuda()
    adv_train = adv_train + torch.Tensor(np.random.laplace(0, 100, adv_train.shape)).cuda()

    adv_test = adv_train


    from model.attack_model import Adv_class
    from model.attack_model import adversary_class_train
    from model.attack_model import adversary_class_test

    AttackModel = Adv_class(latent_dim=hidden_sizes[1], target_dim=6).cuda()

    # 训练攻击者模型
    # optim_ = optim.SGD(AttackModel.parameters(), lr=0.1, momentum=0.9)
    optim_ = optim.Adam(AttackModel.parameters(), lr=0.001, weight_decay=1e-8)

    acc_ = []
    for i in range(150):
        loss = adversary_class_train(optim_, AttackModel, adv_train, train_privacy_property)

    know_port, acc, f1 = adversary_class_test(optim_, AttackModel, adv_test, test_privacy_property)
    print('the attack test epoch {}: the knowledge {} ==> acc is {}, f1 is {}.'.format(i, know_port, acc, f1))


# 训练模型

epochs = 300
loss_list = list()
cda_list = list()


# y_train_x = torch.from_numpy(y_train).long().detach().clone()
# y_test_a = y_test.detach().clone()

adj = torch.tensor(adj, dtype=torch.float32).cuda()
feature_1 = torch.tensor(feature_1, dtype=torch.float32).cuda()
feature_2 = torch.tensor(feature_2, dtype=torch.float32).cuda()

# 是否添加随机噪声
# feature_2 = torch.add(10*torch.randn(feature_2.shape).cuda(), feature_2)
def noisyCount(sensitivety, epsilon):
    beta = sensitivety / epsilon
    u1 = np.random.random()
    u2 = np.random.random()
    if u1 <= 0.5:
        n_value = -beta * np.log(1. - u2)
    else:
        n_value = beta * np.log(u2)
    # print(n_value)
    return n_value

def laplace_mech(data, sensitivety, epsilon):
    for i in range(len(data)):
        data[i] += noisyCount(sensitivety, epsilon)
    return data

# feature_2 = laplace_mech(feature_2, 1,0.5)

print('starting!')
for epoch in range(epochs):
    total_loss = 0

    loss = train(epoch, [(feature_1, adj), (feature_2, adj)], utility_label, splitNN)

    loss_list.append(loss)
    #     print(f"Epoch: {epoch+1}... Training Loss: {loss}")
    # if epoch%60==0:
    #     adversary()
    # # 测试干净样本分类准确率
    cda, precision, recall,f1 = test_cda(epoch, [(feature_1, adj), (feature_2, adj)], utility_label, splitNN)
    # cda, precision, recall, f1 = torch.round(cda, 3), torch.round(precision,3), torch.round(recall, 3), torch.round(f1, 3)
    cda_list.append(cda)

    print(f"Epoch: {epoch + 1}... testing accuracy is: {cda} precition {precision} recall {recall} f1 {f1}")


print('the max cda in the exp is', max(cda_list))


# 泄露测试
import timeit
start = timeit.default_timer()
adversary()
end = timeit.default_timer()
print(str(end-start))
