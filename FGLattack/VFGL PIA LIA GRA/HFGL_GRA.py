#%%
import os.path as osp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch_geometric.utils

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.dense.diff_pool import dense_diff_pool
import time
import copy
import numpy as np
import torch.nn as nn
import random
from sklearn.metrics import roc_curve


def seed_torch(seed=42):
    random.seed(seed) # python seed
    os.environ['PYTHONHASHSEED'] = str(seed) # 设置python哈希种子，for certain hash-based operations (e.g., the item order in a set or a dict）。seed为0的时候表示不用这个feature，也可以设置为整数。 有时候需要在终端执行，到脚本实行可能就迟了。
    np.random.seed(seed) # If you or any of the libraries you are using rely on NumPy, 比如Sampling，或者一些augmentation。 哪些是例外可以看https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed) # 为当前CPU设置随机种子。 pytorch官网倒是说(both CPU and CUDA)
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    # torch.cuda.manual_seed_all(seed) # 使用多块GPU时，均设置随机种子
    torch.backends.cudnn.deterministic = True
seed_torch(seed=42)


# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TU')
dataset_name = 'ENZYMES' #PROTEINS, NCI1, AIDS, MUTAG, ENZYMES
gnn_name = 'GIN'
# print(path)
path = ''
batchsize = 128
if dataset_name == 'ENZYMES':
    EPOCHS = 200
else:
    EPOCHS = 100


dataset = TUDataset(path, name=dataset_name).shuffle()
print(dataset.num_features)
# dataset.num_features = 1
train_dataset = dataset[7*len(dataset) // 10:]
testdataset = dataset[:3*len(dataset) // 10]
train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
testloader = DataLoader(testdataset, batch_size=batchsize)


#%%
class GINNet(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels):
        super().__init__()

        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv3 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv4 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv5 = GINConv(
            Sequential(Linear(dim, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        """for GCN model"""
        self.conv6 = GCNConv(dataset.num_node_features, 64)
        self.conv7 = GCNConv(64, 192)

        """for GAT model"""
        self.conv8 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv9 = GATConv(8 * 8, 192, heads=1, concat=False,
                             dropout=0.6)

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, x, edge_index, batch):

        # GIN模型
        if gnn_name == 'GIN':
            x = self.conv1(x, edge_index)
            x = self.conv2(x, edge_index)
            x = self.conv3(x, edge_index)
            x = self.conv4(x, edge_index)
            x = self.conv5(x, edge_index)
            x1 = F.dropout(x, p=0.2, training=self.training)


        elif gnn_name == 'GCN':

            # GCN模型
            x = self.conv6(x, edge_index)
            x = x.relu()
            x = self.conv7(x, edge_index)
            x1 = F.dropout(x, p=0.2, training=self.training)


        elif gnn_name == 'GAT':

            x = F.elu(self.conv8(x, edge_index))
            x = self.conv9(x, edge_index)
            x1 = F.dropout(x, p=0.2, training=self.training)

        x = global_add_pool(x1, batch)
        # print(x.shape)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), x1



class AttackNet(torch.nn.Module):
    def __init__(self, dim, out_channels=dataset.num_classes):
        super().__init__()
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, x, batch):
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GINNet(dataset.num_features, 192, dataset.num_classes).to(device) #GIN:192
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#%%
def train():
    model.train()
    total_loss = 0
    for data in train_loader:

        data = data.to(device)
        optimizer.zero_grad()
        output, embedding = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)

model.load_state_dict(torch.load('{}_{}_model.pkl'.format(dataset_name, gnn_name)),strict=False)


#%%
import numpy as np

def metric(ori_adj, inference_adj, idx):

    from sklearn.metrics import roc_curve, auc, average_precision_score, roc_auc_score
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)
    pred_edge[pred_edge >= 0.99] = 1
    pred_edge[pred_edge < 0.99] = 0
    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    index = np.where(real_edge == 0)[0]
    index_delete = np.random.choice(index, size=int(len(real_edge)-2*np.sum(real_edge)), replace=False)
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)
    graph_auc = auc(fpr, tpr)
    graph_ap = average_precision_score(real_edge, pred_edge)
    return graph_auc, graph_ap

def Auc(ori_adj, modified_adj, idx):
    real_edge = []
    pred_edge = []
    for i in idx:
        for j in idx:
            if i != j:
                real_edge.append(ori_adj[i][j])
                pred_edge.append(modified_adj[i][j])

    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    return auc(fpr, tpr)

#%%
def dot_product_decode(Z):
    Z = F.normalize(Z, p=2, dim=1)
    A_pred = torch.relu(torch.matmul(Z, Z.t()))
    return A_pred

from torch_geometric.utils import to_dense_adj
import random

# 图重构部分
attack_model = AttackNet(192).to(device)

#%%
batchsize = 8
testloader = DataLoader(testdataset, batch_size=batchsize)
auc_res = []
ap_res = []
testnums = 5

for i, data in enumerate(testloader):
    if i == testnums: break

    # 测试X张Graph的重构效果
    print('正在测试第{}张图的重构效果'.format(i))
    data = data.to(device)
    output, embedding = model(data.x, data.edge_index, data.batch)

    # 获取参与方的梯度信息
    dy_dx = torch.autograd.grad(output, model.parameters(), grad_outputs=torch.ones_like(output), allow_unused=True)



    original_dy_dx = list()
    for ii, j in enumerate(dy_dx):
        # print(i, j.shape)
        if gnn_name == 'GIN':
            if ii >= 42:
                original_dy_dx.append(j)
        elif gnn_name == 'GCN':
            if ii >= 4:
                original_dy_dx.append(j)
        elif gnn_name == 'GAT':
            if ii >= 8:
                original_dy_dx.append(j)


    # 通过梯度信息恢复嵌入表示信息（全连接层之前）
    # 定义假的数据
    import timeit
    start = timeit.default_timer()
    dummy_data = torch.rand(embedding.size()).to(device).requires_grad_(True)
    optimizer_a = torch.optim.Adam([dummy_data, ], lr=0.01)
    # optimizer_a = torch.optim.LBFGS([dummy_data, ])
    pred_label = data.y
    end = timeit.default_timer()
    print("part1:" + str(end - start))

    start = timeit.default_timer()
    for iters in range(300):

        def closure():
            optimizer.zero_grad()
            pred = attack_model(dummy_data, data.batch)

            dummy_loss = F.nll_loss(pred, pred_label)

            dummy_dy_dx = torch.autograd.grad(dummy_loss, attack_model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                # gx = torch.tensor(gx)
                # gy = torch.tensor(gy)
                # grad_diff += ((F.cosine_similarity(gx, gy, dim=0)) ** 2).sum()
                # grad_diff += ((gx - gy) ** 2).sum()
                # grad_diff += ((gx - gy) ** 1).sum()

                grad_diff += (nn.KLDivLoss()(gx,gy)).sum()

            grad_diff.backward()
            # print(grad_diff.item())
            return grad_diff

        optimizer_a.step(closure)
    embedding = dummy_data
    end = timeit.default_timer()
    print("part2:" + str(end - start))
    # embedding = torch.rand(embedding.shape).cuda() # 假设我们的攻击中直接设置随机噪声

    adj = to_dense_adj(data.edge_index)[0]


    torch.save(data.cpu(), r'./temp/data_{}.pt'.format(i))

    print('save the graph', i)

    print('测试Graph中的{}条边评估攻击效果！'.format(int(adj.shape[0] * 0.5)))
    idx_attack = np.array(random.sample(range(adj.shape[0]), int(adj.shape[0] * 0.5)))
    print('重构方法：编码器选择没有任何背景知识的点乘方法')

    start = timeit.default_timer()
    A_pred = dot_product_decode(embedding)
    A_pred = A_pred - torch.eye(A_pred.shape[0]).to(device)  # 去除对角连边
    #Z=np.max(A_pred.cpu().detach().numpy())
    auc, ap = metric(adj.detach().cpu().numpy(), A_pred.detach().cpu().numpy(), idx_attack)

    print('第{}张图数据重构的攻击效果:auc {:.3f},ap {:.3f}'.format(i, auc, ap))
    print('  ')
    auc_res.append(auc)
    ap_res.append(ap)

    A_pred[A_pred > 0.996] = 1
    A_pred[A_pred < 0.996] = 0



    edge_index = torch_geometric.utils.dense_to_sparse(A_pred)


    torch.save(edge_index, r'./temp/edge_index_{}.pt'.format(i))

    A_pred=A_pred.to(dtype=torch.float32)

    end = timeit.default_timer()
    print("part3:" + str(end - start))

    np.savetxt('./temp/A_pred.csv',A_pred.cpu().detach().numpy(), fmt='%d')



print('对于{}张图数据进行重构攻击，测试的平均auc为 {:.3f}，平均ap为 {:.3f}'.format(testnums, np.mean(auc_res), np.mean(ap_res) ))

