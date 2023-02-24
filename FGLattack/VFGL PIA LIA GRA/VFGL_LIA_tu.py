
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit, Amazon
from torch_geometric.nn import SplineConv, GATConv, AGNNConv, SGConv, GCNConv

model_name = 'gcn'
dataset = 'Citeseer'
transform = T.Compose([
    T.RandomNodeSplit(num_val=500, num_test=500),
    T.TargetIndegree(),
])
path = './data/Citeseer'
print(path)
dataset = Planetoid(path, dataset, transform=transform)
data = dataset[0]

#%%

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SplineConv(dataset.num_features, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, dataset.num_classes, dim=1, kernel_size=2)

    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x1 = F.elu(self.conv1(x, edge_index, edge_attr))
        x2 = F.dropout(x1, training=self.training)
        x2 = self.conv2(x2, edge_index, edge_attr)
        return F.log_softmax(x2, dim=1), x1

import torch_geometric.transforms as T
gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128,
                                       dim=0), exact=True)

class Net_gcn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16, cached=True)
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x1 = self.conv1(x, edge_index, edge_weight)
        # x1 = x1 + torch.Tensor(np.random.laplace(0, 0.09, x1.shape)).cuda()
        # x = F.relu(x1)
        # x = F.dropout(x, training=self.training)
        x = self.conv2(x1, edge_index, edge_weight)

        return F.log_softmax(x, dim=1), x1

class Net_gat(torch.nn.Module):
    def __init__(self, in_channels=dataset.num_features, out_channels=dataset.num_classes):
        super().__init__()

        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x1 = F.dropout(x, p=0.6, training=self.training)
        # x1 = x1 + torch.Tensor(np.random.laplace(0, 0.03, x1.shape)).cuda()
        x = self.conv2(x1, edge_index)
        return F.log_softmax(x, dim=-1), x1

class Net_agnn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(dataset.num_features, 16)
        self.prop1 = AGNNConv(requires_grad=False)
        self.prop2 = AGNNConv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, dataset.num_classes)

    def forward(self):
        x = F.dropout(data.x, training=self.training)
        x = F.relu(self.lin1(x))
        x1 = self.prop1(x, data.edge_index)
        x = self.prop2(x1, data.edge_index)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1), x1

    def __init__(self):
        super().__init__()
        self.conv1 = SGConv(dataset.num_features, 16, K=2,
                            cached=True)
        self.conv2 = SGConv(16, dataset.num_classes, K=2,
                            cached=True)

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x1 = self.conv1(x, edge_index)
        x = self.conv1(x1, edge_index)
        return F.log_softmax(x, dim=1), x1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if model_name == 'SplineCNN':
    model = Net().to(device)
elif model_name == 'gcn':
    model = Net_gcn().to(device)
elif model_name == 'gat':
    model = Net_gat().to(device)
elif model_name == 'agnn':
    model = Net_agnn().to(device)
elif model_name == 'sgc':
    model = Net_sgc().to(device)

data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)

def train():
    model.train()
    optimizer.zero_grad()

    if model_name == 'SplineCNN':
        out, _ = model()
    elif model_name == 'gcn':
        out, _ = model()
    elif model_name == 'gat':
        out, _ = model(data.x, data.edge_index)
    elif model_name == 'agnn':
        out, _ = model()
    elif model_name == 'sgc':
        out, _ = model()

    F.nll_loss(out[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def tet1():
    model.eval()
    if model_name == 'SplineCNN':
        log_probs, _ = model()
    elif model_name == 'gcn':
        log_probs, _ = model()
    elif model_name == 'gat':
        log_probs, _ = model(data.x, data.edge_index)
    elif model_name == 'agnn':
        log_probs, _ = model()
    elif model_name == 'sgc':
        log_probs, _ = model()

    accs = []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 100):
    train()
    train_acc, test_acc = tet1()
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Test: {test_acc:.4f}')


#%%
if model_name == 'SplineCNN':
    out, embedding = model()
if model_name == 'gcn':
    out, embedding = model()
elif model_name == 'gat':
    out, embedding = model(data.x, data.edge_index)
elif model_name == 'agnn':
    out, embedding = model()
elif model_name == 'sgc':
    out, embedding = model()

import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score, roc_auc_score

def metric(ori_adj, inference_adj, idx):
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)
    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    index = np.where(real_edge == 0)[0]
    index_delete = np.random.choice(index, size=int(len(real_edge)-2*np.sum(real_edge)), replace=False)
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)
    print("Inference attack AUC: %f AP: %f" % (auc(fpr, tpr), average_precision_score(real_edge, pred_edge)))

def Auc(ori_adj, modified_adj, idx):
    real_edge = []
    pred_edge = []
    for i in idx:
        for j in idx:
            if i != j:
                real_edge.append(ori_adj[i][j])
                pred_edge.append(modified_adj[i][j])

    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    print(auc(fpr, tpr))
    return auc(fpr, tpr)

import timeit
start = timeit.default_timer()
embedding_p = embedding
def dot_product_decode(Z):
    Z = F.normalize(Z, p=2, dim=1)
    A_pred = torch.relu(torch.matmul(Z, Z.t()))
    return A_pred


from torch_geometric.utils import to_dense_adj
import random
adj = to_dense_adj(data.edge_index)[0]
idx_attack = np.array(random.sample(range(adj.shape[0]), int(adj.shape[0]*0.1)))

A_pred = dot_product_decode(embedding_p)
A_pred[A_pred>0.996]=1
A_pred[A_pred<0.996]=0
metric(adj.detach().cpu().numpy(), A_pred.detach().cpu().numpy(), idx_attack)
end = timeit.default_timer()
print(str(end-start))

