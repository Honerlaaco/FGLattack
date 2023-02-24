import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader


# 定义一个恶意的分类器 (防御者具有)
class M_class(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 600),
            nn.ReLU(),

            nn.Linear(600, 200),
            nn.ReLU(),

            nn.Linear(200, 100),
            nn.ReLU(),

            nn.Linear(100, target_dim)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

class ML_class(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 600),
            nn.ReLU(),

            nn.Linear(600, 200),
            nn.ReLU(),

            nn.Linear(200, 100),
            nn.ReLU(),

            nn.Linear(100, target_dim)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)


def Adv_training_rochester_M3(x1, x2, labell, i):

    H_model = torch.load('rochester_model_{}.pkl'.format(i))
    M_model = ML_class(128,6).cuda()
    # C_model = C_class(128, 3).cuda()

    H_optim = optim.Adam(H_model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    M_optim = optim.Adam(M_model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    # C_optim = optim.Adam(C_model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


    z_1 = H_model(x1, x2) #第一次获得的嵌入特征
    z_1.detach_()

    for j in range(20):

        z = H_model(x1, x2) # z为embedding中间表示
        pred = M_model(z)
        # pred_2 = C_model(z)
        loss = nn.CrossEntropyLoss()(pred, labell)
        loss2 = -loss + 50*nn.MSELoss()(z, z_1)
        # loss3 = nn.CrossEntropyLoss()(pred_2, utility_label)

        H_optim.zero_grad()
        loss2.backward(retain_graph=True)
        H_optim.step()
        z.detach_()


        M_optim.zero_grad()
        loss.backward()
        M_optim.step()



    torch.save(H_model, 'rochester_model_{}.pkl'.format(i))