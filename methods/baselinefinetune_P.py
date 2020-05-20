import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class PFinetune(nn.Module):
    def __init__(self, model_func, P_matrix):
        super(PFinetune, self).__init__()
        self.resnet1 = nn.Sequential(*list(model_func()._modules.values())[0][0:6])
        self.resnet2 = nn.Sequential(*list(model_func()._modules.values())[0][6:7])
        self.resnet3 = nn.Sequential(*list(model_func()._modules.values())[0][7:8])
        self.layer1 = nn.Sequential(*list(model_func()._modules.values())[0][8:])

        self.final_feat_dim = model_func().final_feat_dim - 1
        self.P_matrix = P_matrix

    def forward(self, x):
        x = Variable(x.cuda())
        out1 = self.resnet1(x)
        out2 = self.resnet2(out1)
        out3 = self.resnet3(out2)
        fea_b = self.layer1(out3)
        fea_e = torch.mm(fea_b, self.P_matrix)
        return fea_e
