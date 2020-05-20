import utils

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class BSRTrain(nn.Module):
    def __init__(self, model_func, num_class, lamda=0.001):
        super(BSRTrain, self).__init__()
        self.feature = model_func()

        self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()
        self.lamda = lamda

    def forward(self, x):
        x = Variable(x.cuda())
        feature = self.feature(x)
        u, s, v = torch.svd(feature.t())
        BSR = torch.sum(torch.pow(s, 2))
        scores = self.classifier(feature)
        return scores, BSR

    def forward_loss(self, x, y):
        y = Variable(y.cuda())

        scores, BSR = self.forward(x)
        loss_c = self.loss_fn(scores, y)
        loss = loss_c + self.lamda * BSR

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item() * 100 / (y.size(0) + 0.0), y.size(0))

        return loss

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.
                      format(epoch, i, len(train_loader), avg_loss / float(i + 1), self.top1.val, self.top1.avg))


class PBSRTrain(nn.Module):
    def __init__(self, model_func, num_class, P_matrix, lamda=0.001):
        super(PBSRTrain, self).__init__()

        self.resnet1 = nn.Sequential(*list(model_func()._modules.values())[0][0:6])
        self.resnet2 = nn.Sequential(*list(model_func()._modules.values())[0][6:7])
        self.resnet3 = nn.Sequential(*list(model_func()._modules.values())[0][7:8])
        self.layer1 = nn.Sequential(*list(model_func()._modules.values())[0][8:])

        self.classifier = nn.Linear(model_func().final_feat_dim - 1, num_class)
        self.classifier.bias.data.fill_(0)

        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()

        self.P_matrix = P_matrix
        self.lamda = lamda

    def forward(self, x):
        x = Variable(x.cuda())
        out1 = self.resnet1(x)
        out2 = self.resnet2(out1)
        out3 = self.resnet3(out2)
        fea_b = self.layer1(out3)
        fea_e = torch.mm(fea_b, self.P_matrix)
        u, s, v = torch.svd(fea_e.t())
        BSR = torch.sum(torch.pow(s, 2))
        scores = self.classifier(fea_e)
        return scores, BSR

    def forward_loss(self, x, y):
        y = Variable(y.cuda())

        scores, BSR = self.forward(x)

        loss_c = self.loss_fn(scores, y)
        loss = loss_c + self.lamda * BSR

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item() * 100 / (y.size(0) + 0.0), y.size(0))

        return loss
