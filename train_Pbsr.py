import numpy as np
import torch
import torch.optim
import os

import configs
from methods.baselinetrain_bsr import PBSRTrain

from io_utils import model_dict, parse_args
from datasets import miniImageNet_few_shot


def train(base_loader, model, optimization, start_epoch, stop_epoch, params):
    optimizer = []
    out_pre = []
    for i in range(params.M):
        if optimization == 'Adam':
            optimizer.append(torch.optim.Adam(model[i].parameters()))
        elif optimization == 'SGD':
            optimizer.append(torch.optim.SGD(list(model[i].parameters()), lr=0.001, momentum=0.9, weight_decay=0.0005))
        else:
            raise ValueError('Unknown optimization, please define by yourself')
        out_pre.append('%s_%s_e%d' % (params.model, params.method, i))
        model[i].train()

    print_freq = 10
    for epoch in range(start_epoch, stop_epoch):
        avg_loss = np.zeros(params.M)
        for j, (x, y) in enumerate(base_loader):
            for i in range(params.M):
                optimizer[i].zero_grad()
                loss = model[i].forward_loss(x, y)
                loss.backward()
                optimizer[i].step()

                avg_loss[i] = avg_loss[i] + loss.item()
                if j % print_freq == 0:
                    print('Epoch {:d} | Batch {:d}/{:d} | E({:d}) | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.
                          format(epoch, j, len(base_loader), i, avg_loss[i] / float(j + 1),
                                 model[i].top1.val, model[i].top1.avg))

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            for i in range(params.M):
                outfile = os.path.join(params.checkpoint_dir, '%s_%d.tar' % (out_pre[i], epoch))
                torch.save({'epoch': epoch, 'state': model[i].state_dict()}, outfile)

    for i in range(params.M):
        outfile_final = os.path.join(params.checkpoint_dir, '%s.tar' % out_pre[i])
        torch.save({'epoch': stop_epoch - 1, 'state': model[i].state_dict()}, outfile_final)
    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    image_size = 224
    optimization = 'SGD'

    datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size=16)
    base_loader = datamgr.get_data_loader(aug=params.train_aug)

    if params.use_saved:
        P_matrix_file = '%s/saved/P_matrix.npy' % configs.save_dir
    else:
        P_matrix_file = '%s/checkpoints/P_matrix.npy' % configs.save_dir
    P_matrix = torch.from_numpy(np.load(P_matrix_file)).float().cuda()
    params.M = 10
    model = []

    for i in range(params.M):
        model_i = PBSRTrain(model_dict[params.model], params.num_classes, P_matrix[i], lamda=params.lamda)
        model_i = model_i.cuda()
        model.append(model_i)

    save_dir = configs.save_dir

    params.method = 'Pbsr'
    params.checkpoint_dir = '%s/checkpoints/%s_%s' % (save_dir, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    model = train(base_loader, model, optimization, start_epoch, stop_epoch, params)