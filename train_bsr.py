import numpy as np
import torch
import torch.optim
import os

import configs
from methods.baselinetrain_bsr import BSRTrain

from io_utils import model_dict, parse_args
from datasets import miniImageNet_few_shot


def train(base_loader, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif optimization == 'SGD':
        optimizer = torch.optim.SGD(list(model.parameters()), lr=0.001, momentum=0.9, weight_decay=0.0005)
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    out_pre = '%s_%s' % (params.model, params.method)
    model.train()
    for epoch in range(start_epoch, stop_epoch):
        model.train_loop(epoch, base_loader, optimizer)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '%s_%d.tar' % (out_pre, epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    outfile_final = os.path.join(params.checkpoint_dir, '%s.tar' % out_pre)
    torch.save({'epoch': stop_epoch - 1, 'state': model.state_dict()}, outfile_final)
        
    return model

if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')

    image_size = 224
    optimization = 'SGD'

    datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size=16)
    base_loader = datamgr.get_data_loader(aug=params.train_aug)

    model = BSRTrain(model_dict[params.model], params.num_classes, lamda=params.lamda)

    model = model.cuda()
    save_dir = configs.save_dir

    params.method = 'bsr'
    params.checkpoint_dir = '%s/checkpoints/%s_%s' % (save_dir, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    model = train(base_loader, model, optimization, start_epoch, stop_epoch, params)