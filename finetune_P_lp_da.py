import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import os
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import sqrtm

import configs
from methods.baselinefinetune_P import PFinetune

from io_utils import model_dict, parse_args

from datasets import ISIC_few_shot_da, EuroSAT_few_shot_da, CropDisease_few_shot_da, Chest_few_shot_da


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x


def finetune(novel_loader, n_query=15, freeze_backbone=False, n_way=5, n_support=5):

    iter_num = len(novel_loader)

    acc_all_ori = []
    acc_all_lp = []

    if params.use_saved:
        save_dir = '%s/saved' % configs.save_dir
    else:
        save_dir = '%s/checkpoints' % configs.save_dir

    P_matrix_file = os.path.join(save_dir, 'P_matrix.npy')
    P_matrix = torch.from_numpy(np.load(P_matrix_file)).float().cuda()

    for _, (x_all, y_all) in enumerate(novel_loader):
        pretrained_model = []
        classifier = []
        classifier_opt = []
        delta_opt = []
        ###############################################################################################
        # load pretrained model on miniImageNet
        for i in range(params.M):
            pretrained_model.append(PFinetune(model_dict[params.model], P_matrix[i]))
            checkpoint_dir = '%s/%s_%s' % (save_dir, params.model, params.method)
            if params.train_aug:
                checkpoint_dir += '_aug'
            modelfile = os.path.join(checkpoint_dir, '%s_%s_e%d.tar' % (params.model, params.method, i))
            tmp = torch.load(modelfile)
            pretrained_dict = tmp['state']

            new_dict = pretrained_model[i].state_dict()
            pretrained_dict = {u: v for u, v in pretrained_dict.items()
                               if u in new_dict}
            new_dict.update(pretrained_dict)
            pretrained_model[i].load_state_dict(new_dict)

            classifier.append(Classifier(pretrained_model[i].final_feat_dim, n_way))

            classifier_opt.append(torch.optim.SGD(
                classifier[i].parameters(),
                lr=0.01, momentum=0.9, dampening=0.9,
                weight_decay=0.001))

            if freeze_backbone is False:
                delta_opt.append(torch.optim.SGD(
                    filter(lambda p: p.requires_grad, pretrained_model[i].parameters()),
                    lr=0.01))

            pretrained_model[i].cuda()
            classifier[i].cuda()
            ###############################################################################################
            if freeze_backbone is False:
                pretrained_model[i].train()
            else:
                pretrained_model[i].eval()

            classifier[i].train()

        ###############################################################################################
        batch_size = 4
        support_size = n_way * n_support
        x_b_i = []
        for aug, (x, y) in enumerate(zip(x_all, y_all)):
            n_query = x.size(1) - n_support
            x = x.cuda()
            x_var = Variable(x)

            y_a_i_tmp = Variable(torch.from_numpy(np.repeat(range(n_way), n_support))).cuda()

            x_b_i.append(x_var[:, n_support:, :, :, :].contiguous().view(n_way * n_query, *x.size()[2:]))
            x_a_i_tmp = x_var[:, :n_support, :, :, :].contiguous().view(n_way * n_support, *x.size()[2:])
            if aug == 0:
                x_a_i = x_a_i_tmp
                y_a_i = y_a_i_tmp
            else:
                x_a_i = torch.cat((x_a_i, x_a_i_tmp), 0)
                y_a_i = torch.cat((y_a_i, y_a_i_tmp), 0)
        ###############################################################################################
        loss_fn = nn.CrossEntropyLoss().cuda()
        ###############################################################################################
        total_epoch = 100
        support_size_all = support_size * 5
        for epoch in range(total_epoch):
            rand_id = np.random.permutation(support_size_all)

            for j in range(0, support_size_all, batch_size):

                #####################################
                selected_id = torch.from_numpy(rand_id[j: min(j+batch_size, support_size_all)]).cuda()
               
                z_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id] 
                #####################################
                for i in range(params.M):
                    classifier_opt[i].zero_grad()
                    if freeze_backbone is False:
                        delta_opt[i].zero_grad()
                    output = pretrained_model[i](z_batch)
                    output = classifier[i](output)

                    loss = loss_fn(output, y_batch)
                    #####################################
                    loss.backward()

                    classifier_opt[i].step()
                    if freeze_backbone is False:
                        delta_opt[i].step()

        scores_ori = 0
        scores_lp = 0

        y_query = np.repeat(range(n_way), n_query)

        n_lp = len(y_query)
        del_n = int(n_lp * (1.0 - params.delta))
        with torch.no_grad():
            for k in range(params.M):
                pretrained_model[k].eval()
                classifier[k].eval()
                for x_b_i_tmp in x_b_i:
                    output = pretrained_model[k](x_b_i_tmp)
                    scores_tmp = classifier[k](output)
                    scores_tmp = F.softmax(scores_tmp, 1)

                    scores_ori += scores_tmp

                    x_lp = output.cpu().numpy()
                    y_lp = scores_tmp.cpu().numpy()
                    neigh = NearestNeighbors(params.k_lp)
                    neigh.fit(x_lp)
                    d_lp, idx_lp = neigh.kneighbors(x_lp)
                    d_lp = np.power(d_lp, 2)
                    sigma2_lp = np.mean(d_lp)

                    for i in range(n_way):
                        yi = y_lp[:, i]
                        top_del_idx = np.argsort(yi)[0:del_n]
                        y_lp[top_del_idx, i] = 0

                    w_lp = np.zeros((n_lp, n_lp))
                    for i in range(n_lp):
                        for j in range(params.k_lp):
                            xj = idx_lp[i, j]
                            w_lp[i, xj] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
                            w_lp[xj, i] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
                    q_lp = np.diag(np.sum(w_lp, axis=1))
                    q2_lp = sqrtm(q_lp)
                    q2_lp = np.linalg.inv(q2_lp)
                    L_lp = np.matmul(np.matmul(q2_lp, w_lp), q2_lp)
                    a_lp = np.eye(n_lp) - params.alpha * L_lp
                    a_lp = np.linalg.inv(a_lp)
                    ynew_lp = np.matmul(a_lp, y_lp)

                    scores_lp += ynew_lp

        count_this = len(y_query)

        topk_scores, topk_labels = scores_ori.data.topk(1, 1, True, True)
        topk_ind_ori = topk_labels.cpu().numpy()
        top1_correct_ori = np.sum(topk_ind_ori[:, 0] == y_query)
        correct_ori = float(top1_correct_ori)
        print('BSR+DA (Ensemble): %f' % (correct_ori / count_this * 100))
        acc_all_ori.append((correct_ori / count_this * 100))

        topk_ind_lp = np.argmax(scores_lp, 1)
        top1_correct_lp = np.sum(topk_ind_lp == y_query)
        correct_lp = float(top1_correct_lp)
        print('BSR+LP+DA (Ensemble): %f' % (correct_lp / count_this * 100))
        acc_all_lp.append((correct_lp / count_this * 100))
        ###############################################################################################

    acc_all_ori = np.asarray(acc_all_ori)
    acc_mean_ori = np.mean(acc_all_ori)
    acc_std_ori = np.std(acc_all_ori)
    print('BSR+DA (Ensemble): %d Test Acc = %4.2f%% +- %4.2f%%' %
          (iter_num, acc_mean_ori, 1.96 * acc_std_ori / np.sqrt(iter_num)))

    acc_all_lp = np.asarray(acc_all_lp)
    acc_mean_lp = np.mean(acc_all_lp)
    acc_std_lp = np.std(acc_all_lp)
    print('BSR+LP+DA (Ensemble): %d Test Acc = %4.2f%% +- %4.2f%%' %
          (iter_num, acc_mean_lp, 1.96 * acc_std_lp / np.sqrt(iter_num)))


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('finetune')

    image_size = 224
    iter_num = 600
    params.method = 'Pbsr'
    params.M = 10

    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, n_query=15)
    freeze_backbone = params.freeze_backbone

    if params.dtarget == 'ISIC':
        print ("Loading ISIC")
        datamgr = ISIC_few_shot_da.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=True)
    elif params.dtarget == 'EuroSAT':
        print ("Loading EuroSAT")
        datamgr = EuroSAT_few_shot_da.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=True)
    elif params.dtarget == 'CropDisease':
        print ("Loading CropDisease")
        datamgr = CropDisease_few_shot_da.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=True)
    elif params.dtarget == 'ChestX':
        print ("Loading ChestX")
        datamgr = Chest_few_shot_da.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=True)

    print (params.dtarget)
    print (freeze_backbone)
    finetune(novel_loader, freeze_backbone=freeze_backbone, **few_shot_params)
