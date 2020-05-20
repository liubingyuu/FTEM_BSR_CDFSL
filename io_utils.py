import argparse
import backbone

model_dict = dict(ResNet10 = backbone.ResNet10)

def parse_args(script):
    parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
    parser.add_argument('--model'       , default='ResNet10',      help='backbone architecture')
    parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ')
    parser.add_argument('--freeze_backbone'   , action='store_true', help='Freeze the backbone network for finetuning')
    parser.add_argument('--use_saved', action='store_true', help='Use the saved resources')
    parser.add_argument('--lamda', default=0.001, type=float)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--k_lp', default=10, type=int)
    parser.add_argument('--delta', default=0.2, type=float)
    parser.add_argument('--alpha', default=0.5, type=float)

    if script == 'train':
        parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch')
    elif script == 'finetune':
        parser.add_argument('--dtarget', default='CropDisease', choices=['CropDisease', 'EuroSAT', 'ISIC', 'ChestX'])
        parser.add_argument('--test_n_way', default=5, type=int, help='class num to classify for testing')
        parser.add_argument('--n_shot', default=5, type=int,
                            help='number of labeled data in each class, same as n_support')
    else:
       raise ValueError('Unknown script')
        
    return parser.parse_args()
