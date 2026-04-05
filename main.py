from __future__ import print_function

import argparse
import pdb
import os
import math

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, Generic_MIL_Survival_Dataset, Generic_MIL_Cox_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np


def main(args):
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
        datasets = (train_dataset, val_dataset, test_dataset)
        results, test_auc, val_auc, test_acc, val_acc = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc': all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))


parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None)
parser.add_argument('--embed_dim', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--label_frac', type=float, default=1.0)
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--k_start', type=int, default=-1)
parser.add_argument('--k_end', type=int, default=-1)
parser.add_argument('--results_dir', default='./results')
parser.add_argument('--split_dir', type=str, default=None)
parser.add_argument('--log_data', action='store_true', default=False)
parser.add_argument('--testing', action='store_true', default=False)
parser.add_argument('--early_stopping', action='store_true', default=False)
parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', type=float, default=0.25)
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb')
parser.add_argument('--exp_code', type=str)
parser.add_argument('--weighted_sample', action='store_true', default=False)
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small')
parser.add_argument('--cox_batch_size', type=int, default=8)
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 'task_3_prog_vs_noprog', 'task_4_survival_binned_ce', 'task_5_survival_nll', 'task_6_survival_cox'])
parser.add_argument('--no_inst_cluster', action='store_true', default=False)
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None)
parser.add_argument('--subtyping', action='store_true', default=False)
parser.add_argument('--bag_weight', type=float, default=0.7)
parser.add_argument('--B', type=int, default=8)
parser.add_argument('--cohort_csv', type=str, default=None,
                    help='override task-specific CSV path')
parser.add_argument('--survival_csv', type=str, default='/home/jupyter/her2low_project/her2low_survival.csv')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

settings = {'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
    settings.update({'bag_weight': args.bag_weight,
                     'inst_loss': args.inst_loss,
                     'B': args.B})

print('\nLoad Dataset')

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/tumor_vs_normal_dummy_clean.csv',
                                  data_dir=os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                                  shuffle=False, seed=args.seed, print_info=True,
                                  label_dict={'normal_tissue': 0, 'tumor_tissue': 1},
                                  patient_strat=False, ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes = 3
    dataset = Generic_MIL_Dataset(csv_path='dataset_csv/tumor_subtyping_dummy_clean.csv',
                                  data_dir=os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                                  shuffle=False, seed=args.seed, print_info=True,
                                  label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
                                  patient_strat=False, ignore=[])

elif args.task == 'task_3_prog_vs_noprog':
    args.n_classes = 2
    _csv = args.cohort_csv or '/home/jupyter/her2low_project/her2low_task3.csv'
    dataset = Generic_MIL_Dataset(
        csv_path=_csv,
        data_dir=args.data_root_dir,
        shuffle=False, seed=args.seed, print_info=True,
        label_dict={'no_progression': 0, 'progression': 1},
        patient_strat=False, ignore=[])

elif args.task == 'task_4_survival_binned_ce':
    args.n_classes = 4
    _csv = args.cohort_csv or args.survival_csv
    dataset = Generic_MIL_Survival_Dataset(
        csv_path=_csv,
        data_dir=args.data_root_dir,
        shuffle=False, seed=args.seed, print_info=True,
        patient_strat=False)

elif args.task == 'task_5_survival_nll':
    args.n_classes = 4
    _csv = args.cohort_csv or args.survival_csv
    dataset = Generic_MIL_Survival_Dataset(
        csv_path=_csv,
        data_dir=args.data_root_dir,
        shuffle=False, seed=args.seed, print_info=True,
        patient_strat=False)

elif args.task == 'task_6_survival_cox':
    args.n_classes = 1
    _csv = args.cohort_csv or args.survival_csv
    dataset = Generic_MIL_Cox_Dataset(
        csv_path=_csv,
        data_dir=args.data_root_dir,
        shuffle=False, seed=args.seed, print_info=True)
    
else:
    raise NotImplementedError

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# ── Fix split_dir: don't prepend 'splits/' if an absolute path is given ───────
if args.split_dir is None:
    args.split_dir = os.path.join('splits', args.task + '_{}'.format(int(args.label_frac * 100)))
elif os.path.isabs(args.split_dir):
    pass  # use as-is
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")
