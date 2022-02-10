__author__ = 'Connor Heaton and Saptarshi Sengupta'


import os
import argparse
import datetime

from runners import PubmedQARunner
import torch.multiprocessing as mp


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'none':
        return None
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data',
                        default='/home/czh/nvme1/pubmedqa/data',
                        # default='data/200423_covidQA.json',
                        help='Filepath to CovidQA dataset')
    parser.add_argument('--out', default='out', help='Directory to put output')

    # parser.add_argument('--n_splits', default=5, help='How many folds to use for cross val', type=int)
    parser.add_argument('--batch_size', default=32, help='How many items to process as once', type=int)
    parser.add_argument('--lr', default=5e-4, help='How many items to process as once', type=float)
    parser.add_argument('--l2', default=0.001, help='How many items to process as once', type=float)
    parser.add_argument('--n_epochs', default=10, help='If training/fine-tuning, how many epochs to perform', type=int)
    parser.add_argument('--n_stride', default=196, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--model_name',
                        # default='ktrapeznikov/scibert_scivocab_uncased_squad_v2',
                        default='clagator/biobert_squad2_cased',
                        # default='navteca/roberta-base-squad2',
                        # default='phiyodr/bert-base-finetuned-squad2',
                        # default='ktrapeznikov/biobert_v1.1_pubmed_squad_v2',
                        # default='ktrapeznikov/scibert_scivocab_uncased_squad_v2',
                        help='Type of model to use from HuggingFace')

    parser.add_argument('--use_kge', default=False, help='If KGEs should be place in input',
                        type=str2bool)
    parser.add_argument('--use_dict', default=False, help='If KGEs should be place in input',
                        type=str2bool)
    parser.add_argument('--concat_kge', default=False, type=str2bool)
    parser.add_argument('--fancy_concat', default=False, type=str2bool)
    parser.add_argument('--random_kge', default=False, type=str2bool)
    parser.add_argument('--seed', default=16, type=int)
    parser.add_argument('--n_warmup_iters', default=-1, help='Fuck Timo Moller', type=int)
    # parser.add_argument('--vanilla_adam', default=False, type=str2bool)

    parser.add_argument('--dte_lookup_table_fp',
                        default='NN-DTE-to-phiyodr-bert-base-finetuned-squad2.pkl'
                        # default='DTE_to_phiyodr_bert-base-finetuned-squad2.pkl',
                        # default='DTE_to_ktrapeznikov_biobert_v1.1_pubmed_squad_v2.pkl',
                        # default='DTE_to_ktrapeznikov_scibert_scivocab_uncased_squad_v2.pkl'
                        )
    parser.add_argument('--n_neg_records', default=1, type=int)

    parser.add_argument('--on_cpu', default=False, type=str2bool)
    parser.add_argument('--gpus', default=[0], help='Which GPUs to use', type=int, nargs='+')
    parser.add_argument('--port', default='14345', help='Port to use for DDP')
    parser.add_argument('--n_data_workers', default=2, help='# threads used to fetch data *PER DEVICE/GPU*', type=int)


    parser.add_argument('--grad_summary', default=True, type=str2bool)
    parser.add_argument('--grad_summary_every', default=75, type=int)
    parser.add_argument('--save_model_every', default=1, type=int)
    parser.add_argument('--print_every', default=1, type=int)
    parser.add_argument('--log_every', default=25, type=int)
    parser.add_argument('--summary_every', default=15, type=int)
    parser.add_argument('--n_grad_accum', default=1, type=int)
    parser.add_argument('--ckpt_file_tmplt', default='fold{}_model_{}e.pt')

    args = parser.parse_args()
    args.world_size = len(args.gpus)
    mp.set_sharing_strategy('file_system')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.out = os.path.join(args.out, curr_time)
    os.makedirs(args.out)

    args.tb_dir = os.path.join(args.out, 'tb_dir')
    os.makedirs(args.tb_dir)

    args.model_save_dir = os.path.join(args.out, 'models')
    os.makedirs(args.model_save_dir)

    args.model_log_dir = os.path.join(args.out, 'logs')
    os.makedirs(args.model_log_dir)

    fold_dirs = [os.path.join(args.data, dir_) for dir_ in os.listdir(args.data) if dir_.startswith('pqal_fold')]
    fold_dirs = list(sorted(fold_dirs, key=lambda x: int(x[-1])))

    print('Found {} fold dirs: {}'.format(len(fold_dirs), ', '.join(fold_dirs)))

    for fold_dir in fold_dirs:
        print('Creating {} distributed models for fold {}...'.format(len(args.gpus), fold_dir[-1]))
        mp.spawn(PubmedQARunner, nprocs=len(args.gpus), args=(fold_dir, args))
        break

    print('all done :)')







