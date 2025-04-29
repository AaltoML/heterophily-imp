import json
import os
import sys
# for linux env.
sys.path.insert(0,'..')
import argparse
from distutils.util import strtobool
import torch
import torch.nn as nn

from data.data_loader import NumpyTupleDataset
from hetflow.models.hyperparams import Hyperparameters
from hetflow.models.model import HetFlow, rescale_adj
from hetflow.models.utils import check_validity, save_mol_png, generate_smiles_from_data, evaluation_smiles_for_debug, molecules_draw_to_path
from evaluation.evaluator import Evaluator_during_training, Evaluator_saved_model

from hetflow.utils.model_utils import set_up_seed
import time
from hetflow.utils.timereport import TimeReport
from hetflow.generate import generate_mols

import functools
print = functools.partial(print, flush=True)
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')     

from evaluation.utils import ptest_precalculation, dataset_loading


import random
import numpy as np




import wandb
def get_parser():
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str, default='../data', help='Location for the dataset')
    parser.add_argument('--data_name', type=str, default='qm9', choices=['qm9', 'zinc250k'], help='dataset name')
    # parser.add_argument('-f', '--data_file', type=str, default='qm9_relgcn_kekulized_ggnp.npz', help='Name of the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='results/qm9',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-t', '--ite_log_step', type=int, default=20,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=int, default=0,
                        help='Restore training from previous model checkpoint? 1 = Yes, 0 = No')
    parser.add_argument('--load_snapshot', type=str, default='', help='load the model from this path')
    # optimization
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU Id to use')
    parser.add_argument('--save_epochs', type=int, default=1, help='in how many epochs, a snapshot of the model'
                                                                   ' needs to be saved?')
    # data loader
    parser.add_argument('-b', '--batch_size', type=int, default=12, help='Batch size during training per GPU')
    parser.add_argument('--shuffle', type=strtobool, default='false', help='Shuffle the data batch')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers in the data loader')

    # # evaluation
    # parser.add_argument('--sample_batch_size', type=int, default=16,
    #                     help='How many samples to process in paralell during sampling?')
    # reproducibility
    # For bonds
    parser.add_argument('--b_n_flow', type=int, default=10,
                        help='Number of masked glow coupling layers per block for bond tensor')
    parser.add_argument('--b_n_block', type=int, default=1, help='Number of glow blocks for bond tensor')
    parser.add_argument('--b_hidden_ch', type=str, default="128,128",
                        help='Hidden channel list for bonds tensor, delimited list input ')
    parser.add_argument('--b_conv_lu', type=int, default=1, choices=[0, 1, 2],
                        help='0: InvConv2d for 1*1 conv, 1:InvConv2dLU for 1*1 conv, 2: No 1*1 conv, '
                             'swap updating in the coupling layer')
    # For atoms
    parser.add_argument('--a_n_flow', type=int, default=27,
                        help='Number of masked flow coupling layers per block for atom matrix')
    parser.add_argument('--a_n_block', type=int, default=1, help='Number of flow blocks for atom matrix')
    parser.add_argument('--a_hidden_gnn', type=str, default="64,",
                        help='Hidden dimension list for graph convolution for atoms matrix, delimited list input ')
    parser.add_argument('--a_hidden_lin', type=str, default="128,64",
                        help='Hidden dimension list for linear transformation for atoms, delimited list input ')
    parser.add_argument('--mask_row_size_list', type=str, default="1,",
                        help='Mask row size list for atom matrix, delimited list input ')
    parser.add_argument('--mask_row_stride_list', type=str, default="1,",
                        help='Mask row stride list for atom matrix, delimited list input')
    # General
    parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
    parser.add_argument('--debug', type=strtobool, default='true', help='To run training with more information')
    parser.add_argument('--debug_eva', type=strtobool, default='false', help='To run training with more information')
    parser.add_argument('--learn_dist', type=int, default='1', help='learn the distribution of feature matrix')
    parser.add_argument('--noise_scale', type=float, default=0.6, help='x + torch.rand(x.shape) * noise_scale')
    
    
    parser.add_argument('--log_info', type=str, default='default', help='log information for tesorboard')
    parser.add_argument('--wandb_record', type=strtobool, default='false', help='If the wandb used for the training')
    parser.add_argument('--generation_size', type=int, default=1000, help='batch size of molecule generation test for every epoches')
    parser.add_argument('--project_name', type=str, default="Heterophily-flow", help='Name for wandb project')
    parser.add_argument('--record_all_metrics', type=strtobool, default='false', help='If the wandb used for the training')
    parser.add_argument('--true_adj', type=strtobool, default='false', help='whether evaluate the model with true adjacency matrix')
    parser.add_argument('--graph_conv_option', type=str, default='HetFlow11', help='whether evaluate the model with true adjacency matrix')
    
    return parser

def load_NumpyTupleDataset(data_name, data_dir):
    if data_name == 'qm9':
        from data import transform_qm9
        data_file = 'qm9_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_qm9.transform_fn
        atomic_num_list = [6, 7, 8, 9, 0]

    elif data_name == 'zinc250k':
        from data import transform_zinc250k
        data_file = 'zinc250k_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_zinc250k.transform_fn_zinc250k
        atomic_num_list = transform_zinc250k.zinc250_atomic_num_list  # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        # mlp_channels = [1024, 512]
        # gnn_channels = {'gcn': [16, 128], 'hidden': [256, 64]}
    else:
        raise ValueError('Only support qm9 and zinc250k right now. '
                            'Parameters need change a little bit for other dataset.')
    dataset = NumpyTupleDataset.load(os.path.join(data_dir, data_file), transform=transform_fn)
    return dataset


def sample_true_adj(dataset, sample_size):
    index = random.sample(range(len(dataset)), sample_size)
    adj_list = np.array([dataset[i][1] for i in index])
    adj = torch.tensor(adj_list)
    return adj

# def sample_adj(dataset, batch_size):
#     index = random.sample(range(len(dataset)), batch_size)
#     adj_list = [dataset[i][1] for i in index]
#     adj = torch.tensor(adj_list)
#     return adj

def func_init_wandb_config(args):
    keys_uploaded = ['seed', 'data_name', 'save_dir', 'batch_size', 'noise_scale', 'generation_size', 'learning_rate']
    wandb_init_config = {  key:getattr(args, key) for key in keys_uploaded}
    return wandb_init_config

def best_model_save(metrics, best_fcd, best_epoch, epoch, model, save_dir, if_true_adj=False):
    key = 'true_adj' if if_true_adj else 'gen_adj'
    if 'TEST_lb/FCD' not in metrics.keys():
        return 0
    fcd = metrics['TEST_lb/FCD']
    if fcd < best_fcd[key]:
        print(f'*** Best model (fcd based) *** {key} updated!')
        best_epoch[key] = epoch
        best_fcd[key] = fcd
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_snapshot_best_fcd_'+key))


def train():
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))
    parser = get_parser()
    args = parser.parse_args()
    set_up_seed(args.seed)
    # Device configuration
    device = -1
    multigpu = False
    if args.gpu >= 0:
        device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    elif args.gpu == -1:
        # cpu
        device = torch.device('cpu')
    else:
        # multigpu, can be slower than using just 1 gpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        multigpu = True

    debug = args.debug
    
    path_metric = os.path.join(args.save_dir, 'metrics')
    os.makedirs(path_metric, exist_ok=True)
    
    wandb_record = args.wandb_record
    log_info = ''+ args.log_info
    writer = None
    if wandb_record:
        wandb.login()
        wandb.init(
            project=args.project_name,
            name = args.log_info,
            config = func_init_wandb_config(args)
        )
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

    # Model configuration
    b_hidden_ch = [int(d) for d in args.b_hidden_ch.strip(',').split(',')]
    a_hidden_gnn = [int(d) for d in args.a_hidden_gnn.strip(',').split(',')]
    a_hidden_lin = [int(d) for d in args.a_hidden_lin.strip(',').split(',')]
    mask_row_size_list = [int(d) for d in args.mask_row_size_list.strip(',').split(',')]
    mask_row_stride_list = [int(d) for d in args.mask_row_stride_list.strip(',').split(',')]
    if args.data_name == 'qm9':
        from data import transform_qm9
        atomic_num_list = [6, 7, 8, 9, 0]
        b_n_type = 4
        b_n_squeeze = 3
        a_n_node = 9
        a_n_type = len(atomic_num_list)  # 5
        valid_idx = transform_qm9.get_val_ids((args.data_dir)  # len: 13,082, total data: 133,885
    elif args.data_name == 'zinc250k':
        from data import transform_zinc250k
        atomic_num_list = transform_zinc250k.zinc250_atomic_num_list  # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        b_n_type = 4
        b_n_squeeze = 19   # 2
        a_n_node = 38
        a_n_type = len(atomic_num_list)  # 10
        valid_idx = transform_zinc250k.get_val_ids((args.data_dir)
    else:
        raise ValueError('Only support qm9 and zinc250k right now. '
                         'Parameters need change a little bit for other dataset.')
    
    graph_conv_option = args.graph_conv_option
    # if graph_conv_option == 'None':
    #     graph_conv_option = None
    model_params = Hyperparameters(b_n_type=b_n_type,  # 4,
                                   b_n_flow=args.b_n_flow,
                                   b_n_block=args.b_n_block,
                                   b_n_squeeze=b_n_squeeze,
                                   b_hidden_ch=b_hidden_ch,
                                   b_affine=True,
                                   b_conv_lu=args.b_conv_lu,
                                   a_n_node=a_n_node,
                                   a_n_type=a_n_type,
                                   a_hidden_gnn=a_hidden_gnn,
                                   a_hidden_lin=a_hidden_lin,
                                   a_n_flow=args.a_n_flow,
                                   a_n_block=args.a_n_block,
                                   mask_row_size_list=mask_row_size_list,
                                   mask_row_stride_list=mask_row_stride_list,
                                   a_affine=True,
                                   learn_dist=args.learn_dist,
                                   seed=args.seed,
                                   noise_scale=args.noise_scale,
                                   graph_conv_option=graph_conv_option
                                   )
    print('Model params:')
    model_params.print()
    model = HetFlow(model_params)
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_hyperparams(os.path.join(args.save_dir, 'moflow-params.json'))
    if torch.cuda.device_count() > 1 and multigpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    else:
        multigpu = False
    model = model.to(device)

    # Datasets:
    dataset = load_NumpyTupleDataset(args.data_name, args.data_dir)
    # NumpyTupleDataset.load(os.path.join(args.data_dir, data_file), transform=transform_fn)  # 133885
    
    if len(valid_idx) > 0:
        train_idx = [t for t in range(len(dataset)) if t not in valid_idx]  # 120803 = 133885-13082
        # n_train = len(train_idx)  # 120803
        train = torch.utils.data.Subset(dataset, train_idx)  # 120,803
        test = torch.utils.data.Subset(dataset, valid_idx)  # 13,082
    else:
        torch.manual_seed(args.seed)
        train, test = torch.utils.data.random_split(
            dataset,
            [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    train_smiles, _, _ = dataset_loading(args.data_name, data_dir = args.data_dir)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                                   shuffle=args.shuffle, num_workers=args.num_workers)

    print('==========================================')
    print('Load data done! Time {:.2f} seconds'.format(time.time() - start))
    print('Data shuffle: {}, Number of data loader workers: {}!'.format(args.shuffle, args.num_workers))
    if args.gpu >= 0:
        print('Using GPU device:{}!'.format(args.gpu))
    print('Num Train-size: {}'.format(len(train)))
    print('Num Minibatch-size: {}'.format(args.batch_size))
    print('Num Iter/Epoch: {}'.format(len(train_dataloader)))
    print('Num epoch: {}'.format(args.max_epochs))
    print('==========================================')


    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    evaluator = Evaluator_during_training(device = device, batch_size = args.generation_size, data_dir = args.data_dir )
    evaluator.load_dataset(args.data_name, train_smiles)
    
    # Train the models
    iter_per_epoch = len(train_dataloader)
    log_step = args.ite_log_step  # 20 default
    tr = TimeReport(total_iter=args.max_epochs * iter_per_epoch)

    best_fcd = {'gen_adj': 1000, 'true_adj': 1000}
    best_epoch =  {'gen_adj': -1, 'true_adj': -1}
    for epoch in range(1,args.max_epochs+1):
        print("In epoch {}, Time: {}".format(epoch, time.ctime()))
        loss_total = torch.tensor([0., 0.])
        model.train()
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            # turn off shuffle to see the order with original code
            x = batch[0].to(device)  # (256,9,5)
            adj = batch[1].to(device)  # (256,4,9, 9)
            adj_normalized = rescale_adj(adj).to(device)

            # Forward, backward and optimize
            z, sum_log_det_jacs = model(adj, x, adj_normalized)
            if multigpu:
                nll = model.module.log_prob(z, sum_log_det_jacs)
            else:
                nll = model.log_prob(z, sum_log_det_jacs)
            loss = nll[0] + nll[1]
            loss.backward()
            optimizer.step()
            tr.update()

            loss_total += torch.tensor(nll).detach()

            # Print log info
            if (i+1) % log_step == 0:  # i % args.log_step == 0:
                print('Epoch [{}/{}], Iter [{}/{}], loglik: {:.5f}, nll_x: {:.5f},'
                      ' nll_adj: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: '.
                      format(epoch, args.max_epochs, i+1, iter_per_epoch,
                             loss.item(), nll[0].item(), nll[1].item(),
                             tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
                tr.print_summary()
            # if i > 3 and args.debug_eva: break
        loss_total = loss_total/len(train_dataloader)
        
         #  ================================  ================================
        info_train = {"TRAIN/loss_x": loss_total[0].item(),
                          "TRAIN/loss_adj": loss_total[1].item(),
                          "TRAIN/loss": (loss_total[0]+ loss_total[1]).item(),
        }
        if len(model.ln_var) == 1:
            info_train["TRAIN/ln_var"] = model.ln_var.item()
        elif len(model.ln_var) == 2:
            info_train["TRAIN/ln_var_adj"] = model.ln_var[0].item()
            info_train["TRAIN/ln_var_x"] = model.ln_var[1].item()
        print(f'Epoch {epoch} \n training info: {info_train}')
        
        if wandb_record:
            wandb.log(data = info_train, step = epoch)
        
        # breakpoint() 
       
        if debug:
            model.eval()
            metrics = evaluator.model_evaluation_during_training(model, args.save_dir, epoch, wandb_option = args.wandb_record, debug= args.debug_eva)
            print(metrics)
            best_model_save(metrics, best_fcd, best_epoch, epoch, model, args.save_dir, if_true_adj=False)
            
            if args.true_adj:
                true_adj = sample_true_adj(dataset, args.generation_size)
                metrics = evaluator.model_evaluation_during_training(model, args.save_dir, epoch, wandb_option = args.wandb_record, true_adj = true_adj, debug= args.debug_eva)
                print('[Metrics with real adj]:\n',metrics)
                best_model_save(metrics, best_fcd, best_epoch, epoch, model, args.save_dir, if_true_adj=True)
         #  ================================  ================================
        # The same report for each epoch
        print('Epoch [{}/{}], Iter [{}/{}], loglik: {:.5f}, nll_x: {:.5f},'
              ' nll_adj: {:.5f}, {:.2f} sec/iter, {:.2f} iters/sec: '.
              format(epoch, args.max_epochs, -1, iter_per_epoch,
                     loss.item(), nll[0].item(), nll[1].item(),
                     tr.get_avg_time_per_iter(), tr.get_avg_iter_per_sec()))
        tr.print_summary()

        # Save the model checkpoints
        save_epochs = args.save_epochs
        if save_epochs == -1:
            save_epochs = args.max_epochs
        if (epoch) % save_epochs == 0:
            if multigpu:
                torch.save(model.module.state_dict(), os.path.join(
                args.save_dir, 'model_snapshot_epoch_{}'.format(epoch)))
            else:
                torch.save(model.state_dict(), os.path.join(
                args.save_dir, 'model_snapshot_epoch_{}'.format(epoch)))
            tr.end()
    print( *[f"Best {key} model: {best_epoch[key]} epoch, fcd: {best_fcd[key]}\n" for key in  ['gen_adj', 'true_adj']])
    print("[Training Ends], Start at {}, End at {}".format(time.ctime(start), time.ctime()))

    
if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    train()
