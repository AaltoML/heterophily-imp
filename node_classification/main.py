'''
The main training process
'''

import os, sys, time
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # without it, there is bug in `torch.use_deterministic_algorithms(True)`
from tqdm.auto import tqdm # For progress bar
import torch
# from torchsummary import summary
from torch_geometric.nn import summary
from distutils.util import strtobool
import argparse

from models import model_initialization, _all_modes, _model_types
from utils import set_seeds, data_load, data_split, print_highlight, ModelAssitant, _dataset_list, get_gpu_memory
from sklearn.metrics import roc_auc_score

_log_list = ['data_name', 'model_name', 'hidden_channels', 'num_layers', 'het_mode', 'seed', 'max_epochs', 'learning_rate', 'slurm_info']

def get_parser():
    parser = argparse.ArgumentParser()
    ### Basic paras
    # flexible paras
    parser.add_argument('-d', '--data_name', type=str, choices=_dataset_list, default='Cornell', help='Name of the dataset')
    parser.add_argument('--save_dir', type=str, default='results/default', help='Name of the dataset')
    parser.add_argument('--model_name', type=str, default='GCN', help='Name of the model_types', choices = _model_types)
    parser.add_argument('--het_mode', type=str, choices=_all_modes+[None], default='original', help='heterophily mode')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Dimensions of hidden channels')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of GNN layers')
    # parser.add_argument('--out_channels', type=int, default=???, help='heterophily mode')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout ratio during training')
    parser.add_argument('--norm_mode', type=str, default=None, help='normalization mode')
    # Mostly fixed
    parser.add_argument('-i', '--data_dir', type=str, default='data/', help='Location for the dataset')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed to use')
    parser.add_argument('--seed_split', type=int, default=0, help='Random seed for data split')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('-x', '--max_epochs', type=int, default=1000, help='How many epochs to run in total?')
    parser.add_argument('--min_lr', type=float, default= 1e-5, help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', type=float, default= 1e-6, help="Please give a value for weight_decay")
    parser.add_argument('--debug', type=strtobool, default='false', help="help to debug")
    
    
    ### Parameters for assited tools
    # Wandb
    parser.add_argument('--record_wandb', type=strtobool, default='false', help='If wandb module used for the training')
    # parser.add_argument('--wandb_api', type=str, default=None, help='dir for recording wandb run')
    parser.add_argument('--wandb_mode', type=str, default=None, choices=[None, 'offline', 'online'], help='dir for recording wandb run')
    parser.add_argument('--run_name', type=str, default='default', help='Name of current wandb run')
    parser.add_argument('--project_name', type=str, default="ss-gnns", help='Name of the wandb project including current run') # 'SS' means Similarity Scaling, 
    # job info
    parser.add_argument('--slurm_info', type=str, default=None, help='information of slurm job and array id')
    # print info
    parser.add_argument('--miniters_tqdm', type=int, default=5, help='the minimum update iteration for tqdm module')
    return parser


def wandb_init(args = None):
    '''
    Log all the training info on the wandb module
    '''
    if args.record_wandb:
        # os.environ["WANDB_DIR"] = args.save_dir
        mode = args.wandb_mode
        
        if mode is  None:
            os.environ["WANDB_MODE"] = 'online'
        else:
            os.environ["WANDB_MODE"] = args.wandb_mode
        wandb.login()
        info_dict = vars(args)
        run = wandb.init(
            project=args.project_name,
            name = args.run_name,
            config = info_dict
        )
        print('wandb info logged!')
    else:
        print('wandb module not required...')
    return run
    
def device_setting():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device 

def accuracy(out, labels, mask):
    n_class = out.shape[1]
    pred= out.argmax(dim = -1)
    if n_class >2:
        correctness = pred[mask] == labels[mask]
        accuracy = int(correctness.sum())/ int(mask.sum())
    elif n_class == 2:
        accuracy = roc_auc_score(y_true = labels[mask].to('cpu'), y_score = pred[mask].to('cpu') )
    return accuracy

def train_epoch(model, graphs, masks, device, optimizer, criterion):
    epoch_loss = 0
    epoch_train_acc = 0
    model.to(device)
    model.train()
    for graph, mask in zip(graphs, masks['train']):
        graph = graph.to(device)
        
        x, edge_index, labels = graph.x, graph.adj_t, graph.y
        out = model(x, edge_index)
        get_gpu_memory(f'model forward after training')
        loss = criterion(out[mask], labels[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(out.detach(), labels, mask) 
    return epoch_loss, epoch_train_acc

def model_eval(model, graphs, masks, device, criterion):
    results = dict()
    model.to(device)
    model.eval()
    for key in ['test', 'val']:
        results[f'loss/{key}'] =  0.
        results[f'acc/{key}'] = 0.
    with torch.no_grad():
        for i, graph in enumerate(graphs):
            # loss, acc = 0., 0.
            graph.to(device) 
            x, edge_index, labels = graph.x, graph.adj_t, graph.y
            out =  model(x, edge_index)
            get_gpu_memory(f'model forward after eval')
            mask = None
            for key in ['test', 'val']:
                mask = masks[key][i]
                loss = criterion(out[mask], labels[mask])
                acc = accuracy(out.detach(), labels, mask)
                results[f'loss/{key}'] += loss.detach().item()
                results[f'acc/{key}'] += acc

    return results

def print_training_issues(args, dict_add):
    '''print the arguments info, also allow showing additional info buy `dict_add` '''
    # info_dict = {k: getattr(args,k) for k in _log_list}
    info_dict = vars(args)
    info_dict.update(dict_add)
    for k in info_dict.keys():
        print(f'{k}: {info_dict.get(k)}')
        

from utils import data_load, set_seeds
if __name__ == "__main__":
    
    # * 0.1 Hyperparameters
    # hidden_channels, out_channels, num_layers = 2, 3, 4
    # het_mode = 'homophily'
    # model_name = 'GCN'
    
    # * 1. Preparations
    # * 1.0 Parameter loading from arguments
    # device, wandb log and initialize, 
    print('Preparing basic initialization...')
    args = get_parser().parse_args()
    set_seeds(args.seed) # set random seeds for all packages
    seed_split = args.seed_split
    device = device_setting()
    save_dir = args.save_dir
    data_name = args.data_name
    
    os.makedirs(save_dir, exist_ok=True)
    if args.record_wandb:
        import wandb
        wandb_init(args) # initialize wanbd if needed
    
    # * 1.1 Data loading
    print('Data loading...')
    dataset = data_load(data_name) # For Planetoid or webKB graphs
    in_channels = dataset.num_node_features
    out_channels = dataset.num_classes
    graphs = dataset
    
    masks = data_split(args.seed, graphs, data_name)
    # num_train_val = 5000 if data_name == "CoauthorCS" else 1500
    # masks = set_train_val_test_split(seed_split, graphs, data_name, num_development=num_train_val)
    
    
    
    
    mask_size = {k: [torch.sum(m).item() for m in masks[k]] for k in ['train','val', 'test']  }
    
    # * 1.2 Model, optimizer initialization
    print('Model initialization...')
    model = model_initialization(args.model_name, in_channels, args.hidden_channels,
                                 out_channels, args.num_layers, args.het_mode, args.dropout,
                                 args.norm_mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    model_assistant = ModelAssitant()
    
    # * 1.3 Training issues, model summary print
    print_highlight('Training Params')
    _dict_add = {'device': device, 'in_channels': in_channels, 'out_channels':out_channels,
                 'graph_size': [g.y.shape[0] for g in graphs],
                 'train size': mask_size['train'], 'val size': mask_size['val'],
                 'test size': mask_size['test']}
    print_training_issues(args, _dict_add)
    if args.record_wandb:
        wandb.config.update(_dict_add)
    
    # print_training_issues(args, key_list = _arg_list_for_logs)
    
    print_highlight('Model Summary')
    print(summary(model, graphs[0].x, graphs[0].adj_t))
    # print(summary(model, graphs[0].x, graphs[0].edge_index))
    
    # breakpoint()
    # * 2. Training, while evaluation and recording in the cloud (wandb)
    print_highlight('Model training')
    model.to(device)
    with tqdm(range(args.max_epochs), miniters=args.miniters_tqdm, maxinterval=300) as pbar:
        for epoch in pbar:
            
            metrics = {'epoch': epoch}# the dictionary keys: train_loss, train_acc. Optional keys: val_loss, val_acc and test
            # * 2.1 train model
            train_loss, train_acc = train_epoch(model, graphs, masks, device, optimizer, criterion)
            metrics['loss/train'], metrics['acc/train'] = train_loss, train_acc
            
            # * 2.2 test and eval model
            eval_results = model_eval(model, graphs, masks, device, criterion)
            metrics.update(eval_results)
            
            # * 2.2.1 if current model is the best, if so, store the best model and corresponding metris
            update_best = model_assistant.info_update(metrics, save_dir, epoch)
            if update_best:
                model_assistant.save_best(model, metrics, save_dir)
            
            # * 2.2.2 if allow wandb recording, upload the metrics to wandb cloud
            if args.record_wandb :
                wandb.log(data = metrics, step = epoch)
 
            if epoch % args.miniters_tqdm == 0 :
                pbar.set_description('Epoch %d' % epoch)
                loss_val = metrics['loss/val']
                
                best_loss_val= metrics['loss/best_val']
                best_acc_val = metrics['acc/best_val']
                pbar.set_postfix(lr=optimizer.param_groups[0]['lr'],
                                    train_loss=metrics['loss/train'], 
                                    train_acc=metrics['acc/train'], val_acc=metrics['acc/val'],
                                    test_acc=metrics['acc/test'], best_test_acc=metrics['acc/best_test'],val_loss=f'{loss_val:.4f}', best_val_acc=f'{best_acc_val:.4f}',  best_val_loss=f'{best_loss_val:.4f}', mem=get_gpu_memory()
                                    )
            
            # * 2.3 check if the loss converg
            early_stop = model_assistant.early_stop(epoch)
            if early_stop: 
                break
        model_assistant.save_all_metrics(save_dir)
    if args.record_wandb:
        wandb.finish()
    print('[Training finished!]')    