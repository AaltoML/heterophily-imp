import torch
import os
import json
import random 
import numpy as np
import subprocess as sp

# *  =====  General settings  ===== 

def get_gpu_memory(str=None):
    '''
    return GB size 
    '''
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_total_values = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]

    return [f'f{free/1024:.2f}+u{(total-free)/1024:.2f}/{total/1024:.2f} GB' for free, total in zip(memory_free_values, memory_total_values) ]

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def print_highlight(info): 
    infos = '='*20+' '*5+ info +' '*5+'='*20
    print('='* len(infos))
    print(infos)
    print('='* len(infos))
    

class ModelAssitant():
    def __init__(self):
        self.best_val_acc = 0.
        self.best_val_loss = np.inf
        self.best_test_acc = 0.
        self.best_epoch = -1
        self.all_metrics = dict()
        
    def info_update(self, metrics, save_dir, epoch):
        update_best = False
        acc = metrics['acc/val']
        loss = metrics['loss/val']
        
        if (acc > self.best_val_acc) or ((acc == self.best_val_acc) and (loss < self.best_val_loss)):
            self.best_epoch = epoch
            self.best_val_acc = acc
            self.best_val_loss = loss
            self.best_test_acc = metrics['acc/test']
            update_best = True
        metrics['loss/best_val'] = self.best_val_loss
        metrics['acc/best_val'] = self.best_val_acc
        metrics['acc/best_test'] = self.best_test_acc
        metrics['best_epoch'] = self.best_epoch
        self.all_metrics[epoch] = metrics
        return update_best
            
    def save_best(self, model, metrics, save_dir):
        path = os.path.join(save_dir, 'best_model.pt')
        torch.save(model.state_dict(), path)
        
        with open(os.path.join(save_dir, f'best_metrics.txt'), "w") as f:
            json.dump(metrics, f) 
    def early_stop(self, epoch, eps = 1e-5):
        if epoch >100:
            avg_loss = [self.all_metrics[i]['loss/train'] for i in range(epoch-50, epoch)]
            if np.abs( np.mean(avg_loss[:10]) - np.mean(avg_loss[-10:]) ) < eps:
                return True
        return False
    def save_all_metrics(self, save_dir):
        with open(os.path.join(save_dir, f'all_metrics.txt'), "w") as f:
            json.dump(self.all_metrics, f) 
        
    # def merge_metrics_file(self, save_dir, max_epoch):
    #     import shutil
    #     metrics_all = dict()
    #     for epoch in range(max_epoch):
    #         try:
    #             filename = os.path.join(save_dir, 'metrics', f'metrics_epoch_{epoch}.txt')
    #             with open(filename, "r")  as f:
    #                 metrics_all[epoch] = json.load(f)
    #         except:
    #             print(f"save_dir {save_dir}, epoch {epoch} file not existed!")
    #     with open(os.path.join(save_dir, f'all_metrics.txt'), "w") as f:
    #         json.dump(metrics_all, f) 
    #     shutil.rmtree(os.path.join(save_dir, 'metrics'))
        
# class Results_collector():
#     def __init__(self):
#         self.meta_info = None
#         self.epoch_loss = {'train': [], 'val': []}
#         self.epoch_acc = {'train': [], 'val': [],  'test':[], 'best_test':[]}
#         self.final_acc = None
        
#     def meta_info_init(self, meta_info):
#         self.meta_info = meta_info
        
#     def epoch_update(self, train_loss, val_loss, train_acc, val_acc, test_acc, best_val_acc, best_test_acc):
#         for key, value in zip(['train', 'val'] [train_loss, val_loss]):
#             self.epoch_loss[key].append(value)
#         for key, value in zip(['train', 'val', 'test','best_val' 'best_test'], [train_acc, val_acc, test_acc,best_val_acc, best_test_acc]):
#             self.epoch_loss[key].append(value)
        
#         self.epoch_acc['train'].append(train_acc)
#         self.epoch_acc['val'].append(val_acc)
#         pass
    
# * ===== ===== ===== ===== ===== ===== =====