import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from .calculation import calculate_homophily
from .general_utils import set_seeds, print_highlight, ModelAssitant, get_gpu_memory
from .data_load_module import data_load, _dataset_list, dataset_statistic, data_split

