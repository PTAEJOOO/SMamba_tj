from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred
from data_provider.imts_utils import tsdm_collate
from data_provider.imts_utils import CustomDataset

import torch
from torch.utils.data import DataLoader

from tsdm.tasks import USHCN_DeBrouwer2019
from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
from tsdm.tasks.physionet2012 import Physionet2012

##
import lib.utils as utils
from lib.physionet import *

imts_data_dict = {
    'ushcn': USHCN_DeBrouwer2019,
    'mimiciii': MIMIC_III_DeBrouwer2019,
    'mimiciv': MIMIC_IV_Bilos2021,
    'physionet2012': Physionet2012
}

def imts_data_provider(args, flag):
    Data = imts_data_dict[args.data]
    TASK = Data(normalize_time=True, condition_time=args.seq_len, 
                forecast_horizon=args.pred_len,num_folds=args.nfolds)

    if flag == 'train':
        dloader_config = {
            "batch_size": args.batch_size,
            "shuffle": True,
            "drop_last": True,
            "pin_memory": True,
            "num_workers": 1,
            "collate_fn": patch_variable_time_collate_fn,
        }
    elif flag == 'val':
        dloader_config = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "num_workers": 0,
            "collate_fn": patch_variable_time_collate_fn,
        }
    else:
        dloader_config = {
            "batch_size": args.batch_size,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "num_workers": 0,
            "collate_fn": patch_variable_time_collate_fn,
        }

    data_loader = TASK.get_dataloader((0, "train"), **dloader_config)

    return data_loader
