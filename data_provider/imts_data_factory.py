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
            "batch_size": 8128,
            "shuffle": True,
            "drop_last": True,
            "pin_memory": True,
            "num_workers": 1,
            "collate_fn": tsdm_collate,
        }
    elif flag == 'val':
        dloader_config = {
            "batch_size": 2037,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "num_workers": 0,
            "collate_fn": tsdm_collate,
        }
    else:
        dloader_config = {
            "batch_size": 1798,
            "shuffle": False,
            "drop_last": False,
            "pin_memory": True,
            "num_workers": 0,
            "collate_fn": tsdm_collate,
        }

    data_loader = TASK.get_dataloader((0, "train"), **dloader_config)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for _, batch in enumerate(data_loader):
        x_time, x_vals, x_mask, y_time, y_vals, y_mask = (tensor.to(DEVICE) for tensor in batch)

        x_time = x_time[:,:args.seq_len].unsqueeze(-1)
        x_vals = x_vals[:,:args.seq_len,:]
        x_mask = x_mask[:,:args.seq_len,:]

        y_time = y_time[:,args.seq_len:].unsqueeze(-1)
        y_vals = y_vals[:,args.seq_len:,:]
        y_mask = y_mask[:,args.seq_len:,:]

        new_dataset = CustomDataset(x_time,x_vals,x_mask,y_time,y_vals,y_mask)
        new_data_loader = DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True)

    # for _, batch in enumerate(vali_loader):
    #     x_time, x_vals, x_mask, y_time, y_vals, y_mask = (tensor.to(DEVICE) for tensor in batch)

    #     x_time = x_time[:,:args.seq_len].unsqueeze(-1)
    #     x_vals = x_vals[:,:args.seq_len,:]
    #     x_mask = x_mask[:,:args.seq_len,:]

    #     y_time = y_time[:,args.seq_len:].unsqueeze(-1)
    #     y_vals = y_vals[:,args.seq_len:,:]
    #     y_mask = y_mask[:,args.seq_len:,:]

    #     new_dataset = CustomDataset(x_time,x_vals,x_mask,y_time,y_vals,y_mask)
    #     new_vali_loader = DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True)

    # for _, batch in enumerate(test_loader):
    #     x_time, x_vals, x_mask, y_time, y_vals, y_mask = (tensor.to(DEVICE) for tensor in batch)

    #     x_time = x_time[:,:args.seq_len].unsqueeze(-1)
    #     x_vals = x_vals[:,:args.seq_len,:]
    #     x_mask = x_mask[:,:args.seq_len,:]

    #     y_time = y_time[:,args.seq_len:].unsqueeze(-1)
    #     y_vals = y_vals[:,args.seq_len:,:]
    #     y_mask = y_mask[:,args.seq_len:,:]

    #     new_dataset = CustomDataset(x_time,x_vals,x_mask,y_time,y_vals,y_mask)
    #     new_test_loader = DataLoader(new_dataset, batch_size=args.batch_size, shuffle=True)

    return new_data_loader
