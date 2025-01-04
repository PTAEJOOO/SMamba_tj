from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred
from data_provider.imts_utils import tsdm_collate

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

def imts_data_provider(args):
    Data = imts_data_dict[args.data]
    TASK = Data(normalize_time=True, condition_time=args.seq_len, 
                forecast_horizon=args.pred_len,num_folds=args.nfolds)

    dloader_config_train = {
    "batch_size": args.batch_size,
    "shuffle": True,
    "drop_last": True,
    "pin_memory": True,
    "num_workers": 4,
    "collate_fn": tsdm_collate,
    }

    dloader_config_infer = {
        "batch_size": 32,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": True,
        "num_workers": 0,
        "collate_fn": tsdm_collate,
    }

    train_loader = TASK.get_dataloader((0, "train"), **dloader_config_train)
    vali_loader = TASK.get_dataloader((0, "valid"), **dloader_config_infer)
    test_loader = TASK.get_dataloader((0, "test"), **dloader_config_infer)
    
    return train_loader, vali_loader, test_loader
