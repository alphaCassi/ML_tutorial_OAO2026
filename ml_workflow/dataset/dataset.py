from torch.utils.data import Dataset
import os
import glob
from omegaconf import DictConfig
import torch

# import functions from dataset_utils
from dataset.dataset_utils import load_atm_params, load_rms_commands, load_psfee_int, load_psffwhm_int, load_sr_int

#from dataset import AOdataset
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler




class AOdataset(Dataset):

    def __init__(self,
                 cfg: DictConfig,
                 normalizer = None):

        self.cfg = cfg
        self.root = cfg.dataset.root_data
        self.paths = sorted(glob.glob(os.path.join(self.root, 
                                        #  "gpu*", 
                                         "sim_*", 
                                         "*")))
        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        # X = torch.tensor(load_rms_commands(root=self.paths[idx]), dtype = torch.float32)
        X1 = torch.tensor(load_sr_int(root=self.paths[idx]), dtype = torch.float32)
        X2 = torch.tensor(load_psfee_int(root=self.paths[idx]), dtype = torch.float32)
        X3 = torch.tensor(load_psffwhm_int(root=self.paths[idx]), dtype = torch.float32)
        y = torch.tensor(load_atm_params(root=self.paths[idx]), dtype = torch.float32)

        X = torch.cat((X1, X2, X3))

        if self.normalizer is not None:
            

            #X = self.normalizer.transform_input(X)
            y = self.normalizer.transform_output(y)

     
        sample = X, y

        return sample
    


def make_datasets(cfg):
    dataset = AOdataset(cfg)
    len_dataset = len(dataset)

    print(f"The dataset has {len_dataset} samples.")

    num_train = int(cfg.train.train_ratio * len_dataset)
    num_val = int(cfg.val.val_ratio * len_dataset)
    num_test = len_dataset - num_train - num_val

    print(f"There are {num_train} training samples.")
    print(f"There are {num_val} validation samples.")
    print(f"There are {num_test} testing samples.")


    train_dataset, val_dataset, test_dataset = random_split(dataset, 
                                                            (num_train,
                                                            num_val,
                                                            num_test),
                                                            torch.Generator().manual_seed(cfg.train.split_random_seed))

    return dataset, train_dataset, val_dataset, test_dataset



def make_dataloaders(cfg, train_dataset, val_dataset, test_dataset):
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=cfg.train.batch_size,
                            shuffle=True,
                            num_workers=cfg.train.num_workers,
                            pin_memory = True
                            )


    val_loader = DataLoader(val_dataset, 
                            batch_size=cfg.val.batch_size,
                            shuffle=False,
                            num_workers=cfg.val.num_workers,
                            pin_memory = True
                            )
    
    test_loader = DataLoader(test_dataset, 
                            batch_size=cfg.test.batch_size,
                            shuffle=False,
                            num_workers=cfg.test.num_workers,
                            pin_memory = True
                            )
    
    print(f"Dataloaders made!")
    
    return train_loader, val_loader, test_loader






    
    