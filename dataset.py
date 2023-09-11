import time
import pickle
import torch as t
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gzip
import time
# from config import DefaultConfig
from torch_geometric.data import Data, Batch

import torch
import pytorch_lightning as pl
import json

    
class pygFpDataSet(Dataset):
    def __init__(self, dataset, root_dir):
        super().__init__()
        self.dataset = dataset


        self.protein_graph = pickle.load(open(f"{root_dir}/protein_to_pyg.pkl", "rb"))

        self.ligand_graph = pickle.load(open(f"{root_dir}/ligand_to_pyg.pkl", "rb"))
        
        self.ligand_fp = pickle.load(open(f"{root_dir}/ligand_to_fp.pkl", "rb"))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_fn(self, batch):
        batch_ligand_graph = []
        batch_protein_graph = []
        batch_ligand_fp = []
        ys = []

        for ligand_index, protein_index, y in batch:
            batch_protein_graph.append(self.protein_graph[protein_index])
            batch_ligand_graph.append(self.ligand_graph[ligand_index])
            batch_ligand_fp.append(self.ligand_fp[ligand_index])
            ys.append(y)

        batch_ligand_graph = Batch.from_data_list(batch_ligand_graph)
        batch_protein_graph = Batch.from_data_list(batch_protein_graph)
        ys = torch.Tensor(ys)
        return  ys, torch.stack(batch_ligand_fp), batch_ligand_graph, batch_protein_graph




class testDataModule(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.dataset_name = args["dataset_name"]
        self.batch_size = args["batch_size"]
        self.train_fold_index = 0

    def setup(self, stage):
        self.dataset = pickle.load(open(f'./data/{self.dataset_name}/DTAs.pkl','rb'))
        root_dir = f'./data/{self.dataset_name}'
        train_fold = json.load(open(f"data/{self.dataset_name}/folds/train_fold_setting1.txt"))
        test_fold = json.load(open(f"data/{self.dataset_name}/folds/test_fold_setting1.txt"))

        # [dtas[index] for fold_index in set(folds) - set([fold]) for index in train_fold[fold_index]]
        trainset_fold = [self.dataset[index] for fold_index in set([0, 1, 2, 3, 4])  for index in train_fold[fold_index]]
        testset_fold = [self.dataset[index] for index in test_fold]

        self.train_dataset = pygDataSet(trainset_fold, root_dir)
        self.test_dataset  = pygDataSet(testset_fold,  root_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self.train_dataset.collate_fn, batch_size=self.batch_size, pin_memory=(torch.cuda.is_available()), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, collate_fn=self.test_dataset.collate_fn, batch_size=self.batch_size, pin_memory=(torch.cuda.is_available()))


    
class testFpDataModule(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.dataset_name = args["dataset_name"]
        self.batch_size = args["batch_size"]
        self.train_fold_index = 0

    def setup(self, stage):
        self.dataset = pickle.load(open(f'./data/{self.dataset_name}/DTAs.pkl','rb'))
        root_dir = f'./data/{self.dataset_name}'
        train_fold = json.load(open(f"data/{self.dataset_name}/folds/train_fold_setting1.txt"))
        test_fold = json.load(open(f"data/{self.dataset_name}/folds/test_fold_setting1.txt"))

        # [dtas[index] for fold_index in set(folds) - set([fold]) for index in train_fold[fold_index]]
        trainset_fold = [self.dataset[index] for fold_index in set([0, 1, 2, 3, 4])  for index in train_fold[fold_index]]
        testset_fold = [self.dataset[index] for index in test_fold]

        self.train_dataset = pygFpDataSet(trainset_fold, root_dir)
        self.test_dataset  = pygFpDataSet(testset_fold,  root_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self.train_dataset.collate_fn, batch_size=self.batch_size, pin_memory=(torch.cuda.is_available()), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, collate_fn=self.test_dataset.collate_fn, batch_size=self.batch_size, pin_memory=(torch.cuda.is_available()))
    def test_dataloader(self):
        return DataLoader(self.test_dataset, collate_fn=self.test_dataset.collate_fn, batch_size=self.batch_size, pin_memory=(torch.cuda.is_available()))
    
