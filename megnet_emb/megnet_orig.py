from __future__ import annotations
import sys
import os
import shutil
import warnings
import zipfile
from typing import List

import numpy as np
from matgl.config import DEFAULT_ELEMENTS
import dgl
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from dgl.data.utils import split_dataset
from pymatgen.core import Structure
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MEGNetDataset, MGLDataLoader, collate_fn
from matgl.layers import BondExpansion
from matgl.models._megnet import MEGNet
from matgl.utils.io import RemoteFile
from matgl.utils.training import ModelLightningModule
from pymatgen.core import Element
import argparse
from torch.optim.lr_scheduler import LambdaLR
from matbench.bench import MatbenchBenchmark
from dgl.dataloading import GraphDataLoader
# from kgcnn.training.scheduler import LinearLearningRateScheduler

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

# bash megnet_orig.sh
parser = argparse.ArgumentParser(description='MEGNet')
parser.add_argument('--dim_node_embed', type=int, default=64, help='number of node embedding dim')
args = parser.parse_args()


# Dataset Preparation
def get_data(input, output):
    structures = []
    for structure_str in input:
        structures.append(structure_str)
    return structures, output.tolist()


if __name__ == '__main__':
    for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
        try:
            os.remove(fn)
        except FileNotFoundError:
            pass
    init_seed = 42
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)  # 用于numpy的随机数
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    mb = MatbenchBenchmark(
        autoload=False,
        subset=[
            # "matbench_mp_e_form",  # 回归 13w
            "matbench_perovskites"  # 1w8
            # "matbench_jdft2d"
        ]
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        # torch.set_default_device("cuda:1")

    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            train_inputs, train_outputs = task.get_train_and_val_data(fold)

            structures, eform_per_atom = get_data(train_inputs, train_outputs)

            # 数据集中所有的元素类型
            elem_list = get_element_list(structures)

            # 结构转化为图
            converter = Structure2Graph(element_types=elem_list, cutoff=5.0)

            # 把原数据集转化为megnet数据集
            mp_dataset = MEGNetDataset(
                structures=structures,  # 结构
                labels={"Eform": eform_per_atom},  # 标签
                converter=converter,  # 图
                initial=0.0,  # 高斯扩展的初始距离
                final=5.0,  # 高斯扩展的最终距离
                num_centers=100,  # 高斯函数的数量
                width=0.5,  # 高斯函数的宽度
            )

            # 拆分数据集为训练、验证、测试集
            train_data, val_data, _ = split_dataset(
                mp_dataset,
                frac_list=[0.8, 0.2, 0.0],  # 比例
                shuffle=True,
                random_state=42,
            )

            train_loader, val_loader = MGLDataLoader(
                train_data=train_data,
                val_data=val_data,
                collate_fn=collate_fn,
                batch_size=32,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )

            # define the bond expansion
            bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=25, width=0.4)
            # a Set2Set encoder for node and edge embeddings, feed-forward blocks of units [64, 32], softplus activation and gauss distance expansion with cutoff of 5A and 25 bins with 0.4 sigma. We used a larger input embedding vector [64] of atom species and added the charge as input graph attributes. We trained with MAE loss and a linear learning rate scheduler from 5e-4 to 5e-6 over 1000 epochs using Adam. We added a standard scaler for regression.

            # setup the architecture of MEGNet model
            model = MEGNet(
                dim_node_embedding=args.dim_node_embed,  # 元素嵌入维度
                dim_edge_embedding=100,
                dim_state_embedding=2,
                nblocks=3,
                hidden_layer_sizes_input=(64, 32),
                hidden_layer_sizes_conv=(64, 32, 32),
                nlayers_set2set=1,
                niters_set2set=3,
                hidden_layer_sizes_output=(32, 16),
                is_classification=False,
                activation_type="softplus2",
                element_types=DEFAULT_ELEMENTS,  # 更改
                bond_expansion=bond_expansion,
                cutoff=4.0,
                gauss_width=0.5,
            )

            # optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            # scheduler = LinearLearningRateScheduler(learning_rate_start=0.0005, learning_rate_stop=0.5e-05, epo_min=100, epo=1000)
            # setup the MEGNetTrainer
            lit_module = ModelLightningModule(model=model, loss="mae_loss",
                                              # optimizer=optimizer,
                                              # scheduler=scheduler
                                              )

            # early_stop_callback = EarlyStopping(monitor="val_MAE", min_delta=0.00, patience=500, verbose=True, mode="min")
            checkpoint_callback = ModelCheckpoint(
                monitor='val_MAE', dirpath='megnet_orig_pre_train_model',  # Directory to save the checkpoints
                filename=str(args.dim_node_embed) + '_dim__MEGNet_nnEmbed__fold_'+str(fold+1), save_top_k=1,
                mode='min')
            # Training
            trainer = pl.Trainer(max_epochs=1000, callbacks=[checkpoint_callback])
            trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # 测试部分
            model.eval()
            test_inputs, test_outputs = task.get_test_data(fold, include_target=True)
            test_sturcture, test_Eform = get_data(test_inputs, test_outputs)
            # 把原数据集转化为megnet数据集
            test_dataset = MEGNetDataset(
                structures=test_sturcture,  # 结构
                labels={"Eform": test_Eform},  # 标签
                converter=converter,  # 图
                initial=0.0,  # 高斯扩展的初始距离
                final=5.0,  # 高斯扩展的最终距离
                num_centers=100,  # 高斯函数的数量
                width=0.5,  # 高斯函数的宽度
            )

            kwargs = {
                "batch_size": 32,
                "num_workers": 4,
                "pin_memory": torch.cuda.is_available()
            }

            test_loader = GraphDataLoader(test_dataset, collate_fn=collate_fn, **kwargs)

            predict = trainer.test(model=lit_module, dataloaders=test_loader)
            # This code just performs cleanup for this notebook.
            for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
                try:
                    os.remove(fn)
                except FileNotFoundError:
                    pass

    # # Save your results
    # mb.to_file("my_models_benchmark.json.gz")
