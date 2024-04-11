from __future__ import annotations

import argparse
import sys
import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matbench.bench import MatbenchBenchmark
from matgl.ext.pymatgen import get_element_list
from pymatgen.core import Element
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.random_projection import SparseRandomProjection
from sklearn import manifold
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


def get_main_group(elem_list):
    # 原子序数
    atomic_numbers = [Element(elem).Z for elem in elem_list]
    # 主族，镧系19，锕系20
    main_groups = np.array([19 if atomic_number in range(57, 72)
                            else
                            20 if atomic_number in range(89, 104)
                            else
                            Element.from_Z(atomic_number).group
                            for atomic_number in atomic_numbers])
    return main_groups


if __name__ == '__main__':
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
            "matbench_perovskites"  # 1w8
        ]
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    for task in mb.tasks:
        task.load()
        for fold in task.folds:
            train_inputs, train_outputs = task.get_train_and_val_data(fold)
            structures, eform_per_atom = get_data(train_inputs, train_outputs)
            # 数据集中所有的元素类型
            elem_list = get_element_list(structures)
            print("有意义元素：")
            print(elem_list)
            # 读取模型嵌入层权重
            checkpoint = torch.load('64_dim__MEGNet_nnEmbed__fold_' + str(fold + 1) + '.ckpt', map_location='cpu')
            embedding_weights = checkpoint['state_dict']['model.embedding.layer_node_embedding.weight']
            # 读取训练的元素的权重
            embedding_weights = np.array(embedding_weights.tolist()[:len(elem_list)])
            X = embedding_weights
            tsne = manifold.TSNE(
                n_components=1,
                random_state=0,
                learning_rate='auto',
                n_iter=5000,
                n_jobs=8
            )
            X_1d = tsne.fit_transform(X)

            elem_dict = dict(zip(elem_list, X_1d.flatten()))
            # 根据字典的值对其进行排序
            sorted_dict = sorted(elem_dict.items(), key=lambda x: x[1])
            sorted_elem_list = [item[0] for item in sorted_dict]
            print(sorted_elem_list)
            sys.exit()

