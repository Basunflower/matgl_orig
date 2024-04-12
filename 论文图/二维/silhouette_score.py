from __future__ import annotations

from sklearn.metrics import silhouette_score
import argparse
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matbench.bench import MatbenchBenchmark
from matgl.ext.pymatgen import get_element_list
from pymatgen.core import Element

from sklearn import manifold
from matplotlib import ticker

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
            print("有意义元素：", elem_list)

            # MDS
            df = pd.read_csv('MDS_2dim.csv')
            # 从 DataFrame 中选择相关行
            selected_rows = df[df['Element'].isin(elem_list)]
            embedding_weights = selected_rows['Coordinates'].tolist()
            array_data = []
            for line in embedding_weights:
                line = line.replace('[', '').replace(']', '')  # 去掉列表标记
                array_data.append([float(num) for num in line.strip().split(',')])

            embedding_weights = np.array(array_data)
            # 主族
            main_group = get_main_group(elem_list)

            X, y = embedding_weights, main_group
            silhouette_avg = silhouette_score(X, y)
            print("MDS轮廓系数：", silhouette_avg)
            # embedding
            # 读取模型嵌入层权重
            checkpoint = torch.load('64_dim__MEGNet_nnEmbed__fold_' + str(fold + 1) + '.ckpt', map_location='cpu')
            embedding_weights = checkpoint['state_dict']['model.embedding.layer_node_embedding.weight']
            # 读取训练的元素的权重
            embedding_weights = np.array(embedding_weights.tolist()[:len(elem_list)])
            X = embedding_weights
            tsne = manifold.TSNE(
                n_components=2,
                random_state=0,
                learning_rate='auto',
                n_iter=5000,
                n_jobs=8
            )
            X_2d = tsne.fit_transform(X)
            silhouette_avg = silhouette_score(X_2d, y)
            print("test轮廓系数：", silhouette_avg)
            sys.exit()


