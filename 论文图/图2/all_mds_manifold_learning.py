from __future__ import annotations

import argparse
import sys
import warnings
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    # plt.show()


if __name__ == '__main__':
    init_seed = 42
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)  # 用于numpy的随机数
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 数据集中所有的元素类型
    elem_list = [Element.from_Z(z).symbol for z in range(1, 104)]

    df = pd.read_csv('MDS_64dim.csv')
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

    S_points, S_color = embedding_weights, main_group

    n_neighbors = 12  # neighborhood which is used to recover the locally linear structure
    n_components = 2  # number of coordinates for the manifold

    # plt.show()
    # Isomap Embedding
    isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=1)
    S_isomap = isomap.fit_transform(S_points)
    plot_2d(S_isomap, S_color, "Isomap Embedding")
    plt.savefig('Isomap_Embedding.png', dpi=500, bbox_inches='tight', pad_inches=0)
    plt.clf()
    # Multidimensional scaling
    md_scaling = manifold.MDS(
        n_components=n_components,
        max_iter=50,
        n_init=4,
        random_state=0,
        normalized_stress=False,
    )
    S_scaling = md_scaling.fit_transform(S_points)

    plot_2d(S_scaling, S_color, "Multidimensional scaling")
    plt.savefig('Multidimensional_scaling.png', dpi=500, bbox_inches='tight', pad_inches=0)
    plt.clf()

    # Spectral embedding for non-linear dimensionality reduction
    spectral = manifold.SpectralEmbedding(
        n_components=n_components, n_neighbors=n_neighbors, random_state=42
    )
    S_spectral = spectral.fit_transform(S_points)

    plot_2d(S_spectral, S_color, "Spectral Embedding")
    plt.savefig('Spectral_embedding_for_non-linear_dimensionality_reduction.png', dpi=500, bbox_inches='tight', pad_inches=0)
    plt.clf()

    # T-distributed Stochastic Neighbor Embedding
    t_sne = manifold.TSNE(
        n_components=n_components,
        perplexity=30,
        init="random",
        n_iter=250,
        random_state=0,
    )
    S_t_sne = t_sne.fit_transform(S_points)

    plot_2d(S_t_sne, S_color, "T-distributed Stochastic  \n Neighbor Embedding")
    plt.savefig('T-distributed_Stochastic.png', dpi=500, bbox_inches='tight', pad_inches=0)
    plt.clf()

