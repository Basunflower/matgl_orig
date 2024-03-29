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
            # 读取模型嵌入层权重
            checkpoint = torch.load('64_dim__MEGNet_nnEmbed__fold_' + str(fold + 1) + '.ckpt', map_location='cpu')
            embedding_weights = checkpoint['state_dict']['model.embedding.layer_node_embedding.weight']
            # 读取训练的元素的权重
            embedding_weights = np.array(embedding_weights.tolist()[:len(elem_list)])
            # 主族
            main_group = get_main_group(elem_list)

            X, y = embedding_weights, main_group
            n_samples, n_features = X.shape
            n_neighbors = 30

            def plot_embedding(X, title):
                _, ax = plt.subplots()
                X = MinMaxScaler().fit_transform(X)

                unique_labels = np.unique(y)
                for label in unique_labels:
                    ax.scatter(
                        *X[y == label].T,
                        marker=f"${label}$",
                        s=60,
                        color=plt.cm.Dark2(label),
                        alpha=0.425,
                        zorder=2,
                    )
                shown_images = np.array([[1.0, 1.0]])  # just something big
                for i in range(X.shape[0]):
                    # plot every digit on the embedding
                    # show an annotation box for a group of digits
                    dist = np.sum((X[i] - shown_images) ** 2, 1)
                    if np.min(dist) < 4e-3:
                        # don't show points that are too close
                        continue
                    shown_images = np.concatenate([shown_images, [X[i]]], axis=0)
                    # imagebox = offsetbox.AnnotationBbox(
                    #     offsetbox.OffsetImage(np.ones((1, 1)), cmap=plt.cm.gray_r), X[i]
                    # )
                    # imagebox.set(zorder=1)
                    # ax.add_artist(imagebox)

                ax.set_title(title)
                ax.axis("off")

            embeddings_reduction_methods = {
                "Random projection embedding": SparseRandomProjection(
                    n_components=2, random_state=42
                ),
                "Truncated SVD embedding": TruncatedSVD(n_components=2),
                "Linear Discriminant Analysis embedding": LinearDiscriminantAnalysis(
                    n_components=2
                ),
                "Isomap embedding": Isomap(n_neighbors=n_neighbors, n_components=2),
                "Standard LLE embedding": LocallyLinearEmbedding(
                    n_neighbors=n_neighbors, n_components=2, method="standard"
                ),
                "Modified LLE embedding": LocallyLinearEmbedding(
                    n_neighbors=n_neighbors, n_components=2, method="modified"
                ),
                "Hessian LLE embedding": LocallyLinearEmbedding(
                    n_neighbors=n_neighbors, n_components=2, method="hessian"
                ),
                "LTSA LLE embedding": LocallyLinearEmbedding(
                    n_neighbors=n_neighbors, n_components=2, method="ltsa"
                ),
                "MDS embedding": MDS(n_components=2, n_init=1, max_iter=120, n_jobs=2),
                "Random Trees embedding": make_pipeline(
                    RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0),
                    TruncatedSVD(n_components=2),
                ),
                "Spectral embedding": SpectralEmbedding(
                    n_components=2, random_state=0, eigen_solver="arpack"
                ),
                "t-SNE embedding": TSNE(
                    n_components=2,
                    n_iter=500,
                    n_iter_without_progress=150,
                    n_jobs=2,
                    random_state=0,
                ),
                "NCA embedding": NeighborhoodComponentsAnalysis(
                    n_components=2, init="pca", random_state=0
                ),
            }

            projections, timing = {}, {}
            for name, transformer in embeddings_reduction_methods.items():
                if name.startswith("Linear Discriminant Analysis"):
                    data = X.copy()
                    data.flat[:: X.shape[1] + 1] += 0.01  # Make X invertible
                else:
                    data = X

                print(f"Computing {name}...")
                start_time = time()
                projections[name] = transformer.fit_transform(data, y)
                timing[name] = time() - start_time

            for name in timing:
                title = f"{name}"
                plot_embedding(projections[name], title)

                plt.savefig(name+'.png', bbox_inches='tight', dpi=300, pad_inches=0)
            plt.show()
            sys.exit()
