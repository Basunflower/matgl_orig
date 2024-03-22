from __future__ import annotations

import os
import warnings

import pandas as pd
import torch
from dgl.data.utils import split_dataset
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MEGNetDataset
from pymatgen.core import Element
from pymatgen.core import Structure
from tqdm import tqdm


# To suppress warnings for clearer output
warnings.simplefilter("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_DEBUG'] = '1'


# 检查目录下文件数量
def count_files_in_folder(folder_path):

    files_iterator = os.scandir(folder_path)
    file_count = sum(entry.is_file() for entry in files_iterator)
    return file_count


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

folder = "data_save"


def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
    """Raw data loading function.

    Returns:
        tuple[list[Structure], list[str], list[float]]: structures, mp_id, Eform_per_atom
    """
    # 读取数据集（完整）
    data = pd.read_json("../node_embedding/mp.2018.6.1.json")
    # data = pd.read_json("../node_embedding/first_10_data.json")
    structures = []
    mp_ids = []

    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"])):
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_ids.append(mid)

    return structures, mp_ids, data["formation_energy_per_atom"].tolist()


def gen_and_dump(name, dataset):
    io_buffer = []  # 用于存储生成的数据字典的缓冲区
    io_cnt = 0  # 计数生成的文件数量
    for i in range(len(dataset)):
        io_buffer.append(dataset[i])
        if len(io_buffer) > 100 or i == len(dataset) - 1:
            for data in io_buffer:
                save_path = "{}/{}/{}.pth".format(folder, name, io_cnt)
                # Check if the directory exists, and create it if not
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

                # Save the data to the specified path
                torch.save(data, save_path)
                io_cnt += 1
            io_buffer.clear()


def main():
    if not os.path.exists(os.path.dirname(folder)):
        # 结构,_,标签
        structures, mp_ids, eform_per_atom = load_dataset()
        # 数据集中所有的元素类型
        elem_list = get_element_list(structures)
        # 结构转化为图
        converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
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
        train_data, val_data, test_data = split_dataset(
            mp_dataset,
            frac_list=[0.8, 0.1, 0.1],  # 比例
            shuffle=True,
            random_state=42,
        )
        gen_and_dump(name='train', dataset=train_data)
        gen_and_dump(name='val', dataset=val_data)
        gen_and_dump(name='test', dataset=test_data)

        for fn in ("dgl_graph.bin", "lattice.pt", "dgl_line_graph.bin", "state_attr.pt", "labels.json"):
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    main()
