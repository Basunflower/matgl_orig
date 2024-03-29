# import gzip
# import json
#
# with gzip.open("my_models_benchmark.json.gz", "rb") as f:
#     a = json.loads(f.read())
# print(type(a))
# print(a)


import csv
def load_pretrain_embeddings(dimension):
    coordinates = []
    # 读取CSV文件
    with open('../../MDS/MDS_'+str(dimension)+'dim.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        for row in reader:
            # 解析坐标字符串并将其转换为列表
            coordinate_str = row[1][1:-1]  # 去除方括号
            coordinate_list = [float(coord.strip()) for coord in coordinate_str.split(',')]
            coordinates.append(coordinate_list)
    return coordinates


import numpy as np
from pymatgen.core import Element

# 元素列表
elements = ('H', 'Li', 'Be', 'B', 'N', 'O', 'Hf', 'La','Ce','Ac','Th','Pa','Lu')

# 获取每个元素的原子序数
atomic_numbers = [Element(elem).Z for elem in elements]

main_groups = np.array([19 if atomic_number in range(57, 72) else 20 if atomic_number in range(89, 104) else Element.from_Z(atomic_number).group for atomic_number in atomic_numbers])
print(main_groups)

