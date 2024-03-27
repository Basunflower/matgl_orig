# import gzip
# import json
#
# with gzip.open("my_models_benchmark.json.gz", "rb") as f:
#     a = json.loads(f.read())
# print(type(a))
# print(a)

from pymatgen.core import Element

# 生成原子序数从 1 到 103 的元素符号
elements = tuple(Element.from_Z(z).symbol for z in range(1, 104))

print(elements)
