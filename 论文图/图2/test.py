# 导入必要的库和类
from pymatgen.core import Element
import numpy as np

# 定义函数
def get_period(elem_list):
    atomic_numbers = [Element(elem).Z for elem in elem_list]
    # 周期数
    periods = [Element.from_Z(atomic_number).row for atomic_number in atomic_numbers]
    return periods

# 测试用例
elem_list = ["H", "Li", "Na", "K", "Rb", "Cs", "Fr","La","Ce"]
periods = get_period(elem_list)
print("Elements:", elem_list)
print("Periods:", periods)
