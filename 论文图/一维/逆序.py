from pymatgen.core import Element

group_1 = ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr']
group_2 = ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra']
group_3 = ['Sc', 'Y']
group_4 = ['Ti', 'Zr', 'Hf']
group_5 = ['V', 'Nb', 'Ta']
group_6 = ['Cr', 'Mo', 'W']
group_7 = ['Mn', 'Tc', 'Re']
group_8 = ['Fe', 'Ru', 'Os']
group_9 = ['Co', 'Rh', 'Ir']
group_10 = ['Ni', 'Pd', 'Pt']
group_11 = ['Cu', 'Ag', 'Au']
group_12 = ['Zn', 'Cd', 'Hg']
group_13 = ['B', 'Al', 'Ga', 'In', 'Tl']
group_14 = ['C', 'Si', 'Ge', 'Sn', 'Pb']
group_15 = ['N', 'P', 'As', 'Sb', 'Bi']
group_16 = ['O', 'S', 'Se', 'Te', 'Po']
group_17 = ['F', 'Cl', 'Br', 'I', 'At']
group_18 = ['He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn']
group_La = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu']
group_Ac = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

MDS = ['Ne', 'At', 'He', 'Cl', 'Lr', 'Md', 'Ar', 'Br', 'Fr', 'N', 'Xe', 'Es', 'Kr', 'O', 'Po', 'No', 'I', 'Fm', 'Na',
       'Cs', 'Rb', 'K', 'Li', 'As', 'Sb', 'W', 'Ge', 'Si', 'Bi', 'Tl', 'Pb', 'Sn', 'Cd', 'Ga', 'In', 'Ba', 'Sr', 'Ca',
       'La', 'Eu', 'U', 'Th', 'Ce', 'Pr', 'Nd', 'Gd', 'Sm', 'Yb', 'Tb', 'Dy', 'Y', 'Ho', 'Er', 'Tm', 'Lu', 'Np', 'Pu',
       'Sc', 'Zr', 'Cf', 'Pa', 'Cm', 'Am', 'Ac', 'Pm', 'Hf', 'Bk', 'Nb', 'Ta', 'Ti', 'V', 'Al', 'Mn', 'Cr', 'Mg', 'Fe',
       'Co', 'Ni', 'Zn', 'Rh', 'Pt', 'Ir', 'Pd', 'Cu', 'Os', 'Hg', 'Ru', 'Tc', 'Au', 'Ag', 'Mo', 'S', 'Be', 'Se', 'Te',
       'P', 'Re', 'H', 'Rn', 'Ra', 'F', 'B', 'C']


def G_Group(group):  # 生成原子序数在1-103范围内不同主族的元素
    elements = [Element.from_Z(i) for i in range(1, 104)]

    # 第x主族的元素
    group1_elements = [element for element in elements if element.group == group]
    lanthanides = [element for element in elements if 57 <= element.Z <= 71]
    actinides = [element for element in elements if 89 <= element.Z <= 103]
    print([element.symbol for element in group1_elements])


def count_inversions(arr1, arr2):  # 计算逆序数，只计算两者共有元素的逆序数（arr1和arr2位置无关）
    inversions = 0
    n = len(arr1)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if arr1[i] not in arr2 or arr1[j] not in arr2:  # 如果 arr1[i] 或 arr1[j] 不在 arr2 中
                continue  # 忽略缺失的元素
            elif arr2.index(arr1[i]) > arr2.index(arr1[j]):  # 如果 arr1[i] 在 arr2 中的位置比 arr1[j] 大
                inversions += 1  # 逆序数加一
    print(inversions)
    return inversions


group_list = [group_1, group_2, group_3, group_4, group_5, group_6, group_7, group_8, group_9, group_10, group_11,
              group_12, group_13, group_14, group_15, group_16, group_17, group_18, group_La, group_Ac]


def count_main_group_inversions(sorted_1d):
    """
    计算排序和Z之间 在不同主族上的逆序
    输入：排序
    输出：在18+2个主族上的逆序数
    """
    total_inversions = 0
    i = 1
    for group in group_list:
        print("主族", i, ":", end="")
        inversion = count_inversions(group, sorted_1d)
        total_inversions += inversion
        i += 1
    return total_inversions


test_elem = ['Li', 'Be', 'B', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
             'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
             'In', 'Sn', 'Sb', 'Te', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
             'Bi']
test_fold = ['Ba', 'B', 'La', 'W', 'Ta', 'V', 'Sc', 'Nb', 'Sb', 'Hf', 'Ti', 'Ge', 'Zr', 'Al', 'Si', 'As', 'Ga', 'Sn',
             'In', 'Ir', 'Ni', 'Os', 'Re', 'Mo', 'Cr', 'Bi', 'Ru', 'Pb', 'Co', 'Be', 'Te', 'F', 'N', 'Rh', 'Y', 'Zn',
             'Tl', 'Mg', 'Cu', 'Au', 'Pd', 'Ag', 'O', 'Ca', 'Pt', 'Hg', 'Mn', 'S', 'Fe', 'Li', 'Cs', 'Na', 'Cd', 'Sr',
             'Rb', 'K']

if __name__ == '__main__':
    # 保留有意义元素的mds 用交集的方法
    set_test_elem = set(test_elem)  # 转换为集合
    test_mds = [elem for elem in MDS if elem in set_test_elem]

    a = count_main_group_inversions(test_mds)
    b = count_main_group_inversions(test_fold)
    print(a)
    print(b)
# 主族 1 :18
# 主族 2 :10
# 主族 3 :1
# 主族 4 :2
# 主族 5 :2
# 主族 6 :2
# 主族 7 :0
# 主族 8 :1
# 主族 9 :0
# 主族 10 :1
# 主族 11 :1
# 主族 12 :1
# 主族 13 :9
# 主族 14 :6
# 主族 15 :3
# 主族 16 :3
# 主族 17 :7
# 主族 18 :2
# 主族 19 :20
# 主族 20 :74
