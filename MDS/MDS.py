import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from pymatgen.core import Element
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
import csv


def get_main_group(element_symbol):  # 获取主族，把澜系和e系化为19和20主族 一个一个的获取
    ele = Element(element_symbol)
    element_number = ele.number
    if 57 <= element_number <= 71:
        return 19
    elif 89 <= element_number <= 103:
        return 20
    else:
        return ele.group


def load_similar_matrix_100():  # 获得替换矩阵 为非百分比形式
    text_data = [
        "H →",
        "He →",
        "Li → Na (6.2)",
        "Be →",
        "B →",
        "C →",
        "N →",
        "O →",
        "F →",
        "Ne →",
        "Na → K (5.3)",
        "Mg → Mn (9.1), Zn (8.7), Cd (7.9), Fe (7.2), Ni (6.8), Co (6.7)",
        "Al → Ga (13.0), In (7.3), Fe (5.4)",
        "Si → Ge (15.8)",
        "P → As (7.2)",
        "S → Se (9.6)",
        "Cl → Br (9.8)",
        "Ar →",
        "K → Rb (10.3), Cs (5.5)",
        "Ca → Sr (15.2), Ba (7.2), Eu (6.9), Yb (5.7)",
        "Sc → Y (18.2), Lu (15.9), Tm (14.9), Er (14.6), Zr (14.4), Tb (14.3), Yb (13.8), Dy (13.4), Ho (13.3), Gd (12.6), Ti (11.1), Sm (11.1), Nd (10.4), La (9.5), Pr (9.1), Ce (9.0), Hf(9.0), In (8.1), Mn (8.0), U (7.8), Pu (6.0), V (5.3), Eu (5.3), Th (5.3), Al (5.1), Cr(5.1), Nb (5.1)",
        "Ti → Zr (11.5), V (9.8), Hf (7.8), Mn (7.2), Nb (5.9), Fe (5.6), Sc (5.6), Cr (5.6)",
        "V → Cr (9.4), Ti (8.2), Fe (6.2), Nb (5.5), Mn (5.1)",
        "Cr → Fe (14.2), V (12.9), Mn (10.0), Co (7.1), Ti (6.3), Rh (5.4), Ni (5.1)",
        "Mn → Fe (14.7), Co (10.7), Ni (8.2), Zn (6.6), Mg (6.1), Cr (5.5)",
        "Fe → Co (15.5), Mn (12.4), Ni (11.6), Cr (6.6), Al (5.2)",
        "Co → Ni (24.2), Fe (18.8), Mn (10.9), Rh (8.6), Ru (6.6), Zn (6.3), Cu (5.9)",
        "Ni → Co (20.9), Fe (12.1), Pd (10.0), Cu (8.9), Mn (7.3), Pt (7.3), Rh (6.9), Zn (5.1)",
        "Cu → Ni (6.9)",
        "Zn → Mn (8.4), Co (7.9), Mg (7.5), Ni (7.4), Cd (7.4), Fe (5.5)",
        "Ga → Al (22.5), In (12.5), Sn (6.1), Fe (5.5)",
        "Ge → Si (27.5), Sn (7.8)",
        "As → P (18.2), Sb (11.8)",
        "Se → S (21.5), Te (10.2)",
        "Br → Cl (25.0), I (13.3)",
        "Kr →",
        "Rb → K (22.1), Cs (21.9)",
        "Sr → Ba (18.1), Ca (17.4), Eu (9.1)",
        "Y → Ho (29.3), Er (29.2), Dy (29.0), Gd (28.2), Tb (28.1), Sm (23.3), Tm (22.2), Nd (21.8), Lu (20.5), Yb (20.4), La (18.4), Pr (18.1), Ce (17.2), Sc (11.7), Eu (11.3), U (8.2), Th (7.7), Pu (6.3), Ca (5.5), Zr (5.3)",
        "Zr → Hf (24.7), Ti (16.0), Sc (10.0), U (7.7), Nb (6.4), Tb (6.0), Y (5.8), Ta (5.4), Ce (5.2), Tm (5.2)",
        "Nb → Ta (19.6), V (7.9), Ti (7.1), Zr (5.6), Hf (5.5)",
        "Mo → W (9.3)",
        "Tc → Re (19.1), Ru (8.8), Ti (7.4), Os (7.4), Mn (5.9), Fe (5.9), Pt (5.9)",
        "Ru → Rh (16.3), Co (16.0), Fe (13.2), Os (13.1), Ni (11.8), Ir (11.5), Mn (8.8), Pt (7.5), Pd (6.9)",
        "Rh → Ir (21.3), Co (19.9), Pd (18.8), Pt (18.7), Ni (18.5), Ru (15.5), Fe (12.8), Cu (8.3), Cr (7.0), Mn (6.7), Os (6.1), Al (5.5), Au (5.4)",
        "Pd → Pt (19.6), Ni (19.3), Rh (13.5), Cu (10.4), Au (8.4), Co (7.8), Ir (5.7), Fe (5.0)",
        "Ag → Cu (11.9), Au (9.7)",
        "Cd → Zn (10.5), Mg (9.5), Mn (8.1), Hg (7.9)",
        "In → Al (11.1), Ga (11.1), Tl (8.3), Sn (8.2)",
        "Sn → Ge (9.9), Pb (9.4), In (7.0), Si (6.7)",
        "Sb → As (11.1), Bi (9.2), P (6.1)",
        "Te → Se (14.3), S (9.7)",
        "I → Br (10.7), Cl (7.2)",
        "Xe →",
        "Cs → Rb (16.1), K (8.6)",
        "Ba → Sr (12.0), Ca (5.4)",
        "La → Ce (27.8), Nd (27.0), Pr (24.5), Sm (18.9), Gd (16.9), Tb (13.3), Y (13.0), Dy (12.4), Ho (11.2), Er (11.1), Yb (9.3), Eu (9.2), Tm (8.3), Lu (7.4), U (6.6), Th (5.4), Sr (5.3), Ca (5.1)",
        "Ce → La (36.2), Pr (32.6), Nd (31.0), Sm (22.9), Gd (21.9), Tb (18.2), Dy (17.3), Y (15.9), Er (15.7), Ho (15.0), Yb (12.3), Tm (10.7), Eu (9.7), Lu (9.6), U (9.6), Th (9.0), Pu (7.5), Ca (6.1), Np (5.7), Sc (5.3), Sr (5.1)",
        "Pr → Nd (54.6), Ce (42.3), La (41.5), Sm (38.5), Gd (31.2), Tb (27.8), Dy (25.8), Ho (23.2), Er (22.7), Y (21.8), Tm (17.8), Yb (17.6), Eu (16.4), Lu (14.6), U (10.9), Th (10.0), Ca (8.5), Pu (8.5), Np (7.7), Sc (7.1), Sr (6.7)",
        "Nd → Pr (45.5), La (38.1), Sm (36.0), Ce (33.5), Gd (31.8), Tb (26.2), Ho (23.3), Dy (23.2), Y (21.8), Er (21.6), Yb (16.3), Eu (16.0), Tm (15.3), Lu (13.9), U (9.7), Th (8.9), Pu (7.2), Ca (6.7), Sc (6.7), Np (5.9), Sr (5.7)",
        "Pm → Y (71.4), Sm (71.4), Dy (71.4), Ho (71.4), Lu (71.4), Sc (57.1), La (57.1), Ce (57.1), Pr (57.1), Nd (57.1), Gd (57.1), Tb (57.1), Er (57.1), Tm (57.1), Eu (42.9), Tl (28.6), Ca (14.3), Mn (14.3), Sr (14.3), Rh (14.3), Pd (14.3), Ag (14.3), In (14.3), Ba (14.3), Yb (14.3), Hg (14.3), Pb (14.3), Ac (14.3), Pu (14.3), Cf (14.3)",
        "Sm → Nd (49.4), Gd (47.7), Pr (44.0), La (36.7), Ce (34.0), Tb (34.0), Dy (33.9), Y (32.0), Ho (30.8), Er (29.9), Yb (24.0), Eu (23.8), Tm (23.6), Lu (18.6), U (11.1), Sc (9.8), Th (9.8), Pu (8.2), Sr (7.5), Ca (7.0), Np (6.6)",
        "Eu → Sm (23.8), Gd (23.2), Nd (22.1), Sr (21.0), Yb (20.0), Pr (18.8), Ca (18.2), La (17.9), Dy (16.9), Tb (16.6), Ho (15.7), Er (15.7), Y (15.6), Ce (14.4), Ba (13.5), Tm (12.2), Lu (10.3), U (6.2), Th (5.6)",
        "Gd → Sm (41.2), Dy (40.2), Tb (38.6), Nd (37.8), Er (37.6), Ho (34.9), Y (33.5), Pr (30.8), La (28.3), Ce (28.2), Tm (24.7), Yb (22.8), Lu (21.4), Eu (20.1), U (10.2), Sc (9.6), Th (9.3), Pu (7.7), Ca (6.7), Sr (5.7), Np (5.6)",
        "Tb → Dy (49.6), Ho (47.5), Gd (46.7), Er (44.9), Y (40.4), Nd (37.6), Sm (35.6), Pr (33.3), Tm (32.3), Ce (28.3), La (27.0), Yb (27.0), Lu (26.3), Eu (17.4), Sc (13.2), U (12.9), Th (10.4), Pu (8.9), Zr (8.0), Np (7.1), Ca (6.8), Sr (5.8)",
        "Dy → Ho (55.9), Er (55.7), Tb (53.0), Gd (52.0), Y (44.6), Sm (37.9), Tm (37.9), Nd (35.6), Pr (33.0), Yb (31.0), Lu (29.7), Ce (28.7), La (26.9), Eu (18.9), Sc (13.3), U (12.8), Th (11.1), Pu (9.5), Ca (8.2), Np (7.7), Zr (6.2), Sr (5.7)",
        "Ho → Er (57.2), Dy (55.3), Tb (50.1), Y (44.6), Gd (44.6), Tm (39.9), Nd (35.3), Yb (34.5), Sm (34.0), Lu (31.1), Pr (29.3), Ce (24.6), La (24.0), Eu (17.3), Sc (13.0), U (12.6), Th (9.9), Pu (8.9), Ca (7.8), Np (7.1), Zr (7.0), Sr (5.0)",
        "Er → Ho (51.2), Dy (49.3), Gd (42.9), Tb (42.3), Y (39.7), Tm (37.8), Yb (30.4), Sm (29.6), Nd (29.3), Lu (29.3), Pr (25.7), Ce (23.0), La (21.2), Eu (15.5), Sc (12.8), U (10.9), Th (8.7), Pu (8.3), Ca (6.8), Np (6.4), Zr (6.1)",
        "Tm → Er (63.5), Ho (59.9), Dy (56.2), Tb (51.1), Y (50.6), Gd (47.4), Yb (47.2), Lu (45.3), Sm (39.2), Nd (34.8), Pr (33.8), La (26.5), Ce (26.3), Sc (21.9), Eu (20.2), U (16.8), Th (13.6), Pu (11.4), Zr (10.9), Np (9.0), Ca (8.3), Sr (6.6), Ti (5.4), Mn (5.4), In (5.1), Hf (5.1)",
        "Yb → Ho (31.0), Er (30.6), Tm (28.3), Y (27.8), Dy (27.6), Gd (26.2), Tb (25.7), Lu (25.7), Sm (23.9), Nd (22.3), Pr (20.0), Eu (19.8), Ce (18.2), La (17.9), Ca (15.0), Sc (12.1), U (10.5), Sr (9.6), Th (8.3), Pu (7.7), Np (5.5), Zr (5.4), Ba (5.2)",
        "Lu → Er (48.1), Y (45.7), Ho (45.7), Tm (44.3), Dy (43.1), Yb (41.9), Tb (40.7), Gd (40.2), Nd (31.0), Sm (30.2), Pr (27.1), La (23.3), Ce (23.1), Sc (22.9), Eu (16.7), U (13.6), Pu (10.7), Th (10.2), Zr (10.0), Np (8.3), Ca (7.1), In (6.4), Mn (6.2), Hf (5.7)",
        "Hf → Zr (52.6), Ti (23.0), Nb (13.4), Sc (13.2), Ta (9.3), U (8.8), V (7.1), Mn (6.8), Th(6.8), Y (6.4), Ce (6.1), Lu (5.9), Pu (5.9), Tb (5.6), Dy (5.6), Ho (5.4), Gd (5.1), Er(5.1), Tm (5.1)",
        "Ta → Nb (28.4), V (8.6), Ti (6.8), Zr (6.8), Hf (5.5)",
        "W → Mo (17.1)",
        "Re → W (5.7)",
        "Os → Ru (30.5), Ir (17.2), Rh (15.1), Fe (13.6), Co (13.3), Pt (12.7), Ni (10.9), Mn (9.7), Pd (8.2), Cr (6.3), Re (6.0)",
        "Ir → Rh (29.4), Pt (22.5), Ru (15.1), Co (14.9), Ni (12.9), Pd (11.0), Os (9.6), Fe (8.5), Al (6.6), Mn (5.4)",
        "Pt → Pd (24.7), Ni (17.6), Rh (16.9), Ir (14.7), Co (8.6), Au (8.2), Fe (6.4), Ru (6.4), Cu (5.9)",
        "Au → Cu (15.0), Ag (12.9), Pd (11.5), Pt (8.9), Ni (6.9), Rh (5.3)",
        "Hg → Cd (12.3), Zn (5.8)",
        "Tl → In (14.1), K (8.8), Rb (7.3), Sn (5.8), Pb (5.8), Ga (5.2), Cs (5.0)",
        "Pb → Sn (14.3), In (6.3)",
        "Bi → Sb (13.9), As (6.7)",
        "Po →",
        "At →",
        "Rn →",
        "Fr →",
        "Ra →",
        "Ac → La (85.7), Nd (71.4), Ce (57.1), Pr (57.1), Am (57.1), Sm (42.9), Eu (42.9), Gd (42.9), Np (42.9), Pu (42.9), Cm (42.9), Sc (28.6), Ho (28.6), Yb (28.6), Bi (28.6), U (28.6), Cf (28.6), Mg (14.3), S (14.3), Ti (14.3), V (14.3), Cr (14.3), Se (14.3), Y (14.3), Zr (14.3), Nb (14.3), Pm (14.3), Tb (14.3), Dy (14.3), Er (14.3), Tm (14.3), Lu (14.3), Ta (14.3), Bk (14.3)",
        "Th → U (34.4), Ce (22.2), Nd (20.2), Pr (19.0), Gd (17.8), Y (17.6), La (17.3), Tb (16.6), Dy (16.6), Sm (16.3), Ho (14.9), Er (14.6), Yb (13.9), Tm (13.7), Pu (13.2), Np (12.4), Lu (10.5), Zr (10.2), Eu (9.3), Ca (8.0), Sc (7.8), Sr (6.8), Hf (6.8)",
        "Pa → U (60.0), Th (36.7), Np (33.3), Zr (26.7), Ce (26.7), Pu (23.3), Pr (20.0), Hf (20.0), Sc (16.7), Ti (16.7), V (16.7), Tb (16.7), Ta (16.7), Am (16.7), Cm (16.7), Mg (13.3), Sr (13.3), Nb (13.3), Sn (13.3), Nd (13.3), Sm (13.3), Gd (13.3), Dy (13.3), Yb (13.3), Ca (10.0), Cr (10.0), Zn (10.0), Y (10.0), Ba (10.0), La (10.0), Eu (10.0), Ho (10.0), Er (10.0), Tm (10.0), Lu (10.0), Bk (10.0), Si (6.7), Fe (6.7), Ni (6.7), Mo (6.7), Cd(6.7), Pb (6.7), Cf (6.7)",
        "U → Th (13.7), Ce (9.4), Nd (8.8), La (8.4), Pr (8.2), Tb (8.1), Np (8.1), Gd (7.8), Dy (7.6), Ho (7.6), Y (7.5), Sm (7.4), Er (7.3), Yb (7.0), Tm (6.7), Pu (6.7), Zr (6.5), Lu (5.5)",
        "Np → U (49.4), Pr (35.3), Pu (35.3), Ce (34.1), La (33.5), Nd (32.4), Th (30.0), Dy (27.6), Tb (27.1), Sm (26.5), Gd (25.9), Ho (25.9), Er (25.9), Y (25.3), Yb (22.4), Tm (21.8), Lu (20.6), Sc (15.9), Eu (15.9), Zr (14.7), Am (14.7), Ca (10.6), Cm (10.6), Sr (9.4), Hf (8.2), Ti (6.5), Ba (6.5), Nb (5.9), Pa (5.9), Mg (5.3)",
        "Pu → Ce (40.2), U (36.5), Nd (35.4), Pr (34.9), La (33.3), Gd (32.3), Np (31.7), Y (31.2), Tb (30.7), Dy (30.7), Er (30.2), Sm (29.6), Ho (29.1), Th (28.6), Yb (28.0), Tm (24.9), Lu (23.8), Sc (19.0), Zr (17.5), Eu (16.9), Am (15.9), Hf (12.7), Ca (11.1), Cm (11.1),Sr (9.5), Ti (7.9), Ba (7.4), V (6.9), Nb (6.9), Mg (6.3), Ta (5.8), In (5.3), Bk (5.3)",
        "Am → Pu (63.8), La (61.7), Ce (61.7), Nd (61.7), Pr (55.3), Np (53.2), Gd (48.9), Tb (48.9), Dy (46.8), Ho (46.8), Sm (44.7), Yb (44.7), U (44.7), Y (42.6), Cm (42.6), Er (40.4), Eu (38.3), Tm (38.3), Lu (34.0), Sc (27.7), Zr (25.5), Th (25.5), Sr (21.3), Bk (21.3), Ca (17.0), Ba (14.9), Hf (14.9), Ti (12.8), In (12.8), Mg (10.6), Pb (10.6), Pa (10.6), Cf (10.6), V (8.5), Cr (8.5), Mn (8.5), Nb (8.5), Sn (8.5), Bi (8.5), Ac (8.5), Cd (6.4), Ta (6.4)",
        "Cm → Nd (45.1), La (43.1), Ce (43.1), U (43.1), Pu (41.2), Pr (39.2), Am (39.2), Gd (37.3), Np (35.3), Tb (33.3), Ho (33.3), Y (31.4), Dy (31.4), Eu (29.4), Er (29.4), Yb (29.4), Sc (27.5), Tm (27.5), Sm (25.5), Lu (25.5), Th (25.5), Bk (19.6), Sr (17.6), Zr (17.6), Ca (15.7), In (13.7), Ba (13.7), Cf (13.7), Mg (11.8), Ti (9.8), Nb (9.8), Sn (9.8), Pb(9.8), Bi (9.8), Pa (9.8), V (7.8), Cr (7.8), Mn (7.8), Cd (7.8), Hf (7.8), Ta (5.9), Hg (5.9), Ac (5.9)",
        "Bk → Ce (62.5), Pu (62.5), Am (62.5), Cm (62.5), Tb (50.0), Yb (50.0), U (50.0), Np (50.0), Sc (43.8), Y (43.8), La (43.8), Pr (43.8), Nd (43.8), Ho (43.8), Er (43.8), Zr (37.5), Gd (37.5), Dy (37.5), Tm (37.5), Th (37.5), Lu (31.2), Ti (25.0), V (25.0), Cr (25.0), Nb (25.0), Eu (25.0), Sr (18.8), In (18.8), Sm (18.8), Hf (18.8), Ta (18.8), Pa (18.8), Cf (18.8), Mg (12.5), Al (12.5), Ca (12.5), Ba (12.5), Pt (12.5), Bi (12.5), Li (6.2), Be (6.2), Si (6.2), S (6.2), Mn (6.2), Fe (6.2), Co (6.2), Ni (6.2), Cu (6.2), Zn (6.2), Ga (6.2), Se (6.2), Mo (6.2), Pd (6.2), Cd (6.2), Sn (6.2), W (6.2), Pb (6.2), Ac (6.2)",
        "Cf → Ce (52.9), Pr (52.9), La (47.1), Pu (47.1), Nd (41.2), Gd (41.2), Tb (41.2), Cm (41.2), Er (35.3), Y (29.4), Dy (29.4), Ho (29.4), Yb (29.4), Lu (29.4), U (29.4), Np (29.4), Am (29.4), Sc (23.5), Sm (23.5), Eu (23.5), Tm (23.5), Th (23.5), Zr (17.6), Bk (17.6), Ca (11.8), Sr (11.8), In (11.8), Sn (11.8), Ba (11.8), Pb (11.8), Bi (11.8), Ac (11.8), Pa (11.8), Li (5.9), Mg (5.9), Si (5.9), Mn (5.9), Rh (5.9), Cd (5.9), Te (5.9), Pm (5.9), Hf (5.9), Ir (5.9), Hg (5.9), Tl (5.9)",
        "Es →",
        "Fm →",
        "Md →",
        "No →",
        "Lr →"
    ]

    # 建立元素列表
    elements = []
    for text in text_data:
        parts = text.split("→")  # 使用箭头符号 "→" 分割文本，获取元素和替代关系信息
        element = parts[0].strip()  # 获取元素，并去除首尾空格
        elements.append(element)

    # 初始化可替代矩阵 按非百分比初始化，原本是1e-1,按非百分比就是1e-3
    matrix = np.full((len(elements), len(elements)), 1e-3)
    # 填充可替换矩阵 (a,b):a替换b的概率是a的值
    for text in text_data:
        parts = text.split("→")  # 使用箭头符号 "→" 分割文本，获取元素和替代关系信息
        if len(parts) > 1:
            element = parts[0].strip()  # 获取元素，并去除首尾空格
            replacements = parts[1].split(",")  # 获取元素，并去除首尾空格
            for replacement in replacements:  # 替换表中一行的元素
                element_info = replacement.strip().split("(")  # 将每个替代元素的信息分割为元素和替代程度部分
                if len(element_info) > 1:  # 如果一个元素有可替代的
                    replacement_element = element_info[0].strip()  # 获取替代元素，并去除首尾空格
                    replacement_percentage = float(
                        element_info[1].strip().rstrip(')')) / 100  # 获取替代程度，并将其转换为浮点数,之前是百分比形式，/100
                    i = elements.index(element)  # 查找元素在列表中的索引位置
                    j = elements.index(replacement_element)  # 查找元素在列表中的索引位置
                    matrix[i, j] = replacement_percentage  # 在矩阵中存储替代程度信息

    for i in range(len(matrix)):
        matrix[i][i] = 1
    return matrix


def floyd_warshall_multiplication(graph):  # 乘法
    """
    :param graph: 二维数组，表示图的距离矩阵
    :return: 二维数组，表示每对节点之间的最短路径长度
    """
    # 初始化最短路径长度矩阵 将自己和自己的距离设为1，这样想乘没有影响
    dist = [[1 if i == j else graph[i][j] for j in range(len(graph))] for i in range(len(graph))]
    # 更新点之间的最短路径
    for k in range(len(graph)):  # k表示ij之间插的点的个数
        for i in range(len(graph)):
            for j in range(len(graph)):
                dist[i][j] = min(dist[i][j], dist[i][k] * dist[k][j])
    return dist


# 向量a和b，计算欧氏距离。
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def symmetry_smaller_(matrix):  # 对称、取小的矩阵
    size = len(matrix)
    for i in range(size):
        for j in range(i+1, size):  # 将 [i][j] 和 [j][i] 的值设置为两者之中的最小值
            min_val = min(matrix[i][j], matrix[j][i])
            matrix[i][j] = min_val
            matrix[j][i] = min_val

    return matrix


def main_MDS(dimension, n_init=20000, random_state=42, max_iter=500):
    # dim=1,n_init=20w,random_state=5,max_iter=500
    # 获取按非百分比形式的可替代矩阵
    orig_similarity_matrix = load_similar_matrix_100()
    # reciprocal函数计算每个元素的倒数
    inv_s = np.reciprocal(orig_similarity_matrix)
    # 最短距离替换矩阵
    shortest_inx_s = floyd_warshall_multiplication(inv_s)
    # 将自己到自己的距离赋为0
    for i in range(len(shortest_inx_s)):
        shortest_inx_s[i][i] = 0
    distance_matrix = symmetry_smaller_(shortest_inx_s)  # 转化为对称的距离矩阵

    mds = MDS(n_components=dimension, dissimilarity='precomputed', n_init=n_init, max_iter=max_iter, random_state=random_state, n_jobs=24, normalized_stress='auto')
    coordinates = mds.fit_transform(distance_matrix)  # 点

    # 还原
    n = coordinates.shape[0]
    new_adjacency_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                new_adjacency_matrix[i, j] = euclidean_distance(coordinates[i], coordinates[j])
            else:
                new_adjacency_matrix[i, j] = 0
    # 计算两者之间loss
    mae = np.mean(np.abs(distance_matrix - new_adjacency_matrix))

    # 获取原子序数为1到103的元素符号
    elements = [Element.from_Z(atomic_number).symbol for atomic_number in range(1, len(coordinates) + 1)]

    element_coordinates = {element: coordinate for element, coordinate in zip(elements, coordinates)}

    # 获取元素和坐标的元组列表
    element_tuples = [(element, coordinate) for element, coordinate in element_coordinates.items()]

    # 将元素坐标转换为字典，键为元素符号，值为坐标数组
    element_dict = {element: list(coordinate) for element, coordinate in element_tuples}

    # 将元素符号和坐标数组分别存储在两个列表中
    elements = list(element_dict.keys())
    coordinates = list(element_dict.values())

    # 将元素符号和坐标数组写入CSV文件中
    with open('MDS_'+str(dimension)+'dim.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Element', 'Coordinates'])
        for i in range(len(elements)):
            writer.writerow([elements[i], coordinates[i]])
    return coordinates, mae


if __name__ == '__main__':
    # 生成不同维度MDS降维结果，并保存在csv文件中
    dim_list = [1, 2, 4, 8, 16, 32, 64]
    for i in dim_list:
        MDS_ELEMENT_result, mae = main_MDS(dimension=i)
        print("dim = ", i, " mae = ", mae)
