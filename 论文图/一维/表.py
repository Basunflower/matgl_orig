from pymatgen.core import Element
import csv
import pandas as pd

# 读取CSV文件
data = pd.read_csv('MDS_1dim.csv')

# 将 'Coordinates' 列中的字符串转换为浮点数
data['Coordinates'] = data['Coordinates'].apply(lambda x: float(x.strip('[]')))

# 按第二列的值排序
sorted_data = data.sort_values(by='Coordinates')

# 输出排序后的第一列结果
sorted_symbols = sorted_data['Element'].tolist()
print(sorted_symbols)
# 读取已有的元素数据
element_data = []
for atomic_number in range(1, 104):
    element = Element.from_Z(atomic_number)
    element_data.append([atomic_number, element.symbol])

# 添加排序后的符号列表到第三列
for idx, row in enumerate(element_data):
    row.append(sorted_symbols[idx])

# 将数据写入CSV文件
with open('elements.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Number', 'Z', 'MDS'])
    writer.writerows(element_data)
