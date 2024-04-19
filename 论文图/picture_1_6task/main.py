import csv

import numpy as np
from matplotlib import pyplot as plt


def draw_errorbar(mae_data_list, benchemark, task_name):
    num_rows = 2
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    for i, mae_data in enumerate(mae_data_list):
        ax = axes[i // num_cols, i % num_cols]
        # 计算平均值和标准差
        means = [np.mean(values) for key, values in mae_data.items()]
        stds = [np.std(values) for key, values in mae_data.items()]
        # 绘图
        x = np.arange(len(mae_data))
        # ax.errorbar(0, means[0], yerr=stds[0], fmt='o', capsize=4, markersize=6,
        #             color=(144 / 255, 188 / 255, 216 / 255),
        #             linewidth=2.5, label='5-Fold Cross-Validation Mean MAE')
        for j, key in enumerate(mae_data.keys()):
            ax.errorbar(j, means[j], yerr=stds[j], fmt='o', capsize=4, markersize=6,
                        color=(144 / 255, 188 / 255, 216 / 255), linewidth=2.50)
            ax.text(j, means[j] + stds[j], f"{means[j]:.4f}", ha='center', va='bottom', fontweight='bold')  # 输出中心点的值

        # 榜一虚线
        ax.axhline(y=benchemark[i], color='gray', linestyle='--')  # , label='Leaderboard Model MAE')
        ax.set_xticks(x)
        ax.set_xticklabels(mae_data.keys())
        ax.set_xlim(-0.5, len(mae_data) - 0.5)
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Mean Absolute Error (MAE)')
        ax.set_title(task_name[i])
        # ax.xticks(x, mae_data.keys())
        # ax.xlim(-0.5, len(mae_data) - 0.5)
        # ax.xlabel('Embedding Dimension')
        # ax.ylabel('Mean Absolute Error (MAE)')
        # ax.title('MAE with Different Embedding Dimensions')

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.legend()
    # plt.savefig("matbench_jdft2d.pdf", bbox_inches='tight', pad_inches=0)
    plt.suptitle('MEGNet For 6 Task In Matbench', fontsize=16)
    plt.savefig("MEGNet_MDS.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.clf()


def modified_csv(csv_path):
    # grep "test_MAE" matbench_orig_6task.log | awk '{printf "%s,",$4; if(NR%5==0) printf "\n"} END {print ""}'
    """
    传入获取的csv，对它按照要求修改格式
    改为dimx，mae1，mae2，mae3，mae4，mae5
    """
    # 定义要插入的keys和对应的数据列数
    keys = ['dim1', 'dim2', 'dim4', 'dim8', 'dim16', 'dim32', 'dim64', 'dim128']

    # 读取原始CSV文件
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    # 删除每行的最后的逗号
    cleaned_data = [row[:-1] for row in data]

    # 在第一列插入对应的key
    for i in range(len(cleaned_data)):
        key_index = i // 6 % len(keys)  # 计算当前所在的key索引
        cleaned_data[i].insert(0, keys[key_index])
    return cleaned_data


def process_cleaned_data(cleaned_data, task):
    """
    根据任务和清理后的数据处理数据，并返回新的字典
    参数：
    - cleaned_data: 清理后的数据列表
    - task: 第一行数据在清理后的数据列表中的索引（从0开始）
    返回：
    - 新的字典，包含处理后的数据
    """
    processed_dict = {}

    for i in range(task, len(cleaned_data), 6):  # 以间隔为6读取
        dim = cleaned_data[i][0]  # 获取维度信息
        mae_values = [float(val) for val in cleaned_data[i][1:]]  # 获取对应维度的MAE值列表
        processed_dict[dim] = mae_values

    return processed_dict


if __name__ == '__main__':
    # "matbench_jdft2d",  # 636 33.1918
    # "matbench_phonons",  # 1,265 28.7606
    # "matbench_dielectric",  # 4,764 0.2711
    # "matbench_log_gvrh",  # 10,987 0.0670
    # "matbench_log_kvrh",  # 10,987 0.0491
    # "matbench_perovskites"  # 1w8 0.0269

    csv_file1 = 'Embedding.csv'
    csv_file2 = 'MDS.csv'

    mae_data = modified_csv(csv_file2)
    task1 = process_cleaned_data(mae_data, 0)
    task2 = process_cleaned_data(mae_data, 1)
    task3 = process_cleaned_data(mae_data, 2)
    task4 = process_cleaned_data(mae_data, 3)
    task5 = process_cleaned_data(mae_data, 4)
    task6 = process_cleaned_data(mae_data, 5)

    mae_data_list = [task1, task2, task3, task4, task5, task6]
    benchemark_list = [33.1918, 28.7606, 0.2711, 0.0670, 0.0491, 0.0269]
    task_name = ["matbench_jdft2d", "matbench_phonons", "matbench_dielectric", "matbench_log_gvrh", "matbench_log_kvrh",
                 "matbench_perovskites"]
    draw_errorbar(mae_data_list, benchemark_list, task_name)
