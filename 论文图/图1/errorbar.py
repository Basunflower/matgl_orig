import csv
import numpy as np
import matplotlib.pyplot as plt


def get_mae_from_vsv(file_path="different_dim_mae.csv"):
    data = {}
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            dim = row[0]
            mae_values = [float(value) for value in row[1:]]
            data[dim] = mae_values

    return data


def draw_errorbar(mae_data):
    # 计算平均值和标准差
    means = [np.mean(values) for key, values in mae_data.items()]
    stds = [np.std(values) for key, values in mae_data.items()]

    # 绘图
    x = np.arange(len(mae_data))
    plt.errorbar(0, means[0], yerr=stds[0], fmt='o', capsize=4, markersize=6, color=(144 / 255, 188 / 255, 216 / 255),
                 linewidth=2.5, label='5-Fold Cross-Validation Mean MAE')
    for i, key in enumerate(mae_data.keys()):
        plt.errorbar(i, means[i], yerr=stds[i], fmt='o', capsize=4, markersize=6, color=(144/255, 188/255, 216/255), linewidth=2.50)
        plt.text(i, means[i]+stds[i], f"{means[i]:.4f}", ha='center', va='bottom', fontweight='bold')  # 输出中心点的值

    # 榜一虚线
    plt.axhline(y=0.0269, color='gray', linestyle='--', label='Leaderboard Model MAE')

    plt.xticks(x, mae_data.keys())
    plt.xlim(-0.45, len(mae_data)-0.6)
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.title('MAE with Different Embedding Dimensions')
    plt.legend()
    plt.savefig("Embedding_MAE.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()


def main():
    mae_data = get_mae_from_vsv("different_dim_mae.csv")
    draw_errorbar(mae_data)
    # # 不同dim平均mae
    # averages = {key: np.mean(values) for key, values in mae_data.items()}
    # for key, value in averages.items():
    #     print(f"{key}: {value}")


if __name__ == '__main__':
    main()

