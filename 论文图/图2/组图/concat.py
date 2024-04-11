import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 五张图片的路径列表
image_paths = ['element_2d.png', 'Isomap_Embedding.png', 'Multidimensional_scaling.png', 'Spectral_embedding_for_non-linear_dimensionality_reduction.png', 'T-distributed_Stochastic.png']
# 创建一个figure对象和一个包含五个subplot的axes对象数组
fig, axes = plt.subplots(1, 5, figsize=(15, 3))  # 设置figure的大小为(15, 3)，表示宽度为15英寸，高度为3英寸

# 在每个subplot中显示对应的图片
for i, ax in enumerate(axes):
    # 读取图片
    img = mpimg.imread(image_paths[i])
    # 显示图片
    ax.imshow(img)
    # 隐藏坐标轴
    ax.axis('off')

# 调整subplot之间的间距
plt.tight_layout()
plt.savefig('mds_embedding.pdf', bbox_inches='tight', pad_inches=0)
# 显示图形
plt.show()