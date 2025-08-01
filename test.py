import numpy as np
import matplotlib.pyplot as plt

def generate_and_plot_nodes(max_nodes):

    dim = 2
    centers = np.random.rand(2, dim)
        
    # 将max_nodes组数据大致均匀分配到2个中心
    cluster_sizes = np.full(2, max_nodes // 2)
    cluster_sizes[:max_nodes % 2] += 1
    
    # 为每个中心生成数据，添加一些高斯噪声使其集中在中心附近
    node_clusters = []
    for i in range(2):
        # 使用标准差为0.1的高斯噪声，让数据集中在中心附近
        cluster_nodes = np.random.normal(loc=centers[i], scale=0.01, size=(cluster_sizes[i], dim))
        # 确保数据在[0, 1]范围内
        cluster_nodes = np.clip(cluster_nodes, 0, 1)
        node_clusters.append(cluster_nodes)
    
    nodes = np.vstack(node_clusters)
    

    # 绘制二维图，因为数据维度固定为2
    plt.figure(figsize=(8, 6))
    plt.scatter(nodes[:, 0], nodes[:, 1], alpha=0.6)
    plt.title('节点分布')
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.grid(True)
    plt.show()

    return nodes


generate_and_plot_nodes(20)