
class Config:
   
    input_dim = 6         # 每个节点的特征维度
    hidden_dim = 256      # 模型隐藏层维度
    heuristic_dim = 0     # (可选) 启发式方法产生的特征维度
    learning_rate = 0.0005 #5e-4 #
    epochs = 30
    batch_size = 64
    window_size = 3       # 定义“邻近”的窗口大小
    num_neg_samples = 6   # 每个正样本对应生成的负样本数量
    test_size = 0.3       # 划分训练集和验证集的比例
    num_samples = 20000    # 采样数据数量
    root_path = "./data/processed_container_data2.pkl"
    save_path = "./discriminator/model/discriminator.pth"