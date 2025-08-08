import numpy as np


def similarity( x, y, eps=1e-8,pad_value=0.0):
    
    dot_product = np.sum(x * y, axis=-1)  # Shape: (n_traj,)
    
    norm_x = np.linalg.norm(x, axis=-1)  # Shape: (n_traj,)
    norm_y = np.linalg.norm(y, axis=-1)  # Shape: (n_traj,)
            
    pad_mask = (norm_x == 0) | (norm_y == 0)
    sim = np.full_like(norm_x, pad_value)

    valid_mask = ~pad_mask
    if np.any(valid_mask):

        dot_product = np.sum(x[valid_mask] * y[valid_mask], axis=-1)
        cos_sim = dot_product / (norm_x[valid_mask] * norm_y[valid_mask] + eps)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        sim[valid_mask] = (cos_sim + 1) / 2

    return sim

def b_similarity( x, y, eps=1e-8, pad_value=0.0):
    dot_product = np.sum(x * y, axis=-1)  # Shape: [batch, n_traj]
    # 计算 x 和 y 的 L2 范数
    norm_x = np.linalg.norm(x, axis=-1)  # Shape: [batch, n_traj]
    norm_y = np.linalg.norm(y, axis=-1)  # Shape: [batch, n_traj]
    
    # 创建填充掩码，标记 norm_x 或 norm_y 为 0 的位置
    pad_mask = (norm_x == 0) | (norm_y == 0)  # Shape: [batch, n_traj]
    
    # 初始化输出数组，填充 pad_value
    sim = np.full_like(norm_x, pad_value)  # Shape: [batch, n_traj]
    
    # 有效掩码：norm_x 和 norm_y 都不为 0 的位置
    valid_mask = ~pad_mask  # Shape: [batch, n_traj]
    
    # 如果存在有效位置，计算余弦相似度
    if np.any(valid_mask):
        # 提取有效位置的点积和范数
        dot_product_valid = dot_product[valid_mask]  # Shape: [num_valid]
        norm_x_valid = norm_x[valid_mask]  # Shape: [num_valid]
        norm_y_valid = norm_y[valid_mask]  # Shape: [num_valid]
        
        # 计算余弦相似度
        cos_sim = dot_product_valid / (norm_x_valid * norm_y_valid + eps)
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        
        # 映射到 [0, 1]
        sim[valid_mask] = (cos_sim + 1) / 2
    
    return sim

x = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])  # [batch=2, n_traj=3, dim=4]

y = np.array([[[2, 2, 1, 4], [5, 6, 7,6], [9, 10, 11, 12]], [[3, 14, 15, 6], [7, 18, 19, 5], [3, 22, 1, 24]]])  # [batch=2, n_traj=3, dim=4]


sim1 = b_similarity(x, y)
sim2 = similarity(x[0], y[0])

print(sim1)
print(sim2)

