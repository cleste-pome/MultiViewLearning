import torch


def normalized_probability_determinacy(probs):
    """
    计算基于归一化概率的确定性度量。

    参数：
    probs (Tensor): 形状为 (N, K) 的张量，其中 N 是样本数量，K 是类别数量。
                    每个元素为样本对应类别的预测概率。

    返回：
    Tensor: 形状为 (N,) 的张量，表示每个样本的确定性度量。
    """
    K = probs.size(1)  # 获取类别数量 K
    uniform_dist = 1.0 / K  # 均匀分布概率，即每个类别的理想概率

    # 计算不确定性 U，取每个样本的类别概率与均匀分布的绝对差值的平均
    U = torch.mean(torch.abs(probs - uniform_dist), dim=1)
    D = 1 - U  # 计算确定性 D，D 越高表示模型对某个类别的信心越强
    return D


def combined_determinacy(probs, epsilon=1e-10):
    """
    计算综合确定性度量，结合最大概率和熵。

    参数：
    probs (Tensor): 形状为 (N, K) 的张量，其中 N 是样本数量，K 是类别数量。
                    每个元素为样本对应类别的预测概率。
    epsilon (float): 防止计算熵时出现除以零的微小值。

    返回：
    Tensor: 形状为 (N,) 的张量，表示每个样本的综合确定性度量。
    """
    # 找到每个样本的最大概率及其对应的索引
    p_max, _ = torch.max(probs, dim=1)

    # 计算熵，熵越高表示不确定性越高
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)

    # 计算综合确定性 D，结合最大概率和熵
    D = p_max * (1 + 1 / (entropy + epsilon))  # 熵越小，D 越高
    return D


if __name__ == "__main__":
    # 示例：模拟概率输出
    # 假设有 3 个样本，3 个类别的预测概率
    probs = torch.tensor([
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.6, 0.2, 0.2],
        [0.3, 0.4, 0.3]
    ])

    # 计算确定性
    determinacy_normalized = normalized_probability_determinacy(probs)
    print("Normalized Probability Determinacy:", determinacy_normalized)

    # 计算综合确定性
    determinacy_combined = combined_determinacy(probs)
    print("Combined Determinacy:", determinacy_combined)
