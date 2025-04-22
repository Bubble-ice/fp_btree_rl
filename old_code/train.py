import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from objprint import op


class GraphDataset(Dataset):
    def __init__(self, features, areas, wirelengths, max_nodes=1000):
        self.features = features
        self.areas = areas
        self.wirelengths = wirelengths
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.areas)

    def __getitem__(self, index):
        x = torch.as_tensor(self.features[index], dtype=torch.float32)
        n = x.shape[0]

        # 计算需要填充的行数和列数
        pad_rows = self.max_nodes - n
        pad_cols = self.max_nodes - n + 8

        if pad_cols > 0:
            x = torch.cat([x, torch.zeros((n, pad_cols), dtype=torch.float32)], dim=1)

        if pad_rows > 0:
            x = torch.cat(
                [x, torch.zeros((pad_rows, self.max_nodes + 8), dtype=torch.float32)],
                dim=0,
            )

        area = torch.as_tensor(self.areas[index], dtype=torch.float32)
        wl = torch.as_tensor(self.wirelengths[index], dtype=torch.float32)
        return x, area, wl


class DimensionTransform(nn.Module):
    def __init__(self, max_nodes=10, input_feat=16, output_feat=32):
        super().__init__()
        self.max_nodes = max_nodes
        self.output_feat = output_feat

        # 更完整的维度变换模块
        self.node_encoder = nn.Sequential(
            nn.Linear(input_feat, 64), nn.ReLU(), nn.Linear(64, output_feat)
        )

        # 全局信息提取
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        encoded = self.node_encoder(x)

        if len(x) < self.max_nodes:
            global_feat = self.global_pool(
                encoded.unsqueeze(0)
            ).squeeze()  # (output_feat,)
            padding = global_feat.repeat(self.max_nodes - len(x), 1)
            encoded = torch.cat([encoded, padding], dim=0)

        return encoded


class GraphFeatureExtractor(nn.Module):
    def __init__(self, input_dim=1008, hidden_dim=512):
        super().__init__()

        # MLP1: 特征提取网络
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 8),
            nn.ReLU(),
        )

        # MLP2: 线长预测网络
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim // 8, hidden_dim // 16),
            nn.ReLU(),
            nn.Linear(hidden_dim // 16, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        extracted = self.mlp1(x)
        pooled = extracted.mean(dim=0)
        prediction = self.mlp2(pooled)
        return prediction.squeeze()


# 训练设置
def train():
    data = np.load("fplan_dataset3.npz", allow_pickle=True)
    features = data["features"]
    areas = data["areas"]
    wirelengths = data["wirelengths"]

    # 创建数据集
    dataset = GraphDataset(features, areas, wirelengths)
    dataloader = DataLoader(dataset, shuffle=True)

    # 初始化模型
    model = GraphFeatureExtractor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 20
    for epoch in range(num_epochs):
        for batch_x, area, wl in dataloader:
            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, wl)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("训练完成!")
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train()
