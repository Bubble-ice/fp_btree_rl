"""Note
这份代码训练的模型不能收敛，但是代码结构比较完整

"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

ROOT_PATH = Path(__file__).parent.parent


class GraphDataset(Dataset):
    def __init__(self, features, areas, wirelengths, max_nodes=1000):
        self.features = features
        self.areas = areas
        self.wirelengths = wirelengths
        self.max_nodes = max_nodes

        # 预处理wirelengths
        self.wirelengths = self.minmax_scale(torch.as_tensor(wirelengths)).numpy()

    def __len__(self):
        return len(self.areas)

    def __getitem__(self, index):
        x = torch.as_tensor(self.features[index], dtype=torch.float32)
        n = x.shape[0]

        # 填充到max_nodes x (max_nodes + 8)
        pad_cols = self.max_nodes + 8 - x.shape[1]
        pad_rows = self.max_nodes - n

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

    @staticmethod
    def minmax_scale(x: torch.Tensor, feature_range=(-1, 1)) -> torch.Tensor:
        """将输入张量线性缩放到指定范围（默认[-1, 1]）"""
        min_val, max_val = x.min(), x.max()
        # 避免除以0
        if max_val - min_val < 1e-6:
            return torch.zeros_like(x)
        x_std = (x - min_val) / (max_val - min_val)  # 先缩放到[0, 1]
        x_scaled = (
            x_std * (feature_range[1] - feature_range[0]) + feature_range[0]
        )  # 再映射到目标范围
        return x_scaled


class GraphFeaturePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # MLP1
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 8),
            nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_dim // 8, hidden_dim // 16),
            nn.ReLU(),
            nn.Linear(hidden_dim // 16, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        max_nodes = x.shape[1]
        input_dim = x.shape[2]

        x_reshaped = x.reshape(-1, input_dim)
        feat = self.mlp1(x_reshaped)
        feat = feat.reshape(batch_size, max_nodes, -1)

        pooled_feat = feat.mean(dim=1)  # 全局平均池化
        prediction = self.mlp2(pooled_feat)
        return prediction.squeeze(-1)


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # 训练阶段
        for x, _, wl in train_loader:
            x, wl = x.to(device), wl.to(device)

            optimizer.zero_grad()
            pred_wl = model(x)

            loss = criterion(pred_wl, wl)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        # 验证阶段
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, _, wl in val_loader:
                x, wl = x.to(device), wl.to(device)
                pred_wl = model(x)
                loss = criterion(pred_wl, wl)
                val_loss += loss.item() * x.size(0)

        # 计算平均损失
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        # 打印训练信息
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ROOT_PATH / "best_model.pth")

    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据
    data = np.load(ROOT_PATH / "dataset" / "fplan_dataset.npz", allow_pickle=True)

    # 创建数据集
    full_dataset = GraphDataset(
        features=data["features"],
        areas=data["areas"],
        wirelengths=data["wirelengths"],
        max_nodes=1000,
    )

    # 划分训练集和验证集 (80%训练, 20%验证)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # 初始化模型
    model = GraphFeaturePredictor(input_dim=1008, hidden_dim=512).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    trained_model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=50,
        device=device,
    )
