from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

ROOT_PATH = Path(__file__).parent.parent


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
        feat = self.mlp1(x)
        pooled_feat = feat.mean(dim=0)
        prediction = self.mlp2(pooled_feat)
        return prediction.squeeze()


def train(data, model, criterion, optimizer, num_epochs=50): 
    feats = data["features"]
    areas = data["areas"]
    wls = data["wirelengths"]

    norm_wls = minmax_scale(torch.as_tensor(wls)).to(device)

    
    model = model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        model.train()

        running_loss = 0.0
        for x, ar, wl in zip(feats, areas, norm_wls):
            x = resize(torch.as_tensor(x), 1000).to(device)
            ar = torch.as_tensor(ar).to(device)
            wl = torch.as_tensor(wl).to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                pred_wl = model(x)
                loss = criterion(pred_wl, wl)
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(feats)

        print(f"{epoch}/{num_epochs}, loss:{epoch_loss}")
    return model



def resize(x, max_nodes):
    n = x.shape[0]
    # 计算需要填充的行数和列数
    pad_rows = max_nodes - n
    pad_cols = max_nodes - n

    if pad_cols > 0:
        x = torch.cat([x, torch.zeros((n, pad_cols), dtype=torch.float32)], dim=1)

    if pad_rows > 0:
        x = torch.cat(
            [x, torch.zeros((pad_rows, max_nodes + 8), dtype=torch.float32)],
            dim=0,
        )
    return x

def minmax_scale(x: torch.Tensor, feature_range=(-1, 1)) -> torch.Tensor:
    """将输入张量线性缩放到指定范围（默认[-1, 1]）"""
    min_val, max_val = x.min(), x.max()
    x_std = (x - min_val) / (max_val - min_val)  # 先缩放到[0, 1]
    x_scaled = x_std * (feature_range[1] - feature_range[0]) + feature_range[0]  # 再映射到目标范围
    return x_scaled

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(ROOT_PATH / "dataset" / "fplan_dataset.npz", allow_pickle=True)
    
    model = GraphFeaturePredictor(1008, 512)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(data, model, criterion, optimizer)
