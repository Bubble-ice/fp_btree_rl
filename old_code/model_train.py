import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ----------------------------
# 1. 数据预处理与加载
# ----------------------------
class PCBDataset(Dataset):
    def __init__(self, features, areas, wirelengths, transform=None):
        """
        Args:
            features: List of Nx27 matrices (variable N per sample)
            areas: Array of area values
            wirelengths: Array of wirelength values
            transform: Optional transform to apply
        """
        self.features = features
        self.areas = areas
        self.wirelengths = wirelengths
        self.transform = transform

        # 预处理：对面积和线长取对数（假设它们是指数分布的）
        self.areas = np.log(areas + 1e-8)
        self.wirelengths = np.log(wirelengths + 1e-8)

        # 标准化目标值
        self.area_mean, self.area_std = self.areas.mean(), self.areas.std()
        self.wl_mean, self.wl_std = self.wirelengths.mean(), self.wirelengths.std()

        self.areas = (self.areas - self.area_mean) / self.area_std
        self.wirelengths = (self.wirelengths - self.wl_mean) / self.wl_std

    def __len__(self):
        return len(self.areas)

    def __getitem__(self, idx):
        # 获取特征矩阵并转换为float32
        x = self.features[idx].astype(np.float32)

        # 标准化每个特征维度（27维）
        x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)

        # 全局平均池化：将Nx27 -> 1x27
        x_pooled = x.mean(axis=0)

        area = torch.tensor(self.areas[idx], dtype=torch.float32)
        wl = torch.tensor(self.wirelengths[idx], dtype=torch.float32)

        return x_pooled, area, wl


# ----------------------------
# 2. 模型定义
# ----------------------------
class GraphFeaturePredictor(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=128):
        super(GraphFeaturePredictor, self).__init__()

        # 特征提取网络 (MLP1)
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # 预测头
        self.area_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.wirelength_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        area = self.area_head(features)
        wirelength = self.wirelength_head(features)
        return area.squeeze(), wirelength.squeeze()


# ----------------------------
# 3. 训练与评估
# ----------------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs=50):
    train_loss_history = {"area": [], "wirelength": []}
    val_loss_history = {"area": [], "wirelength": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss_area = 0.0
            running_loss_wl = 0.0

            for inputs, areas, wls in dataloaders[phase]:
                inputs = inputs.to(device)
                areas = areas.to(device)
                wls = wls.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    pred_area, pred_wl = model(inputs)
                    loss_area = criterion(pred_area, areas)
                    loss_wl = criterion(pred_wl, wls)
                    loss = loss_area + loss_wl  # 联合损失

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss_area += loss_area.item() * inputs.size(0)
                running_loss_wl += loss_wl.item() * inputs.size(0)

            epoch_loss_area = running_loss_area / len(dataloaders[phase].dataset)
            epoch_loss_wl = running_loss_wl / len(dataloaders[phase].dataset)

            if phase == "train":
                train_loss_history["area"].append(epoch_loss_area)
                train_loss_history["wirelength"].append(epoch_loss_wl)
            else:
                val_loss_history["area"].append(epoch_loss_area)
                val_loss_history["wirelength"].append(epoch_loss_wl)

            print(
                f"{phase} Area Loss: {epoch_loss_area:.4f}, Wirelength Loss: {epoch_loss_wl:.4f}"
            )

    return model, train_loss_history, val_loss_history


def plot_losses(train_loss, val_loss, title):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# ----------------------------
# 4. 主流程
# ----------------------------
if __name__ == "__main__":
    # 加载数据
    data = np.load("fplan_dataset3.npz", allow_pickle=True)
    features = data["features"]
    areas = data["areas"]
    wirelengths = data["wirelengths"]

    # 划分训练集和验证集
    X_train, X_val, y_area_train, y_area_val, y_wl_train, y_wl_val = train_test_split(
        features, areas, wirelengths, test_size=0.2, random_state=42
    )

    # 创建数据集和数据加载器
    train_dataset = PCBDataset(X_train, y_area_train, y_wl_train)
    val_dataset = PCBDataset(X_val, y_area_val, y_wl_val)

    batch_size = 4
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
    }

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphFeaturePredictor(input_dim=27, hidden_dim=256).to(device)

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 训练模型
    model, train_loss, val_loss = train_model(
        model, dataloaders, criterion, optimizer, num_epochs=50
    )

    # 绘制损失曲线
    plot_losses(train_loss["area"], val_loss["area"], "Area Prediction Loss")
    plot_losses(
        train_loss["wirelength"], val_loss["wirelength"], "Wirelength Prediction Loss"
    )

    # 保存模型
    torch.save(model.state_dict(), "pcb_predictor.pth")
