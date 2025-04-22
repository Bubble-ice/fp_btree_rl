''' Note
    这份代码的模型训练时可以收敛，但测试结果不佳
'''


from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

ROOT_PATH = Path(__file__).parent.parent

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 改进的模型架构
class GraphFeaturePredictor(nn.Module):
    def __init__(self, input_dim=1008, hidden_dim=512):
        super().__init__()
        # 节点特征编码器（带BatchNorm和Dropout）
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.BatchNorm1d(hidden_dim//4),
            nn.LeakyReLU(0.2),
        )
        
        # 全局特征处理器
        self.global_processor = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//8),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim//8, 1),
            nn.Tanh(),
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x形状: [batch_size, max_nodes, input_dim]
        batch_size, max_nodes, _ = x.shape
        
        # 处理节点特征
        x = x.reshape(-1, x.size(-1))
        node_features = self.node_encoder(x)
        node_features = node_features.view(batch_size, max_nodes, -1)
        
        # 全局平均池化
        graph_features = node_features.mean(dim=1)
        
        # 最终预测
        return self.global_processor(graph_features).squeeze(-1)

# 2. 改进的数据集类
class GraphDataset(Dataset):
    def __init__(self, features, areas, wirelengths, max_nodes=1000):
        self.features = features
        self.areas = areas
        self.wirelengths = self._minmax_scale(np.array(wirelengths))
        # self.wirelengths = wirelengths
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.areas)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx])
        n = x.shape[0]
        
        # 填充到固定尺寸
        x_padded = torch.zeros((self.max_nodes, self.max_nodes + 8))
        x_padded[:n, :x.shape[1]] = x
        
        return x_padded, torch.FloatTensor([self.areas[idx]]), torch.FloatTensor([self.wirelengths[idx]])

    @staticmethod
    def _minmax_scale(x, feature_range=(-1, 1)):
        x_min, x_max = x.min(), x.max()
        print(f"{x_min=}", f"{x_max=}")
        exit()
        if x_max - x_min < 1e-6:
            return np.zeros_like(x)
        x_std = (x - x_min) / (x_max - x_min)
        return x_std * (feature_range[1] - feature_range[0]) + feature_range[0]

# 3. 训练和验证函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, _, wl in loader:
        x, wl = x.to(device), wl.to(device).view(-1)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, wl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, _, wl in loader:
            x, wl = x.to(device), wl.to(device).view(-1)
            pred = model(x)
            total_loss += criterion(pred, wl).item() * x.size(0)
    return total_loss / len(loader.dataset)

# 4. 主训练流程
def main(file_path, output_path):
    # 加载数据
    data = np.load(file_path, allow_pickle=True)

    # 创建数据集
    dataset = GraphDataset(
        features=data["features"],
        areas=data["areas"],
        wirelengths=data["wirelengths"]
    )
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    
    # 初始化模型和优化器
    model = GraphFeaturePredictor().to(device)
    criterion = nn.SmoothL1Loss()  # 对异常值更鲁棒
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(50):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        # 打印信息
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} | LR: {lr:.1e} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print("※ New best model saved!")
    
    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    cur_ = datetime.now().strftime("%y%m%d-%H-%M")
    _in = ROOT_PATH / "dataset" / "fplan_dataset_1.npz"
    _out = ROOT_PATH / "model" / f"best_model{cur_}.pth"
    main(_in, _out)