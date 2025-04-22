""" Note
    针对ai_train2.py训练的模型的推理测试
"""
from pathlib import Path

import torch

from ..fp_btree import B_Tree_Ext
from data_gen import preprocess_data
from ai_train2 import GraphFeaturePredictor

ROOT_PATH = Path(__file__).parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inverse_minmax_scale(x_scaled, original_min, original_max, feature_range=(-1, 1)):
    """
    反归一化函数公式：
    $x = \frac{(x_{scaled} - feature_{min}) \times (original_{max} - original_{min})}{feature_{range}} + original_{min}$
    """
    feature_min, feature_max = feature_range
    # 先缩放到[0,1]范围
    x_std = (x_scaled - feature_min) / (feature_max - feature_min)
    # 再还原到原始范围
    return x_std * (original_max - original_min) + original_min

def resize_x(x):
    n = x.shape[0]
    
    # 填充到固定尺寸
    x_padded = torch.zeros((1000, 1008))
    x_padded[:n, :x.shape[1]] = x

    x_padded = x_padded.to(device)
    x_padded = x_padded.unsqueeze(0)
    return x_padded

def eval(model, bt):
    model.eval()
    x, a, w = preprocess_data(bt)

    x = torch.FloatTensor(x)
    
    pre_wl = model(resize_x(x))

    print(f"{pre_wl=}")

    annorm_pre_wl = inverse_minmax_scale(pre_wl, 138999, 4719909)

    pre = float(annorm_pre_wl)

    print(f"{pre=}, {w=}\n {((abs(pre - w) / w)*100):.2f}%")

def main():
    model = GraphFeaturePredictor().to(device)
    model.load_state_dict(torch.load(ROOT_PATH / "model" / "best_model250407-22-50.pth"))

    raw_fn = ROOT_PATH / "raw_data" / "ami49"
    bt = B_Tree_Ext(str(raw_fn), 0.5)

    for _ in range(10):
        bt.update()
        eval(model, bt)


if __name__ == "__main__":
    main()