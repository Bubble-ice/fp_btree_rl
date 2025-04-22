""" Note
    旧的数据集生成脚本
    尝试使用稀疏矩阵
"""

import os
from typing import Tuple

import numpy as np
import numpy.typing as npt
from objprint import op
import scipy.linalg as la
import scipy.sparse.linalg as spla
import scipy.sparse as sparse
import fp_btree


def normalize_adjacency(adj, is_sparse=False):
    """对称归一化邻接矩阵 (D^(-1/2) A (D^(-1/2))

    Args:
        adj: 邻接矩阵 (稠密np.ndarray或稀疏scipy.sparse.csr/csc_matrix)
        is_sparse: 是否为稀疏矩阵

    Returns:
        归一化后的邻接矩阵 (保持输入类型)
    """
    if is_sparse:
        # 稀疏矩阵版本
        degrees = np.array(adj.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1  # 避免除零
        D_sqrt_inv = sparse.diags(1.0 / np.sqrt(degrees))
        return D_sqrt_inv @ adj @ D_sqrt_inv
    else:
        # 稠密矩阵版本
        degrees = np.sum(adj, axis=1)
        degrees[degrees == 0] = 1
        D_sqrt_inv = np.diag(1.0 / np.sqrt(degrees))
        return D_sqrt_inv @ adj @ D_sqrt_inv


def extract_spectral_features(adj, is_sparse, k=10):
    """提取混合频谱特征"""
    if is_sparse:
        # 高频特征（最大特征值）
        eigenvalues_hi, U_hi = spla.eigsh(adj, k=k, which="LM")
        # 低频特征（最小特征值）
        eigenvalues_lo, U_lo = spla.eigsh(adj, k=k, which="SM")
    else:
        eigenvalues, U = la.eigh(adj)
        U_hi, U_lo = U[:, -k:], U[:, :k]  # 后k列为高频，前k列为低频
        eigenvalues_hi, eigenvalues_lo = eigenvalues[-k:], eigenvalues[:k]

    # 拼接特征并加权
    hi_feat = U_hi @ np.diag(np.log1p(np.abs(eigenvalues_hi)))  # 对数缩放
    lo_feat = U_lo @ np.diag(np.log1p(np.abs(eigenvalues_lo)))
    return np.concatenate([hi_feat, lo_feat], axis=1)


def preprocess_data(
    bf: fp_btree.B_Tree_Ext, is_sparse: bool = False
) -> Tuple[npt.NDArray[np.float32], float, int]:
    """预处理PCB布局数据，提取图结构特征和全局指标

    Args:
        bf: B_Tree_Ext实例，包含PCB布局数据

    Returns:
        tuple: (节点特征矩阵, 布局面积, 总线长)
    """
    # 获取引脚节点基础信息 (形状: N x 8)
    pn_info = bf.get_pin_nodes_info()

    # 获取邻接矩阵 (形状: N x N)
    if is_sparse:
        adj_zip = bf.get_adj_matrix_zip()
        adj_zip = normalize_adjacency(adj_zip, True)
        spectral_feat = extract_spectral_features(adj_zip, is_sparse, k=10)
        # op(eigenvalues)
    else:
        adj = bf.get_adj_matrix()
        adj = normalize_adjacency(adj, False)
        eigenvalues, U = la.eigh(adj)
        spectral_feat = extract_spectral_features(adj, is_sparse, k=10)
        # op(eigenvalues)
    # # 邻接矩阵特征分解: A = U Λ U^T
    # Lambda = np.diag(eigenvalues)  # 特征值对角矩阵
    # U_Lambda = U @ Lambda  # 边特征表示

    X_prime = np.concatenate([pn_info, spectral_feat], axis=1)  # 合并节点特征和边特征

    # 提取布局全局指标
    Area = bf.get_area()  # 布局总面积
    WireLength = bf.get_wire_length()  # 总线长

    return X_prime, Area, WireLength


def main():
    data_folder = os.path.abspath("../data/")  # 转为绝对路径
    file_list = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if not os.path.splitext(f)[1]
    ]

    all_features = []
    all_areas = []
    all_wirelengths = []

    for f in file_list:
        bf = fp_btree.B_Tree_Ext(f, 0.5)
        for i in range(200):
            if i % 25 == 0:
                print(f"{i / 10}%")
            bf.update()
            X, area, wl = preprocess_data(bf, True)

            all_features.append(X)
            all_areas.append(area)
            all_wirelengths.append(wl)

    padded_features = np.array(all_features, dtype=object)

    np.savez_compressed(
        "fplan_dataset4.npz",
        features=padded_features,
        areas=np.array(all_areas),
        wirelengths=np.array(all_wirelengths),
    )


if __name__ == "__main__":
    main()
