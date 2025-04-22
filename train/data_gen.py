""" Note
    简单的数据集生成脚本
"""

import os
import sys
from typing import Tuple
from pathlib import Path
import argparse

import numpy as np
import numpy.typing as npt
# from objprint import op
import scipy.linalg as la

ROOT_PATH = Path(__file__).parent.parent
# 将fp_btree所在目录添加到Python路径
sys.path.append(str(ROOT_PATH))  # 假设fp_btree在上级目录
import fp_btree  # noqa: E402


def preprocess_data(
    bf: fp_btree.B_Tree_Ext,
) -> Tuple[npt.NDArray[np.float32], float, int]:
    adj = bf.get_adj_matrix()
    eigenvalues, U = la.eigh(adj)
    Lambda = np.diag(eigenvalues)
    U_Lambda = U @ Lambda

    X_prime = np.concatenate([bf.get_pin_nodes_info(), U_Lambda], axis=1)

    return X_prime, bf.get_area(), bf.get_wire_length()


def main(data_folder, num_of_data: int, output_path: str | Path) -> None:
    file_list = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if not os.path.splitext(f)[1]
    ]

    assert file_list, "folder not exists."

    all_features = []
    all_areas = []
    all_wirelengths = []

    num_of_each = num_of_data // len(file_list)
    for f in file_list:
        assert os.path.exists(f), "file not exists."
        bf = fp_btree.B_Tree_Ext(f, 0.5)

        for i in range(num_of_each):
            if i % int(num_of_each / 4) == 0:
                print(f"{i / num_of_each * 100}%")

            bf.update()

            X, area, wl = preprocess_data(bf)
            all_features.append(X)
            all_areas.append(area)
            all_wirelengths.append(wl)
        # 模拟退火获得相对最佳优化
        fp_btree.run_with_sa(bf)
        X, area, wl = preprocess_data(bf)
        all_features.append(X)
        all_areas.append(area)
        all_wirelengths.append(wl)
    
    # 给文件后面加数字
    if os.path.exists(output_path):
        idx = 1
        new_out_path = output_path
        while os.path.exists(new_out_path):
            new_out_path = ROOT_PATH / "dataset" / f"_{idx}".join(os.path.splitext(os.path.basename(out_path)))
            idx += 1
        output_path = new_out_path
    
    np.savez_compressed(
        file=output_path,
        features=np.array(all_features, dtype=object),
        areas=np.array(all_areas),
        wirelengths=np.array(all_wirelengths),
    )
    print(f"dataset保存至{output_path}")
    


if __name__ == "__main__":
    # 文件运行参数
    parser = argparse.ArgumentParser(description="数据集生成")
    parser.add_argument(
        "--output", "-o", type=str, default="fplan_dataset.npz", help="输出文件名"
    )
    args = parser.parse_args()

    # 输入文件夹、输出路径
    raw_data_folder = ROOT_PATH / "raw_data"
    out_path = ROOT_PATH / "dataset" / args.output

    main(raw_data_folder, 2000, out_path)
