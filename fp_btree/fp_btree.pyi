from typing import Literal, TypedDict
import numpy as np
import numpy.typing as npt
import scipy

class ActionType:
    ROTATE_FLIP: Literal[0]  # 旋转 + 可能翻转
    SWAP_NODES: Literal[1]  # 交换两个节点
    DELETE_INSERT: Literal[2]  # 删除并重新插入

class Action:
    type: ActionType
    node1: int  # 主节点索引
    node2: int  # 第二节点索引（默认-1）
    flip: bool  # 是否翻转（仅ROTATE_FLIP有效）

class B_Tree_Ext:
    def __init__(self, filename: str = "", calpha: float = 1.0) -> None: ...
    # show tree
    def show(self) -> None: ...
    # 文件名
    def get_filename(self) -> str: ...

    # 模块累加面积
    def get_total_area(self) -> float: ...
    # 布局占地面积
    def get_area(self) -> float: ...
    # 总线长
    def get_wire_length(self) -> int: ...

    # 引脚节点信息列表
    def get_pin_nodes_info(self) -> npt.NDArray[np.float32]: ...
    # 引脚节点的邻接矩阵
    def get_adj_matrix(self) -> npt.NDArray[np.float32]: ...
    def get_adj_matrix_zip(self) -> scipy.sparse._csc.csc_matrix: ...

    # 核心函数，更新布局
    def update(self) -> None: ...

class SAResult(TypedDict):
    """模拟退火算法结果结构

    Attributes:
        cpu_time: 总CPU时间(秒)
        last_cpu_time: 最后阶段CPU时间(秒)
        cost: 布局成本
        area: 芯片面积
        wire_length: 线长
        dead_space: 死区比例
    """

    cpu_time: float
    last_cpu_time: float
    cost: float
    area: float
    wire_length: float
    dead_space: float

def run_with_sa(
    fp: B_Tree_Ext,
    times: int = 400,
    local: int = 7,
    init_temp: float = 0.9,
    term_temp: float = 0.1,
    alpha: float = 1.0,
    outfile: str = "",
    is_debug: bool = False,
) -> SAResult:
    """运行模拟退火算法进行芯片布局

    Args:
        fp: B*树对象
        times: 迭代次数 (默认400)
        local: 局部搜索参数 (默认7)
        init_temp: 初始温度 (默认0.9)
        term_temp: 终止温度 (默认0.1)
        alpha: 权重系数 (默认1.0)
        outfile: 输出文件路径 (可选)

    Returns:
        包含优化结果的结构体

    Raises:
        FileNotFoundError: 当输入文件不存在时
        RuntimeError: 算法执行失败时
    """
    ...

class FplanEnv:
    state_dim: int = 6
    # 初始化和重置
    def __init__(self, fn: str = "", calpha: float = 1.0, max_times=5000) -> None: ...
    def reset(self) -> None: ...
    # 生成批量动作
    def act_gen_batch(self, num=1) -> list[Action]: ...
    # 步进和还原
    def step(self, act_bool: bool) -> tuple[npt.NDArray[np.float64], float, bool]: ...
    def recover(self) -> None: ...
    # 工具函数
    def get_cost(self) -> float: ...
    def get_init_cost(self) -> float: ...
    def get_baseline(self) -> float: ...
    def show_info(self) -> None: ...
