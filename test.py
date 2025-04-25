from pathlib import Path

from objprint import op
from sympy import total_degree

from fp_btree import FplanEnv
import fp_btree

ROOT_PATH = Path(__file__).parent


raw_fn = ROOT_PATH / "raw_data" / "ami33"

env = FplanEnv(str(raw_fn), 1, 10000, True)
# state = env.reset()
# total_reward = 0
# ss = input()
# while ss != "q":
#     print(state)
#     act: bool = input("act?") == "1"

#     res = env.step(act)

#     op(res)

#     state = res[0]
#     total_reward += res[1]
