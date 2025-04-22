from pathlib import Path

from objprint import op

from fp_btree import FplanEnv
import fp_btree
ROOT_PATH = Path(__file__).parent


raw_fn = ROOT_PATH / "raw_data" / "ami33"

env = FplanEnv(str(raw_fn), 0.5, 10000)

def try_act(env: FplanEnv, act: fp_btree.Action):
    res = env.step(act)
    env.recover()

    return res

op(
    env.get_cost(),
    env.get_init_cost(),
    env.get_baseline()
)