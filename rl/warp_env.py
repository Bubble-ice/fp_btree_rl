from fp_btree import FplanEnv
import numpy as np
import numpy.typing as npt


class FplanEnvWrap(FplanEnv):
    def __init__(self, fn: str = "", calpha: float = 1, max_times=5000) -> None:
        super().__init__(fn, calpha, max_times)
        self.state_dim = 6
        self.first_state = None
        self.pre_state = None
        self.is_recover = False

    def reset(self) -> npt.NDArray[np.float64]:
        super().reset()
        s_0 = super().step_rand()
        self.first_state = s_0
        self.pre_state = s_0
        return s_0[0]

    def step(self, act: bool) -> tuple:
        if act:
            self.pre_state = self.step_rand()
            self.is_recover = False
        elif not self.is_recover:
            self.recover()
            self.is_recover = True
        else:
            pass

        s_n = self.pre_state

        if s_n is None:
            raise RuntimeError("还没reset呢")

        s, r, a = s_n

        return s, r, a, {}


if __name__ == "__main__":
    ...
