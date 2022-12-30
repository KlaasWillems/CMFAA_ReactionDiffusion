from ReactionDiffusionPDE import ReactionDiffusionPDE
import numpy.typing as npt
import numpy as np


class GrayScott(ReactionDiffusionPDE):
    def __init__(self, discretization: npt.NDArray, L: int, Du: float, Dv: float, F: float, k: float) -> None:
        super().__init__(discretization, L, Du, Dv)
        self.k: float = k # Gray Scott model parameters
        self.F: float = F

    def Fex(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate explicit reaction term at u at time t. 
        res: npt.NDArray = np.empty_like(u)
        m: int = np.shape(u)[0]
        m2: int = int(m/2)
        upart: npt.NDArray = u[:m2]
        vpart: npt.NDArray = u[m2:]
        res[:m2] = -upart*(vpart**2) + self.F*(1-upart)
        res[m2:] = upart*(vpart**2) - (self.F + self.k)*vpart
        return res
        

class Schnakeberg(ReactionDiffusionPDE):
    def __init__(self, discretization: npt.NDArray, L: int, Du: float, Dv: float, k: float, a: float, b: float) -> None:
        super().__init__(discretization, L, Du, Dv)
        self.k: float = k
        self.a: float = a
        self.b: float = b

    def Fex(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate explicit reaction term at u at time t.
        res: npt.NDArray = np.empty_like(u)
        m: int = np.shape(u)[0]
        m2: int = int(m/2)
        upart: npt.NDArray = u[:m2]
        vpart: npt.NDArray = u[m2:]
        res[:m2] = self.k*(self.a - upart + vpart*upart**2)
        res[m2:] = self.k*(self.b - vpart*upart**2)
        return res

class HeatEquation(ReactionDiffusionPDE):
    def __init__(self, discretization: npt.NDArray, L: int, Du: float, Dv: float) -> None:
        super().__init__(discretization, L, Du, Dv)

    def Fex(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate explicit reaction term at u at time t. 
        return np.zeros_like(u)
