from ReactionDiffusionPDE import ReactionDiffusionPDE
import numpy.typing as npt
import numpy as np


class GrayScott(ReactionDiffusionPDE):
    # Implementation of the Gray-Scott model
    #   This object is a Reaction-Diffusion PDE and thus inherits from ReactionDiffusionPDE. 
    #   Fvar: Feed coefficient
    #   k: Kill coefficient
    def __init__(self, discretization: npt.NDArray, L: float, Du: float, Dv: float, Fvar: float, k: float, discretize: bool = True) -> None: 
        super().__init__(discretization, L, Du, Dv, discretize)
        self.k: float = k 
        self.Fvar: float = Fvar

    def Fex(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate explicit reaction term at u at time t.
        res: npt.NDArray = np.empty_like(u)
        m: int = np.shape(u)[0]  
        m2: int = int(m/2)
        upart: npt.NDArray = u[:m2]
        vpart: npt.NDArray = u[m2:]
        res[:m2] = -upart*np.power(vpart, 2) + self.Fvar*(1-upart)
        res[m2:] = upart*np.power(vpart, 2) - (self.Fvar + self.k)*vpart
        return res
        

class Schnakenberg(ReactionDiffusionPDE):
    # Implementation of the Gray-Scott model
        #   This object is a Reaction-Diffusion PDE and thus inherits from ReactionDiffusionPDE. 
        #   a and b are parameters specific to the Schnakenberg model
    def __init__(self, discretization: npt.NDArray, L: float, Du: float, Dv: float, k: float, a: float, b: float, discretize: bool = True) -> None:
        super().__init__(discretization, L, Du, Dv, discretize)
        self.k: float = k  
        self.a: float = a
        self.b: float = b

    def Fex(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate explicit reaction term at u at time t.
        assert u.ndim == 1
        res: npt.NDArray = np.empty_like(u)
        m: int = np.shape(u)[0]
        m2: int = int(m/2)
        upart: npt.NDArray = u[:m2]
        vpart: npt.NDArray = u[m2:]
        res[:m2] = self.k*(self.a - upart + vpart**np.power(upart, 2))
        res[m2:] = self.k*(self.b - vpart*np.power(upart, 2))
        return res


class HeatEquation(ReactionDiffusionPDE):
    # Implementation of the Gray-Scott model
        #   When the nonlinear term in a reaction-diffusion equation is put to zero, we get the time-dependent heat equation. u and v then behave independently. Used for testing. 
    def __init__(self, discretization: npt.NDArray, L: int, Du: float, Dv: float, discretize: bool = True) -> None:
        super().__init__(discretization, L, Du, Dv, discretize)

    def Fex(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate explicit reaction term at u at time t. 
        return np.zeros_like(u)
