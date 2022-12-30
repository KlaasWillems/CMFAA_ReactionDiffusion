from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from ReactionDiffusionPDE import ReactionDiffusionPDE
from typing import Optional
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import spsolve


class RD_timestepper(ABC):
    # Timestepper for reaction diffusion equations
    def __init__(self) -> None:
        self.res: Optional[None] = None # After integration, rows contain solution at time steps
        self.time: Optional[None] = None # After integration, time at which solution is computed

    @abstractmethod
    def integrate(t_init: float, t_max: float, N: int) -> None:
        pass


class IMEXEuler(RD_timestepper):
    def __init__(self) -> None:
        super().__init__()

    def integrate(self, tMin: float, tMax: float, N: int, u0: npt.NDArray) -> None:
        # Integrate from t_init to t_max in N time steps with u0 as initial condition
        self.time = np.linspace(tMin, tMax, N+1)
        dt: float = (tMax - tMin)/N
        ODESize: int = np.shape(u0)[0]
        self.res = np.empty((N+1, ODESize))

        # step up linear system for implicit step 
        I: eye = eye(ODESize)
        A: csr_matrix = I - dt*self.RDequation.K

        uOld: npt.NDArray = np.copy(u0) 
        uTemp: npt.NDArray = np.empty_like(u0)
        for i in range(1, N+1):
            # Explicit step of Fex
            self.RDequation.Fex(uOld, self.time[i], uTemp)
            uTemp = uOld + dt*uTemp

            # Implicit step of Fim
            uOld = spsolve(A, uTemp)

            # write result to matrix
            self.res[i, :] = uOld





        