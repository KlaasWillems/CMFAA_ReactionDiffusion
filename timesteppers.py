from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from ReactionDiffusionPDE import ReactionDiffusionPDE
from typing import Optional
from scipy.sparse import eye, csr_matrix, dia_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


class RD_timestepper(ABC):
    # Timestepper for reaction diffusion equations
    def __init__(self, RDEquation: ReactionDiffusionPDE) -> None:
        self.res: Optional[npt.NDArray] = None # After integration, rows contain solution at time steps
        self.time: Optional[npt.NDArray] = None # After integration, time at which solution is computed
        self.RDEquation: ReactionDiffusionPDE = RDEquation

    @abstractmethod
    def integrate(self, tMin: float, tMax: float, N: int, u0: npt.NDArray) -> None:
        pass


class IMEXEuler(RD_timestepper):
    def __init__(self, RDEquation: ReactionDiffusionPDE) -> None:
        super().__init__(RDEquation)

    def integrate(self, tMin: float, tMax: float, N: int, u0: npt.NDArray) -> None:
        # Integrate from t_init to t_max in N-1 time steps with u0 as initial condition
        self.time: npt.NDArray = np.linspace(tMin, tMax, N)
        dt: float = (tMax - tMin)/(N-1)
        ODESize: int = np.shape(u0)[0]
        self.res: npt.NDArray = np.empty((N, ODESize))

        # step up linear system for implicit step 
        I: csr_matrix = eye(ODESize, format='csr') # type:ignore
        A: csr_matrix = I - dt*self.RDEquation.K

        self.res[0, :] = u0
        uOld: npt.NDArray = np.copy(u0) 
        uTemp: npt.NDArray = np.empty_like(u0)
        for i in range(1, N):
            # Explicit step of Fex
            uTemp = self.RDEquation.Fex(uOld, self.time[i])
            uTemp = uOld + dt*uTemp

            # Implicit step of Fim
            uOld = spsolve(A, uTemp)

            # write result to matrix
            self.res[i, :] = uOld





        