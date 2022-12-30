from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from ReactionDiffusionPDE import MATRIX_TYPE, MATRIX_TYPE_STR, ReactionDiffusionPDE
from typing import Optional
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve


class RD_timestepper(ABC):
    # Timestepper for reaction diffusion equations
    def __init__(self, RDEquation: ReactionDiffusionPDE) -> None:
        self.res: Optional[npt.NDArray] = None # After integration, rows contain solution at time steps
        self.time: Optional[npt.NDArray] = None # After integration, time at which solution is computed
        self.RDEquation: ReactionDiffusionPDE = RDEquation

    @abstractmethod
    def integrate(self, tMin: float, tMax: float, N: int, u0: npt.NDArray) -> npt.NDArray:
        pass


class IMEXEuler(RD_timestepper):
    def __init__(self, RDEquation: ReactionDiffusionPDE) -> None:
        super().__init__(RDEquation)

    def integrate(self, tMin: float, tMax: float, N: int, u0: npt.NDArray) -> npt.NDArray:
        # Integrate from t_init to t_max in N-1 time steps with u0 as initial condition
        self.time: npt.NDArray = np.linspace(tMin, tMax, N)
        dt: float = (tMax - tMin)/(N-1)
        ODESize: int = np.shape(u0)[0]
        self.res: npt.NDArray = np.empty((N, ODESize))

        # step up linear system for implicit step 
        I: MATRIX_TYPE = eye(ODESize, format=MATRIX_TYPE_STR) # type:ignore
        A: MATRIX_TYPE = I - dt*self.RDEquation.K

        self.res[0, :] = u0
        uTemp: npt.NDArray = np.empty_like(u0)
        for i in range(1, N):
            # Explicit step of Fex
            uTemp = self.RDEquation.Fex(self.res[i-1, :], self.time[i-1])
            uTemp = self.res[i-1, :] + dt*uTemp

            # Implicit step of Fim
            self.res[i, :] = spsolve(A, uTemp)
        return self.res


class IMEXSP(RD_timestepper):
    def __init__(self, RDEquation: ReactionDiffusionPDE) -> None:
        super().__init__(RDEquation)
    
    def integrate(self, tMin: float, tMax: float, N: int, u0: npt.NDArray) -> npt.NDArray:
        # Integrate from t_init to t_max in N-1 time steps with u0 as initial condition
        self.time: npt.NDArray = np.linspace(tMin, tMax, N)
        dt: float = (tMax - tMin)/(N-1)
        ODESize: int = np.shape(u0)[0]
        self.res: npt.NDArray = np.empty((N, ODESize))

        # step up linear system for implicit step 
        I: MATRIX_TYPE = eye(ODESize, format=MATRIX_TYPE_STR) # type:ignore
        A: MATRIX_TYPE = I - dt*self.RDEquation.K

        self.res[0, :] = u0
        uTemp: npt.NDArray = np.empty_like(u0)
        for i in range(1, N):
            # Implicit step
            uTemp = spsolve(A, self.res[i-1, :])

            # Explicit step
            self.res[i, :] = self.res[i-1, :] + dt*(self.RDEquation.Fex(self.res[i-1, 0], self.time[i-1]) + self.RDEquation.Fim(uTemp, self.time[i])) 
        return self.res


class IMEXTrap(RD_timestepper):
    def __init__(self, RDEquation: ReactionDiffusionPDE) -> None:
        super().__init__(RDEquation)

    def integrate(self, tMin: float, tMax: float, N: int, u0: npt.NDArray) -> npt.NDArray:
        # Integrate from t_init to t_max in N-1 time steps with u0 as initial condition
        self.time: npt.NDArray = np.linspace(tMin, tMax, N)
        dt: float = (tMax - tMin)/(N-1)
        ODESize: int = np.shape(u0)[0]
        self.res: npt.NDArray = np.empty((N, ODESize))

        # step up linear system for implicit step 
        I: MATRIX_TYPE = eye(ODESize, format=MATRIX_TYPE_STR) # type:ignore
        A: MATRIX_TYPE = I - dt*self.RDEquation.K/2

        self.res[0, :] = u0
        uTemp1: npt.NDArray = np.empty_like(u0)
        uTemp2: npt.NDArray = np.empty_like(u0)
        for i in range(1, N):
            # Implicit step
            uTemp1 = self.res[i-1, :] + dt*self.RDEquation.Fex(self.res[i-1, :], self.time[i-1]) + self.RDEquation.Fim(self.res[i-1, :], self.time[i-1])*dt/2
            uTemp2 = spsolve(A, uTemp1)

            # Explicit step
            self.res[i, :] = self.res[i-1, :] + (self.RDEquation.F(self.res[i-1, :], self.time[i-1]) + self.RDEquation.F(uTemp2, self.time[i]))*dt/2
        return self.res

        

        