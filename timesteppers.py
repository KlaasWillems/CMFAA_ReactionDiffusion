from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from ReactionDiffusionPDE import MATRIX_TYPE, MATRIX_TYPE_STR, ReactionDiffusionPDE
from typing import Optional, Tuple
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve, factorized


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
        self.res: npt.NDArray = np.empty((N, u0.size))

        # step up linear system for implicit step 
        s: Tuple = np.shape(self.RDEquation.Ku) # type: ignore  
        I: MATRIX_TYPE = eye(m=s[0], n=s[1], format=MATRIX_TYPE_STR) # type: ignore
        Au: MATRIX_TYPE = I - dt*self.RDEquation.Ku
        Av: MATRIX_TYPE = I - dt*self.RDEquation.Kv
        solveU = factorized(Au)
        solveV = factorized(Av)

        self.res[0, :] = u0
        uTemp: npt.NDArray = np.empty_like(u0)
        for i in range(1, N):
            # Explicit step of Fex
            uTemp = self.RDEquation.Fex(self.res[i-1, :], self.time[i-1])
            uTemp = self.res[i-1, :] + dt*uTemp

            # Implicit step of Fim
            self.res[i, :s[0]] = solveU(uTemp[:s[0]])
            self.res[i, s[0]:] = solveV(uTemp[s[0]:])
        return self.res


class IMEXSP(RD_timestepper):
    def __init__(self, RDEquation: ReactionDiffusionPDE) -> None:
        super().__init__(RDEquation)
    
    def integrate(self, tMin: float, tMax: float, N: int, u0: npt.NDArray) -> npt.NDArray:
        # Integrate from t_init to t_max in N-1 time steps with u0 as initial condition
        self.time: npt.NDArray = np.linspace(tMin, tMax, N)
        dt: float = (tMax - tMin)/(N-1)
        self.res: npt.NDArray = np.empty((N, u0.size))

        # step up linear system for implicit step 
        s: Tuple = np.shape(self.RDEquation.Ku) # type: ignore  
        I: MATRIX_TYPE = eye(m=s[0], n=s[1], format=MATRIX_TYPE_STR) # type:ignore
        Au: MATRIX_TYPE = I - dt*self.RDEquation.Ku
        Av: MATRIX_TYPE = I - dt*self.RDEquation.Kv
        solveU = factorized(Au)
        solveV = factorized(Av)

        self.res[0, :] = u0
        uTemp: npt.NDArray = np.empty_like(u0)
        for i in range(1, N):
            # Implicit step
            uTemp[:s[0]] = solveU(self.res[i-1, :s[0]])
            uTemp[s[0]:] = solveV(self.res[i-1, s[0]:])

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
        self.res: npt.NDArray = np.empty((N, u0.size))

        # step up linear system for implicit step 
        s: Tuple = np.shape(self.RDEquation.Ku) # type: ignore  
        I: MATRIX_TYPE = eye(m=s[0], n=s[1], format=MATRIX_TYPE_STR) # type:ignore
        Au: MATRIX_TYPE = I - dt*self.RDEquation.Ku/2
        Av: MATRIX_TYPE = I - dt*self.RDEquation.Kv/2
        solveU = factorized(Au)
        solveV = factorized(Av)

        self.res[0, :] = u0
        uTemp1: npt.NDArray = np.empty_like(u0)
        uTemp2: npt.NDArray = np.empty_like(u0)
        for i in range(1, N):
            # Implicit step
            uTemp1 = self.res[i-1, :] + dt*self.RDEquation.Fex(self.res[i-1, :], self.time[i-1]) + self.RDEquation.Fim(self.res[i-1, :], self.time[i-1])*dt/2
            uTemp2[:s[0]] = solveU(uTemp1[:s[0]])
            uTemp2[s[0]:] = solveV(uTemp1[s[0]:])

            # Explicit step
            self.res[i, :] = self.res[i-1, :] + (self.RDEquation.F(self.res[i-1, :], self.time[i-1]) + self.RDEquation.F(uTemp2, self.time[i]))*dt/2
        return self.res

        

        