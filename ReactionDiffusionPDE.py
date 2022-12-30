from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple
from scipy.sparse import diags, csr_matrix, hstack, vstack
from timesteppers import RD_timestepper

class ReactionDiffusionPDE(ABC):
    def __init__(self, discretization: npt.NDArray, L: int, Du: float, Dv: float, timeStepper: RD_timestepper) -> None:
        self.discretization: npt.NDArray = discretization # 1D array containing the amount of discrization steps in each dimension
        self.L: int = L # square domain size
        self.Du: float = Du # u diffusion coefficient
        self.Dv: float = Dv # v diffusion coefficient
        self.K: Optional[npt.NDArray] # discretization of laplace operator
        self.timeStepper: RD_timestepper = timeStepper
        self.discretize(discretization, L)

    def discretize(self, discretization: npt.NDArray, L: int) -> None:
        # generate spatially discretized PDE. Assume periodic boundary conditions
        s: Tuple = np.shape(discretization)

        # Assemble matrix
        if len(s) == 1: # 1D problem
            Nx: int = discretization[0] # Nx-1 discretisation points in x direction
            dx: int = L/Nx 
            z: csr_matrix = csr_matrix((Nx, Nx))
            temp: csr_matrix = diags([1, 1, -2, 1, 1], [-Nx+1, -1, 0, 1, Nx-1], format='csr', shape=(Nx, Nx))/(dx**2)
            left: csr_matrix = vstack([temp*self.Du, z])
            right: csr_matrix = vstack([z*self.Dv, temp])
            self.K: csr_matrix = hstack([left, right])
        elif len(s) == 2: # 2D problem
            raise NotImplementedError
        else:
            raise NotImplementedError
    
    @abstractmethod
    def Fex(self, u: npt.NDArray, t: float, res: npt.NDArray) -> None:
        # Evaluate explicit reaction term at u at time t. Writes result into res.
        pass

    def Fim(self, u: npt.NDArray, t: float, res: npt.NDArray) -> None:
        # Evaluate implicit reaction term at u at time t. Writes result into res.
        res: npt.NDArray = self.K.dot(u) 
    

