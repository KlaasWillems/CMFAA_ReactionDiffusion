from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple
from scipy.sparse import diags, csr_matrix, hstack, vstack, csc_matrix, eye, kron
import matplotlib.pyplot as plt

MATRIX_TYPE = csc_matrix
MATRIX_TYPE_STR = 'csc'

class ReactionDiffusionPDE(ABC):
    def __init__(self, discretization: npt.NDArray, L: int, Du: float, Dv: float, discretize=True) -> None:
        self.discretization: npt.NDArray = discretization # 1D array containing the amount of discrization steps in each dimension
        self.L: int = L # square domain size
        self.Du: float = Du # u diffusion coefficient
        self.Dv: float = Dv # v diffusion coefficient
        if discretize:
            self.discretize(discretization, L)

    def discretize(self, discretization: npt.NDArray, L: float) -> None:
        # Generate 1D or 2D laplacian matrix with periodic boundary conditions
        Nx: int = discretization[0] # Nx discretisation points in x direction
        dx: float = L/(Nx-1)

        # Assemble matrix
        if len(discretization) == 1: # 1D 
            laplacian: MATRIX_TYPE = diags([1, 1, -2, 1, 1], [-Nx+1, -1, 0, 1, Nx-1], format=MATRIX_TYPE_STR, shape=(Nx, Nx))/(dx**2) # type: ignore
            self.Ku: MATRIX_TYPE = laplacian*self.Du  # type: ignore
            self.Kv: MATRIX_TYPE = laplacian*self.Dv  # type: ignore
        elif len(discretization) == 2: # 2D 
            assert discretization[0] == discretization[1], 'Only square grids allowed'
            laplacian2D: MATRIX_TYPE = diags([1, 1, -2, 1, 1], [-Nx+1, -1, 0, 1, Nx-1], format=MATRIX_TYPE_STR, shape=(Nx, Nx))/(dx**2) # type: ignore
            I: MATRIX_TYPE = eye(Nx, format=MATRIX_TYPE_STR)  # type: ignore
            temp2: MATRIX_TYPE = kron(I, laplacian2D, format=MATRIX_TYPE_STR) + kron(laplacian2D, I, format=MATRIX_TYPE_STR)
            self.Ku: MATRIX_TYPE = temp2*self.Du
            self.Kv: MATRIX_TYPE = temp2*self.Dv
        else:
            raise NotImplementedError
    
    @abstractmethod
    def Fex(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate explicit reaction term at u at time t. 
        pass

    def Fim(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate implicit reaction term at u at time t. 
        assert u.ndim == 1
        s: int = int(u.size/2)
        res: npt.NDArray = np.empty_like(u)
        res[:s] = self.Ku.dot(u[:s])
        res[s:] = self.Kv.dot(u[s:])
        return res 

    def F(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate implicit and explicit terms together
        assert u.ndim == 1
        return self.Fim(u, t) + self.Fex(u, t)
    

