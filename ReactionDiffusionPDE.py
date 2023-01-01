# This file contains the Reaction-Diffusion object. It implements the methods that all reaction-diffusion models have in common. 
#   For example, in all cases is the diffusion term discretized with central finite differences. All reaction-diffusion models have two diffusion parameters...
#   Specific reaction diffusion equations, like the Gray-Scott model, inherit from this class. They are implemented in RDModels.py

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from scipy.sparse import diags, csr_matrix, csc_matrix, eye, kron
import matplotlib.pyplot as plt

MATRIX_TYPE = csc_matrix  # Matrix type used for all sparse matrices. csr or csc is preferred by the sparse LU solver from scipy.
MATRIX_TYPE_STR = 'csc'

class ReactionDiffusionPDE(ABC):
    def __init__(self, discretization: npt.NDArray, L: float, Du: float, Dv: float, discretize=True) -> None:
        self.discretization: npt.NDArray = discretization # 1D array containing the amount of discrization points in each dimension
        self.L: float = L # square domain size
        self.Du: float = Du # u diffusion coefficient
        self.Dv: float = Dv # v diffusion coefficient

        # This will generate the discretizations of the Laplace matrix. In the case of the ADI method, we don't need the 2D Laplace matrix. 
        #   Using the boolean, we can avoid creating a large matrix that isn't used by the scheme.
        if discretize:  
            self.discretize(discretization, L)

    def discretize(self, discretization: npt.NDArray, L: float) -> None:
        # Generate 1D or 2D laplacian matrix with periodic boundary conditions.
        #   The Laplace matrices for u and v only differ by the diffusion term.
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
        # Evaluate explicit reaction term at u at time t. This depends on the Reaction-Diffusion model and must be implemented there.
        pass

    def Fim(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate implicit reaction term at u at time t. 
        assert u.ndim == 1
        assert hasattr(self, 'Ku')
        assert hasattr(self, 'Kv')
        s: int = int(u.size/2)
        res: npt.NDArray = np.empty_like(u)
        res[:s] = self.Ku.dot(u[:s])
        res[s:] = self.Kv.dot(u[s:])
        return res 

    def F(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate implicit and explicit terms together.
        assert u.ndim == 1
        return self.Fim(u, t) + self.Fex(u, t)
    

