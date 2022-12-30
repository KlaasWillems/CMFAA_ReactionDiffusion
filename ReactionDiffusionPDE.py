from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple
from scipy.sparse import diags, csr_matrix, hstack, vstack, csc_matrix, eye, kron
import matplotlib.pyplot as plt

MATRIX_TYPE = csc_matrix
MATRIX_TYPE_STR = 'csc'

class ReactionDiffusionPDE(ABC):
    def __init__(self, discretization: npt.NDArray, L: int, Du: float, Dv: float) -> None:
        self.discretization: npt.NDArray = discretization # 1D array containing the amount of discrization steps in each dimension
        self.L: int = L # square domain size
        self.Du: float = Du # u diffusion coefficient
        self.Dv: float = Dv # v diffusion coefficient
        self.res: Optional[npt.NDArray] = None
        self.discretize(discretization, L)

    def discretize(self, discretization: npt.NDArray, L: float) -> None:
        # generate spatially discretized PDE. Assume periodic boundary conditions
        Nx: int = discretization[0] # Nx discretisation points in x direction
        dx: float = L/(Nx-1)

        # Assemble matrix
        if len(discretization) == 1: # 1D 
            z: MATRIX_TYPE = MATRIX_TYPE((Nx, Nx))
            self.K: MATRIX_TYPE = diags([1, 1, -2, 1, 1], [-Nx+1, -1, 0, 1, Nx-1], format=MATRIX_TYPE_STR, shape=(Nx, Nx))/(dx**2) # type: ignore

            # left = vstack([laplacian1D*self.Du, z])
            # right = vstack([z, laplacian1D*self.Dv])
            # self.K = hstack([left, right])
        elif len(discretization) == 2: # 2D 
            assert discretization[0] == discretization[1], 'Only square grids allowed'
            raise NotImplementedError
            # laplacian2D: MATRIX_TYPE = diags([1, 1, -2, 1, 1], [-Nx+1, -1, 0, 1, Nx-1], format=MATRIX_TYPE_STR, shape=(Nx, Nx))/(dx**2) # type: ignore
            # I: MATRIX_TYPE = eye(Nx, format=MATRIX_TYPE_STR)  # type: ignore
            # temp2: MATRIX_TYPE = kron(I, laplacian2D, format=MATRIX_TYPE_STR) + kron(laplacian2D, I, format=MATRIX_TYPE_STR)
            # z: MATRIX_TYPE = MATRIX_TYPE((Nx**2, Nx**2))
            # self.K: MATRIX_TYPE = hstack([vstack([temp2*self.Du, z]), vstack([z, temp2*self.Dv])])
        else:
            raise NotImplementedError

    def plot(self, time: npt.NDArray, timeIndex: int, res: npt.NDArray) -> None:
        # Plot solution at timeIndex
        if len(self.discretization):
            Nt: int = np.shape(res)[0]
            Nx: int = self.discretization[0]
            pos: npt.NDArray = np.linspace(0, self.L, Nx)

            plt.figure()
            upart: npt.NDArray = res[timeIndex, :Nx]
            vpart: npt.NDArray = res[timeIndex, Nx:]
            plt.clf()
            plt.plot(pos, upart, label='u')
            plt.plot(pos, vpart, label='v')
            plt.title(f'time = {time[timeIndex]}')
            plt.legend()
            plt.xlabel(('x'))
            plt.show()
        else:
            raise NotImplementedError 

    def plotAnimation(self, time: npt.NDArray, res: npt.NDArray) -> None:
        # Plot u(x, t)
        if len(self.discretization):
            Nt: int = np.shape(res)[0]
            Nx: int = self.discretization[0]
            pos: npt.NDArray = np.linspace(0, self.L, Nx)

            plt.figure()
            for i in range(Nt):
                upart: npt.NDArray = res[i, :Nx]
                vpart: npt.NDArray = res[i, Nx:]
                plt.clf()
                plt.plot(pos, upart, label='u')
                plt.plot(pos, vpart, label='v')
                plt.title(f'time = {time[i]}')
                plt.legend()
                plt.xlabel(('x'))
                plt.pause(0.5)
                if i != Nt-1:
                    plt.show(block=False)
                else:
                    plt.show()
        else:
            raise NotImplementedError 
    
    @abstractmethod
    def Fex(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate explicit reaction term at u at time t. 
        pass

    def Fim(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate implicit reaction term at u at time t. 
        s: int = int(u.size/2)
        res: npt.NDArray = np.empty_like(u)
        res[:s] = self.K.dot(u[:s])
        res[s:] = self.K.dot(u[s:])
        return res 

    def F(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate implicit and explicit terms together
        return self.Fim(u, t) + self.Fex(u, t)
    

