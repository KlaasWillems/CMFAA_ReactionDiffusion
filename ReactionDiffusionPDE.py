from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple
from scipy.sparse import diags, csr_matrix, hstack, vstack, csc_matrix
import matplotlib.pyplot as plt

MATRIX_TYPE = csr_matrix
MATRIX_TYPE_STR = 'csr'

class ReactionDiffusionPDE(ABC):
    def __init__(self, discretization: npt.NDArray, L: int, Du: float, Dv: float) -> None:
        self.discretization: npt.NDArray = discretization # 1D array containing the amount of discrization steps in each dimension
        self.L: int = L # square domain size
        self.Du: float = Du # u diffusion coefficient
        self.Dv: float = Dv # v diffusion coefficient
        self.res: Optional[npt.NDArray] = None
        self.discretize(discretization, L)

    def discretize(self, discretization: npt.NDArray, L: int) -> None:
        # generate spatially discretized PDE. Assume periodic boundary conditions
        s: Tuple = np.shape(discretization)

        # Assemble matrix
        if len(s) == 1: # 1D 
            Nx: int = discretization[0] # Nx discretisation points in x direction
            dx: float = L/(Nx-1)
            z: MATRIX_TYPE = MATRIX_TYPE((Nx, Nx))
            temp: MATRIX_TYPE = diags([1, 1, -2, 1, 1], [-Nx+1, -1, 0, 1, Nx-1], format=MATRIX_TYPE_STR, shape=(Nx, Nx))/(dx**2) # type: ignore
            left = vstack([temp*self.Du, z])
            right = vstack([z, temp*self.Dv])
            self.K = hstack([left, right])
        elif len(s) == 2: # 2D 
            raise NotImplementedError
        else:
            raise NotImplementedError

    def plot(self, time: npt.NDArray, timeIndex: int, res: npt.NDArray) -> None:
        if len(np.shape(self.discretization)):
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
        if len(np.shape(self.discretization)):
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
        # Evaluate explicit reaction term at u at time t. Writes result into res.
        pass

    def Fim(self, u: npt.NDArray, t: float) -> npt.NDArray:
        # Evaluate implicit reaction term at u at time t. Writes result into res.
        res: npt.ArrayLike = self.K.dot(u)
        return res 
    

