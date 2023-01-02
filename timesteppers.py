# File contains all three IMEX schemes and the ADI method. 

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from ReactionDiffusionPDE import MATRIX_TYPE, MATRIX_TYPE_STR, ReactionDiffusionPDE
from typing import Optional, Tuple
from scipy.sparse import eye, diags
from scipy.sparse.linalg import spsolve, factorized
import matplotlib.pyplot as plt


class RDTimestepper(ABC):
    # Timestepper for reaction diffusion equations
    #   All methods inherit from this base class. It contains the datastructures common to all routines and provides methods for plotting results.
    def __init__(self, RDEquation: ReactionDiffusionPDE) -> None:
        self.res: Optional[npt.NDArray] = None # After integration, rows contain solution at time steps
        self.time: Optional[npt.NDArray] = None # After integration, time at which solution is computed
        self.RDEquation: ReactionDiffusionPDE = RDEquation  # Reaction-Diffusion equation object

    @abstractmethod
    def integrate(self, tMin: float, tMax: float, N: int, u0: npt.NDArray) -> npt.NDArray:
        # Integration scheme is implemented here. Must be overriden by subclasses.
        pass

    def plot(self, discretization: npt.NDArray, timeIndex: int, L: float, saveFile: Optional[str] = None) -> None:
        # Plot solution 
        #   discretization: 1D array contain the grid points in all directions
        #   timeIndex: index in self.res? 0 and -1 for first and final results.
        #   L: square domain size. For setting the axis of the plots
        #   saveFile: if not None, plot is saved at location
        assert self.res is not None
        assert self.time is not None
        if len(discretization) == 1:
            Nx: int = discretization[0]
            pos: npt.NDArray = np.linspace(0, L, Nx)
            upart: npt.NDArray = self.res[timeIndex, :Nx]
            vpart: npt.NDArray = self.res[timeIndex, Nx:]

            plt.figure()
            plt.clf()
            plt.plot(pos, upart, label='u')
            plt.plot(pos, vpart, label='v')
            plt.title(f'time = {self.time[timeIndex]}')
            plt.legend()
            plt.xlabel(('x'))
            plt.ylim([-0.1, 1.1])
            if saveFile is not None: plt.savefig(saveFile, dpi=1200)
            plt.show()
        elif len(discretization) == 2:
            Nx: int = discretization[0]
            upart: npt.NDArray = self.res[timeIndex, :Nx**2]
            vpart: npt.NDArray = self.res[timeIndex, Nx**2:]
            umatrix: npt.NDArray = upart.reshape((Nx, Nx))
            vmatrix: npt.NDArray = vpart.reshape((Nx, Nx))

            fig = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(umatrix, extent=[0, L, 0, L])
            plt.colorbar()
            plt.title('u')
            plt.subplot(1, 2, 2)
            plt.imshow(vmatrix, extent=[0, L, 0, L])
            plt.colorbar()
            plt.title('v')
            fig.suptitle(f'time = {self.time[timeIndex]}')
            if saveFile is not None: plt.savefig(saveFile, dpi=1200)
            plt.show()
        else:
            raise NotImplementedError 

    def plotAnimation(self, discretization: npt.NDArray, L: float, stride: int = 1) -> None:
        # Plot u(x, t)
        #   stride: e.g. 10 -> every 10 results are plotted. 
        assert self.res is not None
        assert self.time is not None

        Nt: int = np.shape(self.res)[0]
        Nx: int = discretization[0]

        if len(discretization) == 1:
            pos: npt.NDArray = np.linspace(0, L, Nx)

            plt.figure()
            for i in range(0, Nt, stride):
                upart: npt.NDArray = self.res[i, :Nx]
                vpart: npt.NDArray = self.res[i, Nx:]
                plt.clf()
                plt.plot(pos, upart, label='u')
                plt.plot(pos, vpart, label='v')
                plt.title(f'time = {self.time[i]}')
                plt.legend()
                plt.xlabel('x')
                plt.pause(0.05)
                if i != range(0, Nt, stride)[-1]:
                    plt.show(block=False)
                else:
                    plt.show(block=True)
        elif len(discretization) == 2:
            plt.figure()
            for i in range(0, Nt, stride):
                upart: npt.NDArray = self.res[i, :Nx**2]
                vpart: npt.NDArray = self.res[i, Nx**2:]
                umatrix: npt.NDArray = upart.reshape((Nx, Nx))
                vmatrix: npt.NDArray = vpart.reshape((Nx, Nx))

                plt.suptitle(f'time = {self.time[i]}')
                plt.subplot(1, 2, 1)
                plt.imshow(umatrix, extent=[0, L, 0, L])
                plt.colorbar()
                plt.subplot(1, 2, 2)
                plt.imshow(vmatrix, extent=[0, L, 0, L])
                plt.colorbar()
                plt.pause(0.05)
                if i != range(0, Nt, stride)[-1]:
                    plt.show(block=False)
                else:
                    plt.show(block=True)
        else:
            raise NotImplementedError 



class IMEXEuler(RDTimestepper):
    # IMEX Euler implementation
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
        solveU = factorized(Au)  # calculate LU factorization
        solveV = factorized(Av)

        self.res[0, :] = u0
        uTemp: npt.NDArray = np.empty_like(u0)
        for i in range(1, N):  # timestep
            # Explicit step of Fex
            uTemp = self.RDEquation.Fex(self.res[i-1, :], self.time[i-1])
            uTemp = self.res[i-1, :] + dt*uTemp

            # Implicit step of Fim
            self.res[i, :s[0]] = solveU(uTemp[:s[0]])
            self.res[i, s[0]:] = solveV(uTemp[s[0]:])
        return self.res


class IMEXSP(RDTimestepper):
    # IMEX SP (splitting) method implementation
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
        solveU = factorized(Au)  # calculate LU factorization
        solveV = factorized(Av)

        self.res[0, :] = u0
        uTemp: npt.NDArray = np.empty_like(u0)
        for i in range(1, N):  # timestep
            # Implicit step
            uTemp[:s[0]] = solveU(self.res[i-1, :s[0]])
            uTemp[s[0]:] = solveV(self.res[i-1, s[0]:])

            # Explicit step
            self.res[i, :] = self.res[i-1, :] + dt*(self.RDEquation.Fex(self.res[i-1, :], self.time[i-1]) + self.RDEquation.Fim(uTemp, self.time[i])) 
        return self.res


class IMEXTrap(RDTimestepper):
    # IMEX trapezoidal implementation
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
        solveU = factorized(Au)  # calculate LU factorization
        solveV = factorized(Av)

        self.res[0, :] = u0
        uTemp1: npt.NDArray = np.empty_like(u0)
        uTemp2: npt.NDArray = np.empty_like(u0)
        for i in range(1, N):  # timestep
            # Implicit step
            uTemp1 = self.res[i-1, :] + dt*self.RDEquation.Fex(self.res[i-1, :], self.time[i-1]) + self.RDEquation.Fim(self.res[i-1, :], self.time[i-1])*dt/2
            uTemp2[:s[0]] = solveU(uTemp1[:s[0]])
            uTemp2[s[0]:] = solveV(uTemp1[s[0]:])

            # Explicit step
            self.res[i, :] = self.res[i-1, :] + (self.RDEquation.F(self.res[i-1, :], self.time[i-1]) + self.RDEquation.F(uTemp2, self.time[i]))*dt/2
        return self.res


class ADI(RDTimestepper):
    # ADI (alternating direction implicit) method implementation
    def __init__(self, RDEquation: ReactionDiffusionPDE) -> None:
        super().__init__(RDEquation)

    def integrate(self, tMin: float, tMax: float, N: int, u0: npt.NDArray) -> npt.NDArray:
        # uO: 2Nx by Nx matrix containing the U and V matrices stacked on top of each other

        # variables having to do with time
        self.time = np.linspace(tMin, tMax, N)
        dt: float = (tMax - tMin)/(N-1)
        dt2: float = dt/2
        
        # variables having to do with space
        Nx: int = np.shape(u0)[1]  # Nx**2
        Nx2: int = int(Nx**2)
        dx: float = self.RDEquation.L/(Nx-1)

        # Build matrices for nearly tridiagonal system. See document from exercise session for matrix structure
        coeff: float = dt2/(dx**2)
        alfau: float = coeff*self.RDEquation.Du  # (alfa = beta = Di*dt/(2*dx**2))
        alfav: float = coeff*self.RDEquation.Dv
        Au: MATRIX_TYPE = diags([-alfau, -alfau, 1+2*alfau, -alfau, -alfau], [-Nx+1, -1, 0, 1, Nx-1], format=MATRIX_TYPE_STR, shape=(Nx, Nx)) # type: ignore
        Av: MATRIX_TYPE = diags([-alfav, -alfav, 1+2*alfav, -alfav, -alfav], [-Nx+1, -1, 0, 1, Nx-1], format=MATRIX_TYPE_STR, shape=(Nx, Nx)) # type: ignore
        solveU = factorized(Au)
        solveV = factorized(Av)

        # Allocate arrays for solution
        self.res = np.empty((2*Nx, Nx, N))  # Final result stored tensor. U and V stacked vertically
        self.res[:, :, 0] = u0 # u matrix at t = 0

        # Pre-allocate temporary variables
        fuv: npt.NDArray = np.empty_like(u0)
        uHalf: npt.NDArray = np.empty((Nx, Nx))
        vHalf: npt.NDArray = np.empty((Nx, Nx))
        for i in range(1, N): # timestep
            # step 1
            fuv = self.RDEquation.Fex(self.res[:, :, i-1], self.time[i-1])
            uHalf = self.ADIStep1(solveU, alfau, dt2, self.res[:Nx, :, i-1], fuv[:Nx, :])
            vHalf = self.ADIStep1(solveV, alfav, dt2, self.res[Nx:, :, i-1], fuv[Nx:, :])

            # compute reaction terms at middle of interval
            fuv = self.RDEquation.Fex(np.vstack((uHalf, vHalf)), (self.time[i-1]+self.time[i])/2)

            # step 2
            self.res[:Nx, :, i] = self.ADIStep2(solveU, alfau, dt2, uHalf, fuv[:Nx, :])
            self.res[Nx:, :, i] = self.ADIStep2(solveV, alfav, dt2, vHalf, fuv[Nx:, :])
        return self.res

    # TODO: parallelize loop
    def ADIStep1(self, LuSolve, alfa: float, dt2: float, uOld: npt.NDArray, FuOld: npt.NDArray) -> npt.NDArray:
        # Implements 'x-sweep' of ADI method
        #   LuSolve: matrix with 1+alfa, -alfa/2 and -alfa/2 on its diagonals
        #   alfa: Du*dt/(2*dx**2)
        #   dt2: timestep/2 
        #   uOld: (Nx, Nx) matrix 
        #   FuOld: reaction terms evaluated at uOld. Also, (Nx, Nx) matrix

        uHalf: npt.NDArray = np.empty_like(uOld)
        rows, cols = np.shape(uOld)

        uRHS: npt.NDArray = np.empty_like((rows, ), dtype=float)
        for row in range(rows): # x-sweep 
            rowUp = (row - 1) % cols
            rowDown = (row + 1) % cols
            uRHS = alfa*uOld[rowUp, :] + (1-2*alfa)*uOld[row, :] + alfa*uOld[rowDown, :] + dt2*FuOld[row, :]
            uHalf[row, :] = LuSolve(uRHS)

        return uHalf

    # TODO: parallelize loop
    def ADIStep2(self, LuSolve, alfa: float, dt2: float, uHalf: npt.NDArray, FuHalf: npt.NDArray) -> npt.NDArray:
        # Implements 'x-sweep' of ADI method. Argument analogous to ADIStep1
        uFull: npt.NDArray = np.empty_like(uHalf)
        rows, cols = np.shape(uHalf)

        uRHS: npt.NDArray = np.empty_like((rows, ), dtype=float)
        for col in range(cols): # x-sweep 
            colLeft = (col - 1) % rows
            colRight = (col + 1) % rows
            uRHS = alfa*uHalf[:, colLeft] + (1-2*alfa)*uHalf[:, col] + alfa*uHalf[:, colRight] + dt2*FuHalf[:, col]
            uFull[:, col] = LuSolve(uRHS)
        
        return uFull

    def plot(self, discretization: npt.NDArray, timeIndex: int, L: float, saveFile: Optional[str] = None) -> None:
        # Overrides plot method from baseclass because the ADI method stores the results in matrix format, not vectorized.
        assert self.res is not None
        assert self.time is not None
        assert len(discretization) == 2
        Nx: int = discretization[0]
        umatrix: npt.NDArray = self.res[:Nx, :, timeIndex]
        vmatrix: npt.NDArray = self.res[Nx:, :, timeIndex]

        plt.figure()
        plt.imshow(umatrix, extent=[0, L, 0, L])
        plt.colorbar()
        plt.title(f'time = {self.time[timeIndex]}')
        if saveFile is not None: plt.savefig(saveFile, dpi=1200)
        plt.show()

