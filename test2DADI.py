from timesteppers import ADI
from RDModels import GrayScott
import numpy as np
import numpy.typing as npt

# discretization parameters
L: int = 2
Nx: int = 128
discretization: npt.NDArray = np.array([Nx, Nx], dtype=int)
tmin: float = 0.0 
tmax: float = 1000
Nt: int = tmax*4

# Model parameters
F: float = 0.046
k: float = 0.063
Du: float = 2e-5
Dv: float = 1e-5

# Initial condition
x: npt.NDArray = np.linspace(0, L, Nx, endpoint=False)
y: npt.NDArray = np.linspace(0, L, Nx, endpoint=False)
xv, yv = np.meshgrid(x, y)
x1 = 0.5; x2 = 0.55
y1 = 0.5; y2 = 0.6
p1 = np.exp( -25*((xv - x1)**2 + (yv - y1)**2))/2
p2 = np.exp( -25*((xv - x2)**2 + (yv - y2)**2))/2
umatrix = np.ones_like(p1) - p1 - p2
vmatrix = np.ones_like(p2) + p1 + p2
u0: npt.NDArray = np.vstack((umatrix, vmatrix))

# Make PDE object
GS: GrayScott = GrayScott(discretization, L, Du, Dv, F, k)

# Make time stepper. Can be either IMEXEuler, IMEXSP or IMEXTrap
imex1: ADI = ADI(GS)

# Integrate
imex1.integrate(tmin, tmax, Nt, u0)

# Plot
imex1.plot(discretization, -1, L)
# imex1.plotAnimation(discretization, L)