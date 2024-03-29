# Demo of the Gray-Scott model in 2D using the IMEX Euler scheme. 
#   Simulated until t = 2000 on a 128**2 grid. After the computation, the end concentrations of u and v are plotted.

from timesteppers import IMEXEuler, IMEXSP, IMEXTrap
from RDModels import GrayScott
import numpy as np
import numpy.typing as npt

# discretization parameters
L: int = 2
Nx: int = 128
discretization: npt.NDArray = np.array([Nx, Nx], dtype=int)
tmin: float = 0.0 
tmax: float = 2000
Nt: int = tmax*2

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
p1: npt.NDArray = np.exp( -25*((xv - x1)**2 + (yv - y1)**2))/2
p2: npt.NDArray = np.exp( -25*((xv - x2)**2 + (yv - y2)**2))/2
umatrix: npt.NDArray = np.ones_like(p1) - p1 - p2  # type: ignore
vmatrix: npt.NDArray = np.zeros_like(p2) + p1 + p2
upart: npt.NDArray = umatrix.reshape((Nx**2, ))
vpart: npt.NDArray = vmatrix.reshape((Nx**2, ))
u0: npt.NDArray = np.hstack((upart, vpart))

# Make PDE object
GS: GrayScott = GrayScott(discretization, L, Du, Dv, F, k)

# Make time stepper. Can be either IMEXEuler, IMEXSP or IMEXTrap
imex1: IMEXEuler = IMEXEuler(GS)

# Integrate
imex1.integrate(tmin, tmax, Nt, u0)

# Plot
# imex1.plot(discretization, -1, L)
imex1.plotAnimation(discretization, L, stride = 20)  # Plot concentrations (every 'stride' frames) 