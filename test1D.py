from timesteppers import IMEXEuler, IMEXSP, IMEXTrap
from RDModels import GrayScott
import numpy as np
import numpy.typing as npt

# discretization parameters
L: int = 2
Nx: int = 200
discretization: npt.NDArray = np.array([Nx], dtype=int)
tmin: float = 0.0 
tmax: float = 1500
Nt: int = tmax*10

# Model parameters
F: float = 0.046
k: float = 0.063
Du: float = 2e-5
Dv: float = 1e-5

# Initial condition
x: npt.NDArray = np.linspace(0, L, Nx, endpoint=False)
upart: npt.NDArray = np.ones((Nx, ))
vpart: npt.NDArray = np.zeros_like(upart)
perturb: npt.NDArray = np.exp(-25*np.power(x-1, 2))*0.5
upart = upart - perturb
vpart = vpart + perturb
u0: npt.NDArray = np.hstack((upart, vpart))

# Make PDE object
GS: GrayScott = GrayScott(discretization, L, Du, Dv, F, k)

# Make time stepper. Can be IMEXEuler, IMEXSP or IMEXTrap
imex1: IMEXEuler = IMEXEuler(GS)

# Integrate
imex1.integrate(tmin, tmax, Nt, u0)

# Plot
imex1.plot(discretization, 0, L)
imex1.plot(discretization, -1, L)
# imex1.plotAnimation(discretization, L)