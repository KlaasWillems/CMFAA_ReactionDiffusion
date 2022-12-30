from timesteppers import IMEXEuler
from RDModels import GrayScott
import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import spsolve
from typing import Optional
from scipy.sparse import csc_matrix, diags
import matplotlib.pyplot as plt

# discretization parameters
L: int = 2
Nx: int = 5
Nt: int = 10
discretization: npt.NDArray = np.array([Nx, Nx], dtype=int)
tmin: float = 0.0 
tmax: float = 0.5

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

# Make time stepper
imex1: IMEXEuler = IMEXEuler(GS)

# Integrate
res = imex1.integrate(tmin, tmax, Nt, u0)
# GS.plot(imex1.time, 0, imex1.res)
# GS.plotAnimation(imex1.time, imex1.res)


# print(type(GS.K))
# print(GS.K.toarray())
# plt.spy(GS.K)
# plt.show()
# temp: npt.NDArray = np.random.rand(Nx, 1)
# spsolve(GS.K, temp)
