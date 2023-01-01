# Demo of the Gray-Scott model in 2D using the ADI method. Simulation takes a couple of minutes. After the computation, concentrations of u and v are plotted from some values of time.

from timesteppers import ADI
from RDModels import GrayScott
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import time

# discretization parameters
L: int = 2  # Simulation domain [0, L] x [0, L]
Nx: int = 128  # Grid points in either direction
discretization: npt.NDArray = np.array([Nx, Nx], dtype=int)
tmin: float = 0.0 
tmax: float = 1000
times: int = 10  # Simulation happens in stages of 1000 seconds. This is repeated 10 times. 
Nt: int = tmax*2  # Amount of timesteps

# Model parameters
F: float = 0.046
k: float = 0.063
Du: float = 2e-5
Dv: float = 1e-5

# Initial condition (see section 2.2.2.3 in textbook)
x: npt.NDArray = np.linspace(0, L, Nx, endpoint=False)
y: npt.NDArray = np.linspace(0, L, Nx, endpoint=False)
xv, yv = np.meshgrid(x, y)
x1 = 0.5; x2 = 0.55
y1 = 0.5; y2 = 0.6
p1 = np.exp( -25*((xv - x1)**2 + (yv - y1)**2))/2
p2 = np.exp( -25*((xv - x2)**2 + (yv - y2)**2))/2
umatrix = np.ones_like(p1) - p1 - p2
vmatrix = np.zeros_like(p2) + p1 + p2
u0: npt.NDArray = np.vstack((umatrix, vmatrix))

# Make PDE object
GS: GrayScott = GrayScott(discretization, L, Du, Dv, F, k)

# Make time stepper. Can be either IMEXEuler, IMEXSP or IMEXTrap
imex1: ADI = ADI(GS)

# integrate
res: npt.NDArray = np.empty((2*Nx, Nx, times))
res[:, :, 0] = u0
for i in range(1, times):
    tic = time.perf_counter()
    _ = imex1.integrate(tmin, tmax, Nt, res[:, :, i-1])
    assert imex1.res is not None
    res[:, :, i] = imex1.res[:, :, -1]
    toc = time.perf_counter()
    print(f'Progress: {i/(times-1)}%. Iteration time: {toc-tic}')

# plot 
fig = plt.figure()
for i in range(times):
    umatrix: npt.NDArray = res[:Nx, :, i]
    vmatrix: npt.NDArray = res[Nx:, :, i]

    plt.subplot(1, 2, 1)
    plt.imshow(umatrix, extent=[0, L, 0, L])
    plt.colorbar()
    plt.title('u')
    plt.subplot(1, 2, 2)
    plt.imshow(vmatrix, extent=[0, L, 0, L])
    plt.colorbar()
    plt.title('v')
    fig.suptitle(f'time = {tmax*i}')
    plt.pause(0.8)
    if i != times-1: 
        plt.show(block=False)
    else:
        plt.show(block=True)
    # plt.savefig(f'Figures/fig{i}.png')

