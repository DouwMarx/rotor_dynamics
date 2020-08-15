import src.state_space_sys as ss  # Script for defining state space models
import numpy as np
import matplotlib.pyplot as plt

# Define parameters for SDOF system
# ======================================================================================================================
M = np.array([[1]])
C = lambda t: np.array([[10]])
K = lambda t: np.array([[200]])
f = lambda t: np.array([[20*np.cos(t*2*np.pi*10)]])

# Define a lumped mass model from the parameters
# ======================================================================================================================
lmm1dof = ss.LMM_sys(M, C, K, f)

# Define initial conditions and other solver parameters
# ======================================================================================================================
t_range = np.linspace(0, 1, 1000)

X0 = np.array([-0.5])
Xd0 = np.array([0])
init_cond = np.hstack((X0, Xd0))

#de = ss.FirsOrderDESys(lmm1dof.E_Q, init_cond, t_range)
de = ss.TimeDomainSys(lmm1dof.E_Q, init_cond, t_range)
s = de.solve("RK45")  # Solve differential equation using Runge kutta 4/5 order numerical integration

plt.figure()
plt.plot(s)
plt.show()

xdd = de.get_Xdotdot("RK45")

plt.figure()
plt.plot(xdd)
plt.show()