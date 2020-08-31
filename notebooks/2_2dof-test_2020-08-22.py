import src.state_space_sys as ss  # Script for defining state space models
import numpy as np

# Define parameters for 2DOF system
# ======================================================================================================================
m1 = 1
m2 = 0.8
c1 = 50
c2 = 50
c3 = 50
k1 = 1e6
k2 = 1.2e6
k3 = 0.3e6

M = np.array([[m1,0],[0,m2]])
K = np.array([[k1+k2,-k2],[-k2,k2+k3]])
C = np.array([[c1+c2,-c2],[-c2,c2+c3]])


# Define the operating conditions
# ======================================================================================================================
mu = 5.71  # kinematics of fundamental excitation
f_amplitude = 10  # force amplitude

w0 = 50
dw = 20
operating_conditions = ss.OperatingConditions(w0, dw, f_amplitude, mu)


# Define initial conditions
# ======================================================================================================================
R0 = np.zeros(2*len(M))

# Define a lumped mass model from the selected model parameters
# ======================================================================================================================
lmm = ss.LMMsys(M, C, K, operating_conditions)

# Define initial conditions and other solver parameters
# ======================================================================================================================
x_range = np.linspace(0, 50, 2**14) # Simulate for theta 0 -> 50 rad
solver_parameters = {"x_range": x_range,
                     "initial_condition": R0,
                     "method": "RK45"}

# Solve for accelerations
# ======================================================================================================================
gamma = lmm.get_gamma(solver_parameters)

# Create a signal processing object to investigate the response of mass 1
# ======================================================================================================================
proc = ss.SignalProcessing(x_range,gamma[0,:])

proc.plot_signal()  # Plot the time domain signal
proc.plot_fft(xlim_max=None)  # Plot the frequency spectrum

# Get the eigen frequencies of the model
# ======================================================================================================================
ef =lmm.get_eigen_freqs()
print("eigen frequencies: ", ef)

alpha = -np.real(ef[1]/abs(ef[1]))
print("alpha", alpha)

x_max = x_range[-1]
resonance_in_angle_domain_l = np.imag(ef[0])/lmm.oper_cond.omega(x_max)
resonance_in_angle_domain_h = np.imag(ef[0])/lmm.oper_cond.w0
print(resonance_in_angle_domain_l)
print(resonance_in_angle_domain_h)
