import src.sdof_example as sdofe  # Script for defining state space models
import src.signal_processing as sigproc
import numpy as np

# Define parameters for SDOF system
# ======================================================================================================================
M = np.array([[1]])
K = np.array([[1e6]])
C = np.array([[50]])

# Define the operating conditions
# ======================================================================================================================
mu = 5.71  # kinematics of fundamental excitation
f_amplitude = 10  # [N]  # force amplitude

w0 = 50  # [rad/s] This means excitation frequency is @ mu*w0/(2*pi) -> mu*(w0+dw)/(2*pi), 45Hz -> 63Hz
dw = 20  # [rad/s]

operating_conditions = sdofe.OperatingConditions(w0, dw, f_amplitude, mu) # Create a operating condition object


# Define initial conditions
# ======================================================================================================================
R0 = np.zeros(2)

# Define a lumped mass model from the selected model parameters
# ======================================================================================================================
lmm = sdofe.LMMsys(M, C, K, operating_conditions)

# Define initial conditions and other solver parameters
# ======================================================================================================================
x_range = np.linspace(0, 50, 2**14) # Simulate for theta 0 -> 50 rad
solver_parameters = {"x_range": x_range,
                     "initial_condition": R0,
                     "method": "RK45"} # Use 4/5th order Runge Kutta integration

# Solve for accelerations
# ======================================================================================================================
sol = lmm.solve_sys(solver_parameters)
gamma = lmm.get_gamma(solver_parameters)


# Create a signal processing object to investigate the response of mass 1
# ======================================================================================================================
proc = sigproc.SignalProcessing(x_range,sol[0,:], gamma[0,:], operating_conditions,id_at_wf=False)

proc.plot_signal()  # Plot the time domain signal
proc.plot_fft(xlim_max=30)  # Plot the frequency spectrum

# Get the eigen frequencies of the model
# ======================================================================================================================
damped_natural_frequencies =lmm.get_damped_natural_frequencies()
print("damped natural frequencies [Hz]: ", damped_natural_frequencies)

# Compute natural frequencies
# =====================================================================================================================
x_max = x_range[-1]
resonance_in_angle_domain_l = damped_natural_frequencies[0]*2*np.pi/lmm.oper_cond.omega(x_max)
resonance_in_angle_domain_h = damped_natural_frequencies[0]*2*np.pi/lmm.oper_cond.w0
print("resonance located between,",resonance_in_angle_domain_l, " and",resonance_in_angle_domain_h,"ev/rev")
