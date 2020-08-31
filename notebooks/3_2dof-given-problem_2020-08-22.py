import src.rotor_problem as rp
import src.signal_processing as sigproc
import numpy as np

# Define parameters for 2DOF system
# ======================================================================================================================
I1 = 0.1  # kgm^2
I2 = 0.2  # kgm^2
c = 0.05  # Nms/rad
k = 2500  # Nm/rad

M = np.array([[I1, 0], [0, I2]])  # Mass matrix
K = np.array([[k, -k], [-k, k]])  # Stiffness matrix
C = np.array([[c, 0], [0, c]])  # Damping matrix

# Define the operating conditions
# ======================================================================================================================
mu = 4.7  # [ev/rev] kinematics of fundamental excitation
T_0 = 10 # [N/m] Applied torque
T_theta_amplitude = 1#T_0/10  # Amplitude of cyclic excitation

w0 = 20  # [rad/s]
dw = 40
print("Excitation frequency @ ", mu*w0/(2*np.pi)," -> ", mu*(w0+dw)/(2*np.pi),"Hz")
operating_conditions = rp.OperatingConditions(w0, dw, T_0, T_theta_amplitude, mu)

# Define initial conditions
# ======================================================================================================================
# start at w0
R0 = np.array([T_0/k, 0, w0, w0])  # [theta_1,theta_2,theta_dot_1,theta_dot_2]

# Define a lumped mass model from the selected model parameters
# ======================================================================================================================
lmm = rp.LMMsys(M, C, K, operating_conditions)

# Define initial conditions and other solver parameters
# ======================================================================================================================
x_range = np.linspace(0, 90, 2 ** 16)  # Simulate for theta 0 -> 50 rad
solver_parameters = {"x_range": x_range,
                     "initial_condition": R0,
                     "method": "RK45"}

# Solve for accelerations
# ======================================================================================================================
sol = lmm.solve_sys(solver_parameters)  # Get displacements and velocities
gamma = lmm.get_gamma(sol, solver_parameters)  # Get accelerations

# Get the eigen frequencies of the model
# ======================================================================================================================
damped_natural_frequency = lmm.get_damped_natural_frequencies()
print("damped natural frequencies [Hz]: ", damped_natural_frequency)

# Create a signal processing object to investigate the response of mass 1
# ======================================================================================================================
omega_0 = sol[2, :]
gamma = gamma[0, :]
proc = sigproc.SignalProcessing(x_range, sol[2, :], gamma, operating_conditions)


# proc.plot_signal()  # Plot the time domain signal
proc.show_time_domain_resonance(2*np.pi*damped_natural_frequency/mu,
                                save_name = "angle_domain.pdf")


# Compute natural frequencies
# =====================================================================================================================
omega_0 = sol[2][0]  # The initial rotational speed
omega_f = sol[2][-1]  # The final rotational speed
resonance_in_angle_domain_l = damped_natural_frequency * 2 * np.pi / omega_f
resonance_in_angle_domain_h = damped_natural_frequency * 2 * np.pi / omega_0
print("resonance located between,", resonance_in_angle_domain_l, " and", resonance_in_angle_domain_h, "ev/rev")
resonance_band = [resonance_in_angle_domain_l,resonance_in_angle_domain_h]
proc.plot_fft(xlim_max=15,
              resonance_band = resonance_band,
              excitation_mu=mu,
              save_name="baseline_fft.pdf")  # Plot the frequency spectrum
