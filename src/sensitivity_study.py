import scipy.optimize as opt
import os
import numpy as np
import matplotlib.pyplot as plt
import src.rotor_problem as rp
import src.signal_processing as sigproc
import numpy as np
from tqdm import tqdm


def response_surface(parameters, constants_dict):
    # See "3_2dof-given-problem for wel documented code for what follows
    grad_eval_loc = constants_dict["parameters_at_gradient_evaluation"] # The location in the parameters space where
    # gradient is computed
    I1 = parameters[0]*grad_eval_loc[0]  # kgm^2
    I2 = parameters[1]*grad_eval_loc[1] # kgm^2
    c = parameters[2]*grad_eval_loc[2] # Nms/rad
    k = parameters[3]*grad_eval_loc[3]  # Nm/rad

    M = np.array([[I1, 0], [0, I2]])  # Mass matrix
    K = np.array([[k, -k], [-k, k]])  # Stiffness matrix
    C = np.array([[c,  0], [0,  c]])  # Damping matrix

    # Define the operating conditions
    # ======================================================================================================================
    mu = 4.7  # [ev/rev] kinematics of fundamental excitation
    T_0 = constants_dict["operating_conditions"]["T_0"]  # [N/m] Applied torque
    T_theta_amplitude = 1  # Amplitude of cyclic excitation

    w0 = 20  # This means excitation frequency is @ mu*w0/(2*pi) -> mu*(w0+dw)/(2*pi), 45Hz -> 63Hz
    dw = 40
    operating_conditions = rp.OperatingConditions(w0, dw,T_0,T_theta_amplitude, mu)

    # Define initial conditions
    # ======================================================================================================================
    R0 = np.array([T_0 / k, 0, w0, w0])  # [theta_1,theta_2,theta_dot_1,theta_dot_2]

    # Define a lumped mass model from the selected model parameters
    # ======================================================================================================================
    lmm = rp.LMMsys(M, C, K, operating_conditions)

    # Define initial conditions and other solver parameters
    # ======================================================================================================================
    x_range = np.linspace(0, 90, 2 ** 16)  # Simulate for theta 0 -> 50 rad
    solver_parameters = {"x_range": x_range,
                         "initial_condition": R0,
                         "method": "RK45"}


    # Create a signal processing object to investigate the response of mass 1
    # ======================================================================================================================
    sol = lmm.solve_sys(solver_parameters)
    gamma = lmm.get_gamma(sol, solver_parameters)

    omega_0 = sol[2, :]
    #gamma = gamma[0, :] # Change this to change between I_1 and I_2
    gamma = gamma[1, :]
    proc = sigproc.SignalProcessing(x_range, omega_0, gamma, operating_conditions)

    #return proc.get_max_mag()  # Return the maximum acceleration value
    return proc.get_rms()

def compute_grad_for_sys(constants_dict):
    eps = np.sqrt(np.finfo(float).eps)  # Increment used to determine finite differences
    grad = opt.approx_fprime(np.ones(np.shape(constants_dict["parameters_at_gradient_evaluation"])),
                             response_surface,
                             eps,
                             constants_dict)
    return grad

def compute_grad_for_T0_change(save_name = None):

    param_range = np.linspace(10, 20, 100)
    I1 = 0.1  # kgm^2
    I2 = 0.2  # kgm^2
    c = 0.05  # Nms/rad
    k = 2500  # Nm/rad

    plt_list = []
    for param in tqdm(param_range):
        constants_dict = {"parameters_at_gradient_evaluation": np.array([I1,I2,c,k]),
                          "operating_conditions": {"T_0": param}}
        plt_list.append(compute_grad_for_sys(constants_dict))

    plt.figure()
    plt.plot(param_range,plt_list)
    plt.legend([r"$\frac{\partial y_1}{\partial I_1}$",
                r"$\frac{\partial y_1}{\partial I_2}$",
                r"$\frac{\partial y_1}{\partial c}$",
                r"$\frac{\partial y_1}{\partial k}$"])
    plt.xlabel(r"$T_0$")
    plt.ylabel(r"Partial derivative of $y$, y=rms($\ddot \theta_1$)")

    if save_name:
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plt.savefig(repo_dir + "\\reports\\" + save_name)
    plt.show()
    return plt_list

