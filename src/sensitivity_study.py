import scipy.optimize as opt
import os
import numpy as np
import matplotlib.pyplot as plt
import src.rotor_problem as rp
import src.signal_processing as sigproc
import numpy as np



def response_surface(parameters, constants_dict):
    # See "3_2dof-given-problem for wel documented code for what follows
    grad_eval_loc = constants_dict["parameters_at_gradient_evaluation"] # The location in the parameters space where
    # gradient is computed
    I1 = parameters[0]*grad_eval_loc[0]  # kgm^2
    I2 = parameters[1]*grad_eval_loc[1] # kgm^2
    c =  parameters[2]*grad_eval_loc[2] # Nms/rad
    k = parameters[3]*grad_eval_loc[3]  # Nm/rad

    M = np.array([[I1, 0], [0, I2]])  # Mass matrix
    K = np.array([[k, -k], [-k, k]])  # Stiffness matrix
    C = np.array([[c,  0], [0,  c]])  # Damping matrix

    # Define the operating conditions
    # ======================================================================================================================
    mu = 5.71  # [ev/rev] kinematics of fundamental excitation
    T_0 = 1  # [N/m] Applied torque
    T_theta_amplitude = 1  # Amplitude of cyclic excitation

    w0 = 50  # This means excitation frequency is @ mu*w0/(2*pi) -> mu*(w0+dw)/(2*pi), 45Hz -> 63Hz
    dw = 20
    operating_conditions = rp.OperatingConditions(w0, dw,T_0,T_theta_amplitude, mu)

    # Define initial conditions
    # ======================================================================================================================
    R0 = np.array([0,0,20,20])  # [theta_1,theta_2,theta_dot_1,theta_dot_2]

    # Define a lumped mass model from the selected model parameters
    # ======================================================================================================================
    lmm = rp.LMMsys(M, C, K, operating_conditions)

    # Define initial conditions and other solver parameters
    # ======================================================================================================================
    x_range = np.linspace(0, 50, 2 ** 14)  # Simulate for theta 0 -> 50 rad
    solver_parameters = {"x_range": x_range,
                         "initial_condition": R0,
                         "method": "RK45"}

    # Solve for accelerations
    # ======================================================================================================================
    gamma = lmm.get_gamma(solver_parameters)

    # Create a signal processing object to investigate the response of mass 1
    # ======================================================================================================================
    proc = sigproc.SignalProcessing(x_range, gamma[0, :])
    return proc.get_max_mag() # Return the maximum acceleration value



def compute_grad_for_sys(constants_dict):

    eps = np.sqrt(np.finfo(float).eps)  # Increment used to determine finite differences
    grad = opt.approx_fprime(np.ones(np.shape(constants_dict["parameters_at_gradient_evaluation"])),
                             response_surface,
                             eps,
                             constants_dict)
    return grad

def compute_grad_for_sys_param_change(save_name = None):
    param_range = np.linspace(0,0.1,10)

    #pltlist = [opt.approx_fprime(np.array([I1+par0,I2,c,k]),testfunc,eps) for par0 in param_0_range]
    plt_list = []
    for param in param_range:
        constants_dict = {"parameters_at_gradient_evaluation": np.array([I1+param,I2,c,k])}
        plt_list.append(compute_grad_for_sys(constants_dict))

    plt.figure()
    plt.plot(plt_list)
    plt.legend(["I1","I2","c","k"])

    if save_name:
        repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plt.savefig(repo_dir + "\\reports\\" + save_name)
    plt.show()
    return plt_list


I1 = 0.1  # kgm^2
I2 = 0.2  # kgm^2
c = 0.05  # Nms/rad
k = 2500  # Nm/rad
parameters_at_gradient_evaluation = np.array([I1, I2, c, k])

constants_dict = {"parameters_at_gradient_evaluation": parameters_at_gradient_evaluation}
print(compute_grad_for_sys(constants_dict))

