import scipy.integrate as inter
import numpy as np


class LMMsys(object):
    """
    Lumped mass model object for used to define system equations for the example given in Matlab
    """

    def __init__(self, M, C, K, oper_cond_obj):
        """

        Parameters
        ----------
        M: Numpy array 2nx2n, mass matrix
        C:  Numpy array 2nx2n, damping matrix
        K:  Numpy array 2nx2n, stiffness matrix
        f:  Time dependent function f(t) = Numpy array 2nx1
        oper_cond_obj: Object from OpperatingConditions class

        where n is the number of degrees of freedom of the system
        """

        # Assign attributes to the lumped mass model object
        self.M = M
        self.C = C
        self.K = K
        self.oper_cond = oper_cond_obj

        self.dof = np.shape(self.M)[0]  # number of degrees of freedom

        # Pre-compute a few constant for speedup
        self.M_inv = np.linalg.inv(M)  # Inverse of mass matrix
        c_over_m = np.dot(self.M_inv, self.C)
        k_over_m = np.dot(self.M_inv, self.K)

        A = np.zeros((self.dof * 2, self.dof * 2))  # State space system A matrix
        A[0:self.dof, self.dof:] = np.eye(self.dof)  # Insert the elements of system A matrix
        A[self.dof:, 0:self.dof] = -k_over_m
        A[self.dof:, self.dof:] = -c_over_m
        self.A = A

        B = np.zeros((2 * self.dof, self.dof))  # State space system B matrix
        B[self.dof:] = self.M_inv
        self.B = B
        return

    def f(self,x):
        """
        Force magnitude as function of integration variable x
        Parameters
        ----------
        x: The integration variable (time/angle)

        Returns
        -------
        fvec: A vector of forces

        """

        fvec = np.zeros((self.dof, 1))
        #fvec[0,0] = self.oper_cond.T_theta_amplitude * np.cos(self.oper_cond.mu * x) + self.oper_cond.T_0
        fvec[0,0] = self.oper_cond.get_applied_T(x)
        # cos(2*pi * (ev/rev) * (rev/(2*pi)) * x) , therefore cos(mu*x)
        return fvec

    def F(self, x):
        """
        Computes the F vector for the state-space formulation
        ----------
        t  : Float
             Time

        Returns
        -------
        F  : 2xn with n the number of degrees of freedom

        """
        F = np.zeros((self.dof, 1))
        F[0:self.dof, :] = self.f(x)

        return F

    def R_dot(self, x, R):
        """
        A function used for integration of the differential equation
        Parameters
        ----------
        x: The integration variable
        R: The state vector

        Returns
        -------
        Rdot: The time derivative of the state vector
        """
        return (np.dot(self.A, R) + np.dot(self.B, self.F(x))) / R[2]  # R[2] is omega(theta_1)

    def solve_sys(self, solver_parameters):
        """
        Function for solving the R_dot function
        Parameters
        ----------
        solver_parameters: A dictionary of solver parameters

        Returns
        -------
        sol.y: A numpy array of n rows of the solution of the differential equation
        """
        x_range = solver_parameters["x_range"]
        initial_condition = solver_parameters["initial_condition"]
        method = solver_parameters["method"]

        sol = inter.solve_ivp(self.R_dot,
                              [x_range[0], x_range[-1]],
                              initial_condition,
                              method=method,
                              dense_output=True,
                              t_eval=x_range,
                              vectorized=True,
                              rtol=1e-6,
                              atol=1e-9)

        self.sol = sol
        return sol.y

    def get_gamma(self, sol, solver_parameters):
        """
        Calculates accelerations from computed displacements and velocities
        Parameters
        ----------
        solver_parameters: A dictionary of solver parameters

        Returns
        -------
        gamma: Angular accelerations
        """
        # sol = self.solve_sys(solver_parameters)

        T1 = np.dot(self.K, sol[0:self.dof, :])
        T2 = np.dot(self.C, sol[self.dof:, :])
        T3 = np.array([self.f(x) for x in solver_parameters["x_range"]])[:, :, 0].T

        return np.dot(self.M_inv, -T1 - T2 + T3)

    def get_damped_natural_frequencies(self):
        """
        Computes the damped natural frequencies from the eigenfrequencies [Hz]
        Returns
        -------

        """
        val, vec = np.linalg.eig(self.A)
        freqs = np.imag(val)/(2*np.pi)
        return freqs[0]


class OperatingConditions(object):
    """
    Class for specifying the operating conditions of the lumped mass model.
    """

    def __init__(self, w0, dw, T_0, T_theta_amplitude, mu):
        self.w0 = w0
        self.dw = dw
        self.T_0 = T_0
        self.T_theta_amplitude = T_theta_amplitude
        self.mu = mu

    def get_applied_T(self,x):
        """
        Get the total torque applied at a given angle
        Parameters
        ----------
        x: The integration variable, angle in this case.

        Returns
        -------
        T

        """

        return self.T_theta_amplitude * np.cos(self.mu * x) + self.T_0
