import matplotlib.pyplot as plt
import scipy.integrate as inter
import numpy as np
import scipy.linalg as sl



class LMMsys(object):
    """
    Lumped mass model object for second order equation of Newtons second law
    Time variable stiffness, All other parameters are constant
    """

    def __init__(self, M, C, K, f):
        """

        Parameters
        ----------
        M: Numpy array 2nx2n
        C  Numpy array 2nx2n
        K  Time dependent function K(t) = Numpy array 2nx2N
        f  Time dependent function f(t) = Numpy array 2nx1
        X0 Numpy array 2nx1
        Xd0 Numpy array 2nx1
        time_range Numpy array
        """

        self.M = M
        self.C = C
        self.K = K
        self.f = f

        self.dof = np.shape(self.M)[0]  # number of degrees of freedom

        self.M_inv = np.linalg.inv(M)  # Inverse of mass matrix
        c_over_m = np.dot(self.M_inv, self.C)
        k_over_m = np.dot(self.M_inv, self.K)

        A = np.zeros((self.dof * 2, self.dof * 2))  # State space system A matrix
        A[0:self.dof, self.dof:] = np.eye(self.dof) # Insert the elements of system A matrix
        A[self.dof:, 0:self.dof] = -k_over_m
        A[self.dof:, self.dof:] = -c_over_m

        self.A = A

        B = np.zeros((2 * self.dof, self.dof))
        B[self.dof:] = self.M_inv

        self.B = B

        return

    def F(self, x):
        """
        Converts the second order differential equation to first order (E matrix and Q vector)

        Parameters
        ----------
        t  : Float
             Time

        Returns
        -------
        E  : 2x(9+3xN) x 2x(9+3xN) Numpy array

        Based on Runge-kutta notes

        """
        F = np.zeros((self.dof,1))
        F[0:self.dof,:] = self.f(x)

        return F

    def omega(self,x):
        w0 = 50
        dw = 20

        w = (w0 + np.sqrt(w0*w0 + 4*dw*x))/2
        return w

    def R_dot(self,x,R):
        return (np.dot(self.A, R) + np.dot(self.B, self.F(x)))/self.omega(x)

    def solve_sys(self, solver_parameters):
        """
        Solve the system of differential equations
        :param method: method of solution, "RK45" "BDF"
        :return:
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

    def get_gamma(self, solver_parameters):
        """
        Calculates accelerations from computed displacements and velocities
        Parameters
        ----------
        sol

        Returns
        -------

        """
        sol = self.solve_sys(solver_parameters)

        T1 = np.dot(self.K, sol[0:self.dof, :])
        T2 = np.dot(self.C, sol[self.dof:, :])
        T3 = np.array([lmm.f(x) for x in solver_parameters["x_range"]])[:, :, 0].T

        return np.dot(self.M_inv, -T1 - T2 + T3)

    def get_eigen_freqs(self):
        val,vec = np.linalg.eig(self.A)
        return val
class SolutionProcessing(object):
    def __init__(self,xrange,gamma):
        self.xrange = xrange
        self.gamma = gamma
        self.fs = 1/np.average(np.diff(self.xrange))

    def plot_signal(self):
        plt.figure()
        plt.plot(self.xrange,self.gamma)
        plt.xlabel("Angle [rad]")
        plt.ylabel("Acceleration [$m/s^2$]")
        plt.show()

    def fft(self):
        """

        Parameters
        ----------
        data: String
            The heading name for the dataframe

        Returns
        -------
        freq: Frequency range
        magnitude:
        phase:
        """
        d = self.gamma
        length = len(d)
        Y = np.fft.fft(d) / length
        magnitude = np.abs(Y)[0:int(length / 2)]
        phase = np.angle(Y)[0:int(length / 2)]
        freq = np.fft.fftfreq(length, 1 / self.fs)[0:int(length / 2)]
        return freq, magnitude, phase

    def plot_fft(self):
        """
        Computes and plots the FFT for a given signal
        Parameters
        ----------
        data: String
            Name of the heading of the dataset to be FFTed

        Returns
        -------

        """
        freq, mag, phase = self.fft()
        plt.figure()
        plt.semilogy(freq*np.pi*2, mag, "k")
        plt.xlim(0,30)
        plt.ylabel("Spectral Amplitude")
        plt.xlabel("Angle Frequency [ev/rev]")
        plt.grid()
        plt.show()

        # plt.vlines(np.arange(1, n_to_plot) * GMF, 0, max_height, 'g', zorder=10, label="GMF and Harmonics")
        return


# M = np.array([[1, 0], [0, 1]])
# C = np.array([[50, 0], [-50, 50]])
# K = np.array([[1e6, 0], [-1e6, 1e6]])
# # f = np.array([[100, 0]])
# mu = 5.71
# f = lambda x: np.array([[10*np.cos(mu*x)],[0]])
# X0 = np.array([0,0])
# Xd0 = np.array([0,0])

# def test_1dof_system():
M = np.array([[1]])
K = np.array([[1e6]])
C = np.array([[50]])
mu = 5.71
f = lambda x: np.array([[10*np.cos(mu*x)]])

X0 = np.array([0])
Xd0 = np.array([0])

init_cond = np.hstack((X0, Xd0))

lmm = LMMsys(M, C, K, f)

x_range = np.linspace(0,50,2**14)
solver_parameters = {"x_range": x_range,
                    "initial_condition": init_cond,
                     "method": "RK45"}

acc = lmm.get_gamma(solver_parameters)

# plt.figure()
# plt.plot(sol.T)
# plt.show()

proc = SolutionProcessing(x_range,acc[0,:])
#proc.plot_signal()
proc.plot_fft()

print("eigfreqs")
e =lmm.get_eigen_freqs()

print("alpha",-np.real(e[1]/abs(e[1])))

# print("resonance @", np.real(e))
# "Resonance located between ',num2str(2*imag(Lamb(1))/(w0+sqrt(w0*w0+4*dw*Xmax))"