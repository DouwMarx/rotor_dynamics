import numpy as np
import os
import matplotlib.pyplot as plt

class SignalProcessing(object):
    def __init__(self, xrange, omega, gamma, operating_condition_object):
        wi = operating_condition_object.w0
        dw = operating_condition_object.dw

        wf = wi + dw

        index_at_wf = np.argmax(omega>wf)  # Find the index at which the desired speed is reached
        #print("Angle range was ", 100*(1-(index_at_wf/len(xrange))),"% too long")
        # This will give an error if simulation time is nog long enough



        self.xrange = xrange[0:index_at_wf]
        self.omega = omega[0:index_at_wf]
        self.gamma = gamma[0:index_at_wf]
        self.fs = 1 / np.average(np.diff(self.xrange))

    def plot_signal(self):
        plt.figure()
        plt.plot(self.xrange, self.gamma)
        plt.xlabel("Angle [rad]")
        plt.ylabel("Acceleration [$m/s^2$]")
        plt.show()

    def show_time_domain_resonance(self, omega_nat_div_mu, save_name=None):

        fig, axs = plt.subplots(2)
        axs[0].plot(self.xrange, self.omega)
        axs[0].set_ylabel(r"$\dot \theta_1$")
        angle_at_nat_freq = self.xrange[np.argmax(self.omega > omega_nat_div_mu)]
        axs[0].hlines(omega_nat_div_mu, self.xrange[0],angle_at_nat_freq, colors="k",linestyles="--")
        axs[0].text(0.33*angle_at_nat_freq,
                    1.1*omega_nat_div_mu,
                    r"$\dot \theta_1 = \frac{\omega_d}{\mu}$",
                    fontsize = 14)
        min_omega = np.min(self.omega)*0.9
        axs[0].vlines(angle_at_nat_freq,min_omega, omega_nat_div_mu, colors="k",linestyles="--")
        axs[0].set_ylim(min_omega,None)

        axs[1].plot(self.xrange, self.gamma)
        axs[1].set_ylabel(r"$\ddot \theta_1$")
        gamma_max = np.max(self.gamma)
        gamma_min = np.min(self.gamma)

        axs[1].vlines(angle_at_nat_freq, gamma_min, gamma_max, colors="k",linestyles="--")
        plt.ylim(gamma_min,gamma_max)

        plt.xlabel('Angle [rad]')
        plt.setp(axs[0].get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=.0)
        #plt.margins(x=0)
        if save_name:
            repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            plt.savefig(repo_dir + "\\reports\\" + save_name)
        plt.show()
        return



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

    def plot_fft(self, xlim_max,excitation_mu = None, resonance_band = None, save_name = None):
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
        plt.semilogy(freq * np.pi * 2, 2*mag)  # mag is single-sided magnitude
        plt.xlim(0, xlim_max)
        #plt.ylim
        plt.ylabel("Spectral Amplitude")
        plt.xlabel("Angle Frequency [ev/rev]")
        plt.grid()


        if resonance_band:
            max_height = np.max(2 * mag)
            min_height = 0.095
            plt.vlines(resonance_band,
                       min_height,
                       max_height,
                       linestyles='--',
                       colors='k',
                       zorder=10,
                       label= "Expected \n resonance \n band")
        if excitation_mu:
            max_height = np.max(2 * mag)
            min_height = 0.095
            plt.vlines(excitation_mu,
                       min_height,
                       max_height,
                       linestyles='-.',
                       colors='k',
                       zorder=10,
                       label= "Fundamental \n excitation")
            plt.ylim(min_height,max_height)
        plt.legend()

        if save_name:
            repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            plt.savefig(repo_dir + "\\reports\\" + save_name)
        plt.show()
        return

    def get_max_mag(self):
        return np.max(np.abs(self.gamma))

    def get_rms(self):
        return np.sqrt(np.mean(self.gamma**2))
