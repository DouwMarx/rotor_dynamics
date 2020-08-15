%% ECODRIVE 2020
%
% (c) Adeline Bourdon + Didier Rémond
%
% Simulation of simple mechanical system with cyclic excitation
%
% -------------------------------------------------------------------------
%
clear all
close all
clc
%
global A B M K C
global f mu
global w0 dw
global typ
%
% --- model characteristics
%
M=1;                            % mass of single DOF (kg)
K=1E6;                          % stiffness (N/m)
C=50;                           % damping (Ns/m)
f=10;                           % force amplitude
mu=5.71;                        % kinematics of fundamental excitation
w0=50;                          % rotational speed of the shaft
dw=20;                          % speed ramp
%typ='Time';                     % integration type 'Angl' or 'Time'
typ='Angl';
%
% --- state space representation
%
A=[0 1;-K/M -C/M];
B=[0; 1/M];
%
% --- modal characteristics
%
[Vec Lamb]=eig(A);
alpha=-real(Lamb(1))/abs(Lamb(1))
%
% ---- Initial conditions
%
R0=zeros(3,1);                  % initial conditions
Xm=50;                         % simulation duration (s) or (rad)
%Xm=1;                           % simulation duration (s) or (rad)
%
% ---- Resolution
%
option=odeset('maxstep',1E-3);
[X R]=ode45(@sys_diff,[0 Xm],R0,option);
%
% ---- aceleration estimation
%      gamma=1/M*(-C*x'-K*x+F(theta)
%
if typ=='Time'
    gamma=1/M*(-C*R(:,2)-K*R(:,1)+f*cos(mu*R(:,3)));
else
    gamma=1/M*(-C*R(:,2)-K*R(:,1)+f*cos(mu*X));
end
%
figure
plot(X, R)

figure
plot(X,gamma)
grid on
if typ=='Time'
    xlabel('Time (s)')
else
    xlabel('Angle (rad)')
end
ylabel('Acceleration m/s²')
%
% --- Definition of time history for FFT (avoiding transient behavior)
%
xmin=0;                         %  end of transient state (s)
ind=find(X>xmin);
%
Xu=X(ind);                      % useful time
Xu=Xu-Xu(1);                    % set initial time to 0
gammaU=gamma(ind);              % useful acceleration signal
%
figure
if typ=='Time'
    plot(Xu,w0+dw*Xu)
    xlabel('Time (s)');
    ylabel('rotational speed(rad/s)')
else
    plot(Xu,(w0+sqrt(w0*w0+4*dw*Xu))/2)
    xlabel('Angle (rad)');
    ylabel('rotational speed(rad/s)')
end
%
% --- sampling of signal
%
Xmax=max(Xu);                   % duration of signal
n=14;
N=2^n;                          % number of measurment points 
dx=Xmax/N;                      % sampling period
%
Xe=[0:N-1]*dx;                  % time vector
%
gammaE=interp1(Xu,gammaU,Xe);   % interpolation of gamma signal at sampling points
%
% ---- FFT calculation and vizualisation
%
Gamma=fft(gammaE)/N;
%
if typ=='Time'                  % frequency resolution
    df=1/Xmax;
else
    df=1/Xmax*2*pi();
end
Fe=[0:N-1]*df;                  % frequency vector
%
figure
semilogy(Fe(1:N/2),2*abs(Gamma(1:N/2)))
grid on
if typ=='Time'
    xlabel('Frequency (Hz)');
    title({'Spectral amplitude' ; ['Resonance located at ',num2str(imag(Lamb(1))/2/pi(),6),' Hz']})
    set(gca,'xlim',[0 300]);
else
    xlabel('Angle frequency (ev/rev)');
    title({'Spectral amplitude' ; ['Resonance located between ',num2str(2*imag(Lamb(1))/(w0+sqrt(w0*w0+4*dw*Xmax)),6),' and ',num2str(imag(Lamb(1))/w0,6),' ev/rev']})
    set(gca,'xlim',[0 30]);
end
ylabel('Amplitude ');
