%% ECODRIVE 2020
%
% (c) Adeline Bourdon + Didier Rémond
%
% Simulation of simple mechanical system
%
% -------------------------------------------------------------------------
%
% --- function for state space integration
%       augmented version of space vector R with theta or time
%       resolution is performed aither in time or in angle domain
%
function dR=sys_diff(x,R)
%
global A B
global mu f
global typ
%
if typ=='Time'
    F=f*cos(mu*R(3));               % excitation load at time t R(3)=theta(t)
else
    F=f*cos(mu*x);                  % excitation load at location x=theta
end
%
    dR(1:2,1)=A*R(1:2,1)+F*B;
    dR(3)=omega(x);
%
if typ=='Angl'
    dR=1/omega(x)*dR;
end
end