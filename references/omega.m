%% ECODRIVE 2020
%
% (c) Adeline Bourdon + Didier Rémond
%
% Simulation of simple mechanical system
%
% -------------------------------------------------------------------------
%
% --- function for speed of rotating machine corresponding to an excitation
%       
function w=omega(x)
global w0 dw
global typ
if typ=='Time'
    w=w0+dw*x;                      % linear evolution of omega with time
else
    w=(w0+sqrt(w0*w0+4*dw*x))/2;    % corresponds to a linear evolution of
end                                 % omega with time expressed wrt angle