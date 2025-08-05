function varargout = spm_prf_fcn_numerosity(P,M,U,varargin)
% Template pRF response function. This example models neuronal
% activity as a scaled version of the inputs:
%
% z(t) = alpha * u(t)
%
% Where alpha is a parameter estimated from the data.
% 
%
% Inputs:
%
% P      - parameter structure
% M      - model structure
% U      - experimental timing
% action - (optional) the action to performm. If not provided, the
%          predicted BOLD and neuronal timeseries are returned.
%
% -------------------------------------------------------------------------
% FORMAT [y,Z] = spm_prf_fcn_template(P,M,U)
% Return the BOLD and neuronal predicted timeseries
%
% P         parameters
% M,U       model, inputs
%
% y         fMRI time series
% Z         neuronal response
% -------------------------------------------------------------------------
% FORMAT P = spm_prf_fcn_template(P,M,U,'get_parameters')
% Return the given parameters corrected for display
%
% P         parameters
% M,U       model, inputs
% -------------------------------------------------------------------------
% FORMAT S = spm_prf_fcn_template(P,M,U,'get_summary')
% Summarises the pRF with simple (Gaussian) parameters x,y,width,beta
%
% S         structure with fields x,y,width,beta
% M,U       model, inputs
% -------------------------------------------------------------------------
% FORMAT tf = spm_prf_fcn_template(P,M,U,'is_above_threshold',Cp,v)
% Return whether the model with parameters P and covariance Cp passes an
% arbitrary threshold for display
%
% P         parameters
% M,U       model, inputs
% Cp        parameter covariance matrix
% v         voxel index
% -------------------------------------------------------------------------
% FORMAT x = spm_prf_fcn_template(P,M,U,'get_response',xy)
% Return the instantaneous response of the PRF at coordinates xy
%
% P         parameters
% M,U       model, inputs
% xy        [2xn] vector of coordinates to evaluate the PRF
% -------------------------------------------------------------------------
% FORMAT [pE,pC] = spm_prf_fcn_template(P,M,U,'get_priors')
% Return the priors for the model. Importantly, this defines which
% parameters are in the model.
%
% pE        structure or vector of prior expectations
% pC        prior covariance maitrx
%
% ---------------------------------------------------------------------
% Copyright (C) 2016 Peter Zeidman
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.   
% ---------------------------------------------------------------------
%
% Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
% 2023-07-06, 17:50: first version
% 2023-07-13, 09:37: changed priors and parameters
% 2023-08-17, 12:26: replace beta by latent beta


if nargin < 4
    % Integrate the model over time and return neuronal timeseries z and 
    % BOLD timeseries y. As an example, here we have neuronal model 
    % z(t) = alpha, where alpha is an estimated parameter.
    
    % Number of volumes = inputs
    n  = length(U); 

    % Neural timeseries. A vector with one entry per microtime bin. The 
    % U.nbins field is injected automatially by spm_prf_analyse
    z = zeros(1,U(1).nbins);
        
    for t = 1:n    
        % Microtime index for this volume
        ind = U(t).ind;        
        
        % pRF response
        z(ind) = exp(P.beta_lat) .* exp( -1/2 * (log(U(t).num) - P.mu_log).^2 ./ exp(P.sig_lat)^2 );
        % z(ind) = P.alpha;
    end
            
    % Integrate the BOLD model
    Z.u  = z';
    Z.dt = M.dt;
    y    = spm_int(P,M,Z);      
    
    varargout{1} = y;
    varargout{2} = Z;
else
    % This section of the code provides information on the model,
    % primarily for plotting purposes in spm_prf_review().
    
    % extract requested action
    action = varargin{1};

    switch action         
        case 'get_parameters'
            % Get the parameters with any corrections needed for display            
            % varargout{1} = P;
            
            % correct neuronal parameters
            % Note: mu_log and sig_log are not reparametrized into
            % mu and fhwm here; the latent beta is also not exponentiated.
            % For this purpose, use 'get_summary'.
            
            % correct hemodynamic parameters
            P.transit = exp(P.transit);
            P.decay   = exp(P.decay);
            P.epsilon = exp(P.epsilon);     
            
            % return parameters
            varargout{1} = P;
            
        case 'get_summary'
            % Get a summary of the pRF shape under Gaussian assumptions
            % varargout{1} = struct('x',P.x,'y',P.y,'width',P.width,'beta',P.beta);
            
            % correct neuronal parameters
            beta = exp(P.beta_lat);
            mu   = exp(P.mu_log);
            fwhm = exp(P.mu_log + sqrt(2*log(2))*exp(P.sig_lat)) - ...
                   exp(P.mu_log - sqrt(2*log(2))*exp(P.sig_lat));
            
            % return parameter structure
            varargout{1} = struct('beta',beta,'mu',mu,'fwhm',fwhm);
            
        case 'is_above_threshold'
            % Return binary vector identifying whether each voxel is
            % above some threshold for display            
            % varargout{1} = 1;
            
            % display each voxel
            varargout{1} = true;            
            
        case 'get_response'            
            % Return the prediction of the model at coordinates xy            
            % xy = varargin{2};
            
            % extract numerosities
            num = varargin{2};
            num = sqrt(sum(num.^2,2));
            
            % compute responses
            resp = zeros(size(num));
            for i = 1:numel(num)
                resp(i) = exp(P.beta_lat) * exp( -1/2 * (log(num(i)) - P.mu_log)^2 / exp(P.sig_lat)^2 );
            end;
            varargout{1} = resp;
            
        case 'get_priors'
            % Return a structure containing the priors
            % pE.alpha = 1;
            % pC.alpha = 1;
            % varargout{1} = pE;
            % varargout{2} = pC;
            
            % specify neuronal priors
            pE.mu_log   = 0;     pC.mu_log   = 1;
            pE.sig_lat  =-1;     pC.sig_lat  = 1;
            pE.beta_lat =-2;     pC.beta_lat = 5;
            varargout{1} = pE;
            varargout{2} = pC;
            
        case 'glm_initialize'                        
            % (Optional) Return parameters initialized using some
            % rapid initial search on timeseries y
            % y = varargin{2};
            
            % extract time series
            y = varargin{2};
            
            % initialize parameters
            mu       = [0.5:0.5:5.5];
            mu_log   = log(mu);
            sig_log  = [0.1:0.1:1];
            sig_lat  = log(sig_log);
            beta_lat = 0;
            
            % preallocate results
            SSE = zeros(numel(mu_log),numel(sig_lat));
            
            % perform grid search
            for i = 1:numel(mu_log)
                for j = 1:numel(sig_lat)
                    
                    % compute neuronal response
                    z = zeros(1,U(1).nbins);
                    for t = 1:numel(U)
                        ind    = U(t).ind;     
                        z(ind) = exp(beta_lat) * exp( -1/2 * (log(U(t).num) - mu_log(i))^2 / exp(sig_lat(j))^2 );
                    end;
                    
                    % convolve with canonical HRF
                    XU.u       = z';
                    XU.dur     = 0;
                    XU.dt      = U(1).dt;
                    XU.name{1} = 'NumpRF';
                    xBF.name   = 'hrf';
                    xBF.order  = 1;
                    xBF.length = 32;
                    xBF.dt     = U(1).dt;
                    xBF        = spm_get_bf(xBF);    
                    z = spm_Volterra(XU, xBF.bf, 1);
                    
                    % downsample regressor
                    TR  = 2.1;
                    mtr = TR/U(1).dt;
                    ind = (0:(M.ns - 1))*mtr + M.T0;
                    z   = z(ind,:);
                    X   = [z, ones(M.ns,1)];
                    
                    % estimate GLM
                    b_est    = pinv(X) * y;
                    e_est    = y - X*b_est;
                    SSE(i,j) = e_est'*e_est;
                    
                end;
            end;
            
            % find initial parameters
            [SSE_min, ind] = min(SSE(:));
            [ii, jj]       = ind2sub(size(SSE), ind);
            pE.mu_log   = mu_log(ii);
            pE.sig_lat  = sig_lat(jj);
            pE.beta_lat = -2;               % see 'get_priors'

            % return initial parameters
            varargout{1} = pE;
            
        otherwise
            error('Unknown action');
    end
end