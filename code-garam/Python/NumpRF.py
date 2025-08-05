"""
NumpRF - numerosity population receptive modelling

Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
2023-06-22, 14:34: log2lin, lin2log
2023-06-22, 17:04: f_log, neuronal_signals, simulate
2023-06-22, 17:55: hemodynamic_signals, simulate
2023-06-22, 21:49: testing
2023-06-26, 18:11: refactoring
2023-06-29, 16:58: plot_task_signals
2023-07-03, 10:26: plot_signals_axis, plot_signals_figure, debugging
2023-07-13, 10:45: plot_signals_figure
2023-07-13, 16:58: estimate_MLE_rgs, log2lin, lin2log
2023-07-14, 17:54: estimate_MLE_rgs, hemodynamic_signals
2023-08-10, 14:07: simulate, estimate_MLE_fgs
2023-08-21, 17:50: rewriting to OOP
2023-08-23, 09:22: estimate_MLE
2023-08-24, 16:39: refactoring
2023-08-28, 16:25: hemodynamic_signals, simulate, estimate_MLE
2023-08-31, 19:38: estimate_MLE, refactoring, testing
2023-09-07, 11:24: estimate_MLE, free_parameters
2023-09-14, 14:14: estimate_MLE
2023-09-18, 17:39: estimate_MLE
2023-09-21, 11:47: estimate_MLE, free_parameters
2023-09-26, 16:46: estimate_MLE
2023-10-26, 17:27: MLL2Rsq
2023-11-02, 08:01: yp2Rsq
2023-11-02, 09:52: Rsqtest
2023-11-02, 13:51: Rsqtest
2023-11-07, 15:54: f_lin, fwhm2sigma
2023-11-23, 11:33: estimate_MLE, estimate_MLE_rgs
2023-11-27, 14:43: corrtest, Rsqtest
"""


# import packages
#-----------------------------------------------------------------------------#
import math
import numpy as np
import scipy as sp
import PySPMs as PySPM

# function: linear tuning
#-----------------------------------------------------------------------------#
def f_lin(x, mu_lin, sig_lin):
    """
    Calculate Linear Tuning Function
    y = f_lin(x, mu_lin, sig_lin)
        
        x       - array of floats; stimuli at which to evaluate tuning
        mu_lin  - float; mean of tuning in linear space
        sig_lin - float; standard deviation of tuning in linear space
        
        y       - array of floats; expected response based on tuning function
        
    y = f_lin(x, mu_lin, sig_lin) returns the value of the Gaussian tuning
    function with mean mu_lin and standard deviation sig_lin in linear
    stimuli space at the argument x.
    """
    
    # calculate function value
    y = np.exp(-1/2 * (x-mu_lin)**2 / sig_lin**2)
    return y

# function: logarithmic tuning
#-----------------------------------------------------------------------------#
def f_log(x, mu_log, sig_log):
    """
    Calculate Logarithmic Tuning Function
    y = f_log(x, mu_log, sig_log)
        
        x       - array of floats; stimuli at which to evaluate tuning
        mu_log  - float; mean of tuning in logarithmic space
        sig_log - float; standard deviation of tuning in logarithmic space
        
        y       - array of floats; expected response based on tuning function
        
    y = f_log(x, mu_log, sig_log) returns the value of the Gaussian tuning
    function with mean mu_log and standard deviation sig_log in logarithmic
    stimuli space at the argument x.
    """
    
    # calculate function value
    y = np.exp(-1/2 * (np.log(x)-mu_log)**2 / sig_log**2)
    return y

# function: logarithmic to linear
#-----------------------------------------------------------------------------#
def log2lin(mu_log, sig_log):
    """
    Transform Logarithmic Tuning Parameters to Linear Space
    mu, fwhm = log2lin(mu_log, sig_log)
        
        mu_log  - float; mean of tuning in logarithmic space
        sig_log - float; standard deviation of tuning in logarithmic space
        
        mu      - float; mean of tuning in linear space
        fwhm    - float; full width at half maximum in linear space
        
    mu, fwhm = log2lin(mu_log, sig_log) transforms logarithmic tuning parameters
    mu_log and sig_log and returns linear tuning parameters mu and fhwm.
    """
    
    # calculate mu and fwhm
    mu   = np.exp(mu_log)
    fwhm = np.exp(mu_log + math.sqrt(2*math.log(2))*sig_log) - \
           np.exp(mu_log - math.sqrt(2*math.log(2))*sig_log)
    return mu, fwhm
    
# function: linear to logarithmic
#-----------------------------------------------------------------------------#
def lin2log(mu, fwhm):
    """
    Transform Linear Tuning Parameters to Logarithmic Space
    mu_log, sig_log = log2lin(mu, fwhm)
        
        mu      - float; mean of tuning in linear space
        fwhm    - float; full width at half maximum in linear space
        
        mu_log  - float; mean of tuning in logarithmic space
        sig_log - float; standard deviation of tuning in logarithmic space
        
    mu_log, sig_log = log2lin(mu, fwhm) transforms linear tuning parameters
    mu and fhwm and returns logarithmic tuning parameters mu_log and sig_log.
    """
    
    # catch, if arrays
    if type(mu) == np.ndarray and type(fwhm) == np.ndarray:
        mu_log  = np.zeros(mu.shape)
        sig_log = np.zeros(fwhm.shape)
        for i in range(mu.size):
            mu_log[i], sig_log[i] = lin2log(mu[i], fwhm[i])
    
    # otherwise, numericals
    else:
    
        # calculate mu_log
        mu_log = math.log(mu)
        
        # calculate sig_log
        sig_log = 1                             # iterative algorithm to find sig_log
        step    = 0                             # start at 1, increase or decrease by 10^-s,
        sign    = 1                             # if the sign changes, increase s by 1
        while step < 5:
            m, f = log2lin(mu_log, sig_log)     # calculate mu, fwhm, as of now
            if f == fwhm:  
                s = 0                     # if f equal fwhm, sig_log is found
                break
            elif f < fwhm:                      # if f smaller fwhm, increase sig_log
                s = +1
            elif f > fwhm:                      # if f larger fwhm, decrease sig_log
                s = -1
            if s != sign:                       # if direction has changed, reduce step size
                step = step + 1                 # calculate new estimate for sig_log
            sig_log = sig_log + s*math.pow(10,-step)
            sign    = s
        del m, f, s
    
    # return mu_log and sig_log
    return mu_log, sig_log

# function: fwhm to sigma
#-----------------------------------------------------------------------------#
def fwhm2sig(fwhm):
    """
    Transform Tuning Width to Standard Deviation
    sig = fwhm2sig(fwhm)
        
        fwhm - float; full width at half maximum in linear space
        
        sig  - float; standard deviation of tuning in linear space
    
    sig = fwhm2sig(fwhm) transforms tuning width fwhm measured as full width at
    half maximum to standard deviation sig of tuning in linear space.
    """
    
    # calculate sigma
    sig = fwhm/(2*math.sqrt(2*math.log(2)))
    return sig

# function: MLL to R^2
#-----------------------------------------------------------------------------#
def MLL2Rsq(MLL1, MLL0, n):
    """
    Convert Maximum Log-Likelihoods to Coefficients of Determination
    Rsq = MLL2Rsq(MLL1, MLL0, n)
        
        MLL1 - 1 x v array; maximum log-likelihoods for model of interest
        MLL0 - 1 x v array; maximum log-likelihoods for model with intercept only
        n    - int; number of data points used to calculate MLL1 and MLL0
        
        Rsq  - 1 x v array; coefficients of determination for model of interest
        
    Rsq = MLL2Rsq(MLL1, MLL0, n) converts the difference in maximum log-
    likelihoods (MLL1-MLL0) to coefficients of determination ("R-squared")
    assuming linear regression models and number of observations n [1].
    
    [1] https://statproofbook.github.io/P/rsq-mll
    """
    
    # calculate R-squared
    Rsq = 1 - np.power(np.exp(MLL1-MLL0), -2/n)
    return Rsq

# function: y/y_p to R^2
#-----------------------------------------------------------------------------#
def yp2Rsq(y, yp):
    """
    Convert Predicted Time Series to Coefficients of Determination
    Rsq = yp2Rsq(y, yp)
        
        y   - n x v array; observed time series
        yp  - n x v array; predicted time series
        
        Rsq - 1 x v array; coefficients of determination
        
    Rsq = yp2Rsq(y, yp) converts the observed and predicted time series into
    coefficients of determination ("R-squared") assuming linear regression
    models [1].
    
    [1] https://statproofbook.github.io/P/rsq-der
    """
    
    # calculate R-squared
    RSS = np.sum( np.power(y-yp, 2), axis=0)
    TSS = np.sum( np.power(y-np.tile(np.mean(y,axis=0),(y.shape[0],1)), 2), axis=0)
    Rsq = 1 - RSS/TSS
    return Rsq

# function: test for R-squared
#-----------------------------------------------------------------------------#
def Rsqtest(y, yp, p=2, alpha=0.05):
    """
    Significance Test for Coefficient of Determination based on F-Test
    h, p, stats = Rsqtest(y, yp, p, alpha)
    
        y       - n x v array; observed time series
        yp      - n x v array; predicted time series
        p       - int; number of explanatory variables used to
                       predict signals from which Rsq is calculated
                       (default: 2 [intercept and slope])
        alpha   - float; significance level for the F-test
        
        h       - 1 x v vector; indicating rejectance of the null hypothesis
        p       - 1 x v vector; p-values computed under the null hypothesis
        stats   - dict; further information on statistical inference:
        o Fstat - 1 x v vector; values of the F-statistic
        o df    - list of ints; degrees of freedom for the F-statistic
    
    h, p, stats = Rsqtest(y, yp, p, alpha) performs an F-test for the
    coefficient of determination Rsq assuming predicted signals coming from
    linear regression models with p free parameters and returns a
    vector of logicals h indicating rejectance of the null hypothesis
        H0: Rsq = 0
    and the vector of p-values in favor of the alternative hypothesis
        H1: Rsq > 0
    as well as further information on the statistical test [1].
    
    [1] https://en.wikipedia.org/wiki/F-test#Regression_problems
    """
    
    # calculate residual sum of squares for R^2 model
    RSS2 = np.sum(np.power(y-yp,2), axis=0)
    
    # calculate residual sum of squares for null model
    ym   = np.tile(np.mean(y, axis=0), (y.shape[0],1))
    RSS1 = np.sum(np.power(y-ym,2), axis=0)
    
    # calculate F-statistics
    n  = y.shape[0]
    p2 = p
    p1 = 1
    F  = ((RSS1-RSS2)/(p2-p1))/(RSS2/(n-p2))
    
    # calculate p-values
    stats = {'Fstat': F, 'df': [p2-p1, n-p2]}
    p     = 1 - sp.stats.f.cdf(F, p2-p1, n-p2)
    h     = p < alpha
    
    # return test statistics
    return h, p, stats

# function: test for correlation
#-----------------------------------------------------------------------------#
def corrtest(r, n, p=2, alpha=0.05):
    """
    Significance Test for Correlation Coefficient based on t-Test
    h, p, stats = corrtest(r, n, p, alpha)
    
        r       - 1 x v array; Pearson correlation coefficients
        n       - int; number of data points used to calculate r
        p       - int; number of explanatory variables used to
                       predict signals from which r was calculated
                       (default: 2 [intercept and slope])
        alpha   - float; significance level for the t-test
        
        h       - 1 x v vector; indicating rejectance of the null hypothesis
        p       - 1 x v vector; p-values computed under the null hypothesis
        stats   - dict; further information on statistical inference:
        o tstat - 1 x v vector; values of the t-statistic
        o df    - int; degrees of freedom of the t-statistic
    
    h, p, stats = corrtest(r, n, p, alpha) performs a t-test for the
    correlation coefficients r assuming linear regression models with
    n observations, p free parameters and significance level alpha and returns
    a vector of logicals h indicating rejectance of the null hypothesis
        H0: r = 0
    and the vector of p-values in favor of the alternative hypothesis
        H1: r > 0
    as well as further information on the statistical test [1].
    
    [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Testing_using_Student's_t-distribution
    """
    
    # calculate t-statistics
    t = r * math.sqrt(n-p) / np.sqrt(1-np.power(r,2))
    
    # calculate p-values
    stats = {'tstat': t, 'df': n-p}
    p     = 1 - sp.stats.t.cdf(t, n-p)
    h     = p < alpha
    
    # return test statistics
    return h, p, stats

# function: neuronal model
#-----------------------------------------------------------------------------#
def neuronal_signals(ons, dur, stim, TR, mtr, mu_log, sig_log):
    """
    Compute Signals According to Neuronal Model
    Z, t = neuronal_signals(ons, dur, stim, TR, mtr, mu_log, sig_log)
    
        ons     - t x 1 vector; trial onsets [s]
        dur     - t x 1 vector; trial durations [s]
        stim    - t x 1 vector; trial stimuli (t = trials)
        TR      - float; repetition time of fMRI acquisition [s]
        mtr     - int; microtime resolution (= bins per TR)
        mu_log  - 1 x v vector; preferred numerosity in logarithmic space
        sig_log - 1 x v vector; tuning width in logarithmic space
        
        Z       - m x v matrix; neuronal signals tuned to stimuli
        t       - m x 1 vector; time vector at temporal resolution TR/mtr
    """
    
    # get maximum time points
    T = math.ceil(np.max(ons+dur))
    
    # calculate timing in microtime space
    dt  = TR/mtr
    ons = np.round(np.array(ons)/dt)
    dur = np.round(np.array(dur)/dt)
    
    # compute neuronal signals
    v = mu_log.size
    Z = np.zeros((math.ceil(T/dt),v))
    t = np.arange(0,T,dt)
    for o,d,s in zip(ons,dur,stim):
        Z[int(o):(int(o)+int(d)),:] = f_log(s, mu_log, sig_log)
    
    # return neuronal signals
    return Z, t

# function: hemodynamic signals
#-----------------------------------------------------------------------------#
def hemodynamic_signals(Z, t, n, mtr, mto=1, p=None, order=1):
    """
    Compute Signals According to Hemodynamic Model
    S, t = hemodynamic_signals(Z, t, n, mtr, mto, p, order)
    
        Z      - m x v matrix; neuronal signals (see "neuronal_signals")
        t      - m x 1 vector; time vector (see "neuronal_signals")
        n      - int; number of fMRI scans acquired in run
        mtr    - int; microtime resolution (= bins per TR)
        mto    - int; microtime onset (= reference slice; default: 1)
        p      - list of floats; HRF parameters (default: see "PySPM.spm_hrf")
        order  - int; order of HRF basis set (default: 1; see "PySPM.spm_get_bf")
        
        S      - n x v x order array; hemodynamic signals after convolution
        t      - n x 1 vector; time vector after temporal down-sampling
    """
    
    # get hemodynamic response function
    dt = t[1]-t[0]
    bf = PySPM.spm_get_bf(dt, 'HRF', p, order)
    
    # compute hemodynamic signals
    v = Z.shape[1]
    S = np.zeros(((Z.shape[0]+bf.shape[0]-1),v,order))
    for k in range(order):
        for j in range(v):
            S[:,j,k] = np.convolve(Z[:,j], bf[:,k])
            S[:,j,k] = S[:,j,k] / np.max(np.abs(S[:,j,k]))
    
    # down-sample signals temporally
    i = np.arange(mto-1, n*mtr, mtr)
    S = S[i,:,:]
    t = t[i]
    
    # return hemodynamic signals
    return S, t
    
# class: data set
#-----------------------------------------------------------------------------#
class DataSet:
    """
    A DataSet object is initialized by a data matrix, onsets/durations/stimuli,
    fMRI repetition time and a matrix of confound variables of no interest.
    """
    
    # function: initialize data set
    #-------------------------------------------------------------------------#
    def __init__(self, Y, ons, dur, stim, TR, X_c):
        """
        Initialize a Data Set
        ds = NumpRF.DataSet(Y, ons, dur, stim, TR, X_c)
            
            Y    - n x v x r array; measured BOLD signals (n = scans, v = voxels) OR
                   any other type; if purpose is simulation (enter e.g. 0)
            ons  - list of arrays of floats; t x 1 vectors of onsets [s]
            dur  - list of arrays of floats; t x 1 vectors of durations [s]
            stim - list of arrays of floats; t x 1 vectors of stimuli (t = trials)
            TR   - float; repetition time of fMRI acquisition [s]
            X_c  - n x c x r array; confound regressors (c = variables, r = runs)
            
            ds   - a DataSet object
            o Y    - data matrix
            o ons  - run-wise onsets
            o dur  - run-wise durations
            o stim - run-wise stimuli
            o TR   - fMRI repetition time
            o X_c  - confound matrix
        """
        
        # store data set properties
        self.Y    = Y
        self.ons  = ons
        self.dur  = dur
        self.stim = stim
        self.TR   = TR
        self.X_c  = X_c
    
    # function: simulate data set
    #-------------------------------------------------------------------------#
    def simulate(self, mu, fwhm, mu_b=10, mu_c=1, s2_k=1, s2_j=0.1, s2_i=1, tau=0.001, hrf=None):
        """
        Simulate Data across Scans, Voxels and Runs
        Y, S, X, B = ds.simulate(mu, fwhm, mu_b=1, mu_c, s2_k, s2_j, s2_i, tau, hrf)
            
            mu   - 1 x v vector; preferred numerosities in linear space (v = voxels)
            fwhm - 1 x v vector; tuning widths in linear space
            mu_b - float; expected value of signal betas
            mu_c - float; expected value of confound betas
            s2_k - float; between-voxel variance
            s2_j - float; between-run variance
            s2_i - float; within-run variance
            tau  - float; time constant, serial correlation
            hrf  - list of floats; HRF parameters (default: see "PySPM.spm_hrf")
            
            Y    - n x v x r array; simulated BOLD signals
            S    - n x v x r array; predicted numerosity signals
            X    - n x p x r array; created design matrices
            B    - p x v x r array; sampled regression coefficients
        """
        
        # part 1: create design matrix
        #---------------------------------------------------------------------#
        n = self.X_c.shape[0]       # number of scans
        c = self.X_c.shape[1]       # number of variables
        r = self.X_c.shape[2]       # number of runs
        v = mu.size                 # number of voxels
        
        # preallocate design matrices
        p = 1 + c + 1               # number of regression coefficients:
        S = np.zeros((n,v,r))       # numerosity effect + confounds + implicit baseline
        X = np.zeros((n,p,r))
        
        # specify microtime resolution
        import EMPRISE
        mtr  = EMPRISE.mtr
        mto  = EMPRISE.mto
        del EMPRISE
        
        # transform tuning parameters
        mu_log, sig_log = lin2log(mu, fwhm)
        
        # for all runs
        for j in range(r):
            
            # calculate neuronal signals
            Z, t = neuronal_signals(self.ons[j], self.dur[j], self.stim[j], \
                                    self.TR, mtr, mu_log, sig_log)
            
            # calculate hemodynamic signals
            S[:,:,[j]], t = hemodynamic_signals(Z, t, n, mtr, mto, p=hrf, order=1)
            
            # create design matrix for this run
            X[:,:,j] = np.c_[np.zeros((n,1)), self.X_c[:,:,j], np.ones((n,1))]
        
        # part 2: sample measured signals
        #---------------------------------------------------------------------#
        z = np.zeros(n)             # n x 1 zero vector
        
        # preallocate data matrices
        Y = np.zeros((n,v,r))       # measured signals
        B = np.zeros((p,v,r))       # regression coefficients
        
        # calculate temporal covariance
        V = sp.linalg.toeplitz(np.power(tau, np.arange(0,n)))
            
        # sample mean activations
        B_mean = np.r_[np.random.normal(mu_b, math.sqrt(s2_k), size=(1,v)), \
                       np.random.normal(mu_c, math.sqrt(s2_k), size=(c,v)), \
                       np.random.normal(mu_b, math.sqrt(s2_k), size=(1,v))]
        
        # for all runs 
        for j in range(r):
            
            # sample beta values
            B[:,:,j] = B_mean + np.random.normal(0, math.sqrt(s2_j), size=(p,v))
            
            # sample noise terms
            E = np.random.multivariate_normal(z, s2_i*V, size=v).T
            
            # simulate measured signal
            Y[:,:,j] = S[:,:,j] @ np.diag(B[0,:,j]) + \
                       X[:,1:,j] @ B[1:,:,j] + E
            # signal = signal due stimulus + signal due to confounds + noise
            # this is equivalent to the following formulation (but faster)
            # for k in range(v):
            #     X[:,0,j] = S[:,k,j]
            #     Y[:,:,j] = X[:,:,j]*B[:,:,j] + E
                    
        # return simulated signals
        self.Y = Y
        del Z, t, E
        return Y, S, X, B
    
    # function: maximum likelihood estimation
    #-------------------------------------------------------------------------#
    def estimate_MLE(self, avg=[False, False], corr='iid', order=1, mu_grid=None, sig_grid=None, fwhm_grid=None, Q_set=None):
        """
        Maximum Likelihood Estimation of Numerosity Tuning Parameters
        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =
            ds.estimate_MLE(avg, corr, order, mu_grid, sig_grid, fwhm_grid, Q_set)
            
            avg       - list of bool; indicating whether signals are averaged
                                      (see "EMPRISE.average_signals")
            corr      - string; method for serial correlations ('iid' or 'ar1' for
                                "i.i.d. errors" or "AR(1) model")
            order     - int; order of the HRF model, must be 1, 2 or 3
                             (see "PySPM.spm_get_bf")
            mu_grid   - vector; candidate values for mu
            sig_grid  - vector; candidate values for sigma (in logarithmic space)
            fwhm_grid - vector; candidate values for fwhm (in linear space)
            Q_set     - list of matrices; covariance components for AR estimation
                                          (default: see below, part 2)
            
            mu_est    - 1 x v vector; estimated numerosities in linear space
            fwhm_est  - 1 x v vector; estimated tuning widths in linear space
            beta_est  - 1 x v vector; estimated scaling factors for each voxel
            MLL_est   - 1 x v vector; maximum log-likelihood for model using
                                      the estimated tuning parameters
            MLL_null  - 1 x v vector; maximum log-likelihood for model using
                                      only confound and constant regressors
            MLL_const - 1 x v vector; maximum log-likelihood for model using
                                      only the constant regressor
            corr_est  - dict; specifying estimated covariance structure
            o Q       - 1 x q list of matrices; additive covariance components
            o h       - r x q matrix; multiplicative variance factors
            o V       - n x n x r array; estimated covariance matrix
        
        Note: Only one of the variables "sig_grid" and "fwhm_grid" should be
        specified. If both are (not) specified, "sig_grid" is prioritized.
        """
        
        # part 1: prepare data and design matrix
        #---------------------------------------------------------------------#
        n = self.Y.shape[0]; n_orig = n     # number of scans
        v = self.Y.shape[1]                 # number of voxels
        r = self.Y.shape[2]                 # number of runs
        c = self.X_c.shape[1]               # number of variables
        o = order                           # number of HRF regressors
        
        # if averaging across runs or epochs, regress out confounds first
        if avg[0] or avg[1]:
            Y = np.zeros(self.Y.shape)
            for j in range(r):
                glm      = PySPM.GLM(self.Y[:,:,j], np.c_[self.X_c[:,:,j], np.ones((n,1))])
                B_est    = glm.OLS()
                # subtract confounds from signal, then re-add constant regressor
                Y[:,:,j] = glm.Y - glm.X @ B_est + glm.X[:,[-1]] @ B_est[[-1],:]
        
        # otherwise, use signals without further manipulation
        else:
            Y = self.Y
        
        # then average signals across runs and/or epochs, if applicable
        import EMPRISE
        Y, t = EMPRISE.average_signals(Y, t=None, avg=avg)
        if avg[1]: n = Y.shape[0]           # update number of scans
        
        # since design is identical across runs, always use first run
        ons = self.ons; dur = self.dur; stim = self.stim;
        ons = ons[0];   dur = dur[0];   stim = stim[0];
        
        # if averaging across epochs, correct onsets to first epoch
        if avg[1]:
            ons, dur, stim = EMPRISE.correct_onsets(ons, dur, stim)
        
        # if averaging across runs or epochs, exclude confounds
        if avg[0] or avg[1]:
            p = o + 1           # number of regression coefficients
            X = np.c_[np.zeros((n,o)), np.ones((n,1))]
            if not avg[0]:      # run-independent design matrices
                X = np.repeat(np.expand_dims(X, 2), r, axis=2)
        
        # otherwise, add confounds to design matrix
        else:
            p = o + c + 1       # number of regression coefficients
            X = np.zeros((n,p,r))
            for j in range(r):  # run-wise design matrices
                X[:,:,j] = np.c_[np.zeros((n,o)), self.X_c[:,:,j], np.ones((n,1))]
        
        # specify further parameters
        mtr = EMPRISE.mtr       # microtime resolution (= bins per TR)
        mto = EMPRISE.mto       # microtime onset (= reference slice)
        del EMPRISE
        
        # part 2: prepare correlation matrix
        #---------------------------------------------------------------------#
        if Q_set is None:
            a     = 0.4         # AR parameter
            Q_set = [np.eye(n), # covariance components
                     sp.linalg.toeplitz(np.power(a, np.arange(0,n))) - np.eye(n)]
        if Q_set is not None:
            q     = len(Q_set)
        
        # prepare condition regressors
        if corr == 'ar1':
            # create names, onsets, durations
            names     = ['1', '2', '3', '4', '5', '20']
            onsets    = []
            durations = []
            # collect onsets/durations from first run
            for name in names:
                onsets.append([o for (o,s) in zip(ons,stim) if s == int(name)])
                durations.append([d for (d,s) in zip(dur,stim) if s == int(name)])
            # call PySPM to create design matrix
            X_d, L_d = PySPM.get_des_mat(names, onsets, durations, \
                                         settings={'n': n, 'TR': self.TR, 'mtr': mtr, 'mto': mto, 'HRF': 'spm_hrf'})
            # add confound variables to design
            if avg[0] or avg[1]:
                X0 = np.c_[X_d, np.ones((n,1))]
                if not avg[0]:
                    X0 = np.repeat(np.expand_dims(X0, 2), r, axis=2)
            else:
                X0 = np.zeros((n,len(names)+c+1,r))
                for j in range(r):
                    X0[:,:,j] = np.c_[X_d, self.X_c[:,:,j], np.ones((n,1))]
            # announce ReML estimation
            print('\n-> Restricted maximum likelihood estimation ({} rows, {} columns):'. \
                  format(n, v))
        
        # prepare correlation matrices
        if corr == 'iid':
            # invoke identity matrices, if i.i.d. errors
            if avg[0]:
                h   = np.array([1]+[0 for x in range(q-1)])
                V   = np.eye(n)
                P   = V
                ldV = 0
            else:
                h   = np.tile(np.array([1]+[0 for x in range(q-1)]), (r,1))
                V   = np.repeat(np.expand_dims(np.eye(n), 2), r, axis=2)
                P   = V
                ldV = np.zeros(r)
            # prepare estimated correlation dictionary
            corr_est = {'Q': Q_set, 'h': h, 'V': V}
        elif corr == 'ar1':
            # perform ReML estimation, if AR(1) process
            if avg[0]:
                V, Eh, Ph, F, Acc, Com = PySPM.spm_reml(Y, X0, Q_set)
                V   = (n/np.trace(V)) * V
                P   = np.linalg.inv(V)
                ldV = np.linalg.slogdet(V)[1]
            else:
                Eh  = np.zeros((q,r))
                V   = np.zeros((n,n,r))
                P   = np.zeros((n,n,r))
                ldV = np.zeros(r)
                for j in range(r):
                    print('   Run {}:'.format(j+1))
                    V[:,:,j], Eh[:,[j]], Ph, F, Acc, Com = PySPM.spm_reml(Y[:,:,j], X0[:,:,j], Q_set)
                    V[:,:,j] = (n/np.trace(V[:,:,j])) * V[:,:,j]
                    P[:,:,j] = np.linalg.inv(V[:,:,j])
                    ldV[j]   = np.linalg.slogdet(V[:,:,j])[1]
            # prepare estimated correlation dictionary
            corr_est = {'Q': Q_set, 'h': Eh.T, 'V': V}
            del Eh, Ph, F, Acc, Com
        else:
            err_msg = 'Unknown correlation method: "{}". Method must be "iid" or "ar1".'
            raise ValueError(err_msg.format(corr))
        
        # part 3: prepare grid search
        #---------------------------------------------------------------------#
        if mu_grid is None:                 # range: mu = 0.8,...,5.2 | 20
            mu_grid = np.concatenate((np.arange(0.80, 5.25, 0.05), np.array([20])))
        if sig_grid is None:
            if fwhm_grid is None:           # range: sig_log = 0.05,...,3.00
                sig_grid  = np.arange(0.05, 3.05, 0.05)
        # Explanation: If "sig_grid" and "fwhm_grid" are not specified,
        # then "sig_grid is specified (see comment in help text).
        elif sig_grid is not None:
            if fwhm_grid is not None:
                fwhm_grid = None
        # Explanation: If "sig_grid" and "fwhm_grid" are both specified,
        # then "fwhm_grid is disspecified (see comment in help text).
        
        # specify parameter grid
        mu   = mu_grid
        sig  = sig_grid
        fwhm = fwhm_grid
        
        # initialize parameters
        mu_est   = np.zeros(v)
        fwhm_est = np.zeros(v)
        beta_est = np.zeros(v)
        
        # prepare maximum likelihood
        MLL_est = -np.inf*np.ones(v)
        
        # part 4: perform grid search
        #---------------------------------------------------------------------#
        print('\n-> Maximum likelihood estimation ({} runs, {} voxels, {} scans, '. \
              format(r, v, n_orig), end='')
        print('\n   {}averaging across runs, {}averaging across epochs):'. \
              format(['no ', ''][int(avg[0])], ['no ', ''][int(avg[1])]))
        
        # for all values of mu
        for k1, m in enumerate(mu):
            
            # define current grid
            if sig is None:
                mus     = m*np.ones(fwhm.size)
                mu_log  = math.log(m)*np.ones(fwhm.size)
                fwhms   = fwhm
                sig_log = lin2log(mus, fwhms)[1]
            else:
                mus     = m*np.ones(sig.size)
                mu_log  = math.log(m)*np.ones(sig.size)
                sig_log = sig
                fwhms   = log2lin(mu_log, sig_log)[1]
            MLL  = np.zeros((fwhms.size,v))
            beta = np.zeros((fwhms.size,v))
            
            # display message
            print('   - grid chunk {} out of {}:'.format(k1+1, mu.size), end=' ')
            print('mu = {:.2f}, fwhm = {:.2f}, ..., {:.2f}'.format(m, fwhms[0], fwhms[-1]))
            
            # predict time courses
            Z, t = neuronal_signals(ons, dur, stim, self.TR, mtr, mu_log, sig_log)
            S, t = hemodynamic_signals(Z, t, n, mtr, mto, p=None, order=o)
            del Z, t
            
            # for all values of fwhm
            for k2, f in enumerate(fwhms):
                
                # generate design & estimate GLM
                if avg[0]:
                    X[:,0:o]   = S[:,k2,:]
                    glm        = PySPM.GLM(Y, X, P=P, ldV=ldV)
                    MLL[k2,:]  = glm.MLL()
                    beta[k2,:] = glm.WLS()[0,:]
                else:
                    for j in range(r):
                        X[:,0:o,j] = S[:,k2,:]
                        glm        = PySPM.GLM(Y[:,:,j], X[:,:,j], P=P[:,:,j], ldV=ldV[j])
                        MLL[k2,:]  = MLL[k2,:]  + glm.MLL()
                        beta[k2,:] = beta[k2,:] + glm.WLS()[0,:]/r
            
            # find maximum likelihood estimates
            k_max   = np.argmax(MLL, axis=0)
            MLL_max = MLL[k_max, range(v)]
            
            # update, if MLL larger than for previous MLEs
            mu_est[MLL_max>MLL_est]   = m
            fwhm_est[MLL_max>MLL_est] = fwhms[k_max][MLL_max>MLL_est]
            beta_est[MLL_max>MLL_est] = beta[k_max[MLL_max>MLL_est],MLL_max>MLL_est]
            MLL_est[MLL_max>MLL_est]  = MLL_max[MLL_max>MLL_est]
        
        # part 5: estimate reduced models
        #---------------------------------------------------------------------#
        print('\n-> Maximum likelihood estimation of reduced models', end='')
        print('\n   ({}averaging across runs, {}averaging across epochs):'. \
              format(['no ', ''][int(avg[0])], ['no ', ''][int(avg[1])]))
        
        # estimate model using only confound and constant regressors
        print('   - no-numerosity model ... ', end='')
        if avg[0]:
            MLL_null = PySPM.GLM(Y, X[:,o:], V).MLL()
        else:
            MLL_null = np.zeros(v)
            for j in range(r):
                MLL_null = MLL_null + PySPM.GLM(Y[:,:,j], X[:,o:,j], V[:,:,j]).MLL()
        print('successful!')
        
        # estimate model using only the constant regressor
        print('   - baseline-only model ... ', end='')
        if avg[0]:
            MLL_const = PySPM.GLM(Y, X[:,-1:], V).MLL()
        else:
            MLL_const = np.zeros(v)
            for j in range(r):
                MLL_const = MLL_const + PySPM.GLM(Y[:,:,j], X[:,-1:,j], V[:,:,j]).MLL()
        print('successful!')
        
        # return estimated parameters
        return mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est

    # function: maximum likelihood estimation (refined grid search)
    #-------------------------------------------------------------------------#
    def estimate_MLE_rgs(self, avg=[False, False], corr='iid', order=1, mu_grid=None, fwhm_grid=None, Q_set=None):
        """
        Maximum Likelihood Estimation using Refined Grid Search
        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =
            ds.estimate_MLE(avg, corr, order, mu_grid, fwhm_grid, Q_set)
            
            avg       - list of bool; indicating whether signals are averaged
                                      (see "EMPRISE.average_signals")
            corr      - string; method for serial correlations ('iid' or 'ar1' for
                                "i.i.d. errors" or "AR(1) model")
            order     - int; order of the HRF model, must be 1, 2 or 3
                             (see "PySPM.spm_get_bf")
            mu_grid   - list of floats; initial values [mu_0, dmu_0]
            fwhm_grid - list of floats; initial values [fwhm_0, dfwhm_0]
            Q_set     - list of matrices; covariance components for AR estimation
                                          (default: see below, part 2)
            
            mu_est    - 1 x v vector; estimated numerosities in linear space
            fwhm_est  - 1 x v vector; estimated tuning widths in linear space
            beta_est  - 1 x v vector; estimated scaling factors for each voxel
            MLL_est   - 1 x v vector; maximum log-likelihood for model using
                                      the estimated tuning parameters
            MLL_null  - 1 x v vector; maximum log-likelihood for model using
                                      only confound and constant regressors
            MLL_const - 1 x v vector; maximum log-likelihood for model using
                                      only the constant regressor
            corr_est  - dict; specifying estimated covariance structure
            o Q       - 1 x q list of matrices; additive covariance components
            o h       - r x q matrix; multiplicative variance factors
            o V       - n x n x r array; estimated covariance matrix
            
        Note: This function is considered obsolete and exists merely for legacy
        reasons. Use "estimate_MLE" to estimate numerosity tuning parameters.
        """
        
        # part 1: prepare data and design matrix
        #---------------------------------------------------------------------#
        n = self.Y.shape[0]; n_orig = n     # number of scans
        v = self.Y.shape[1]                 # number of voxels
        r = self.Y.shape[2]                 # number of runs
        c = self.X_c.shape[1]               # number of variables
        o = order                           # number of HRF regressors
        
        # if averaging across runs or epochs, regress out confounds first
        if avg[0] or avg[1]:
            Y = np.zeros(self.Y.shape)
            for j in range(r):
                glm      = PySPM.GLM(self.Y[:,:,j], np.c_[self.X_c[:,:,j], np.ones((n,1))])
                B_est    = glm.OLS()
                # subtract confounds from signal, then re-add constant regressor
                Y[:,:,j] = glm.Y - glm.X @ B_est + glm.X[:,[-1]] @ B_est[[-1],:]
        
        # otherwise, use signals without further manipulation
        else:
            Y = self.Y
        
        # then average signals across runs and/or epochs, if applicable
        import EMPRISE
        Y, t = EMPRISE.average_signals(Y, t=None, avg=avg)
        if avg[1]: n = Y.shape[0]           # update number of scans
        
        # since design is identical across runs, always use first run
        ons = self.ons; dur = self.dur; stim = self.stim;
        ons = ons[0];   dur = dur[0];   stim = stim[0];
        
        # if averaging across epochs, correct onsets to first epoch
        if avg[1]:
            ons, dur, stim = EMPRISE.correct_onsets(ons, dur, stim)
        
        # if averaging across runs or epochs, exclude confounds
        if avg[0] or avg[1]:
            p = o + 1           # number of regression coefficients
            X = np.c_[np.zeros((n,o)), np.ones((n,1))]
            if not avg[0]:      # run-independent design matrices
                X = np.repeat(np.expand_dims(X, 2), r, axis=2)
        
        # otherwise, add confounds to design matrix
        else:
            p = o + c + 1       # number of regression coefficients
            X = np.zeros((n,p,r))
            for j in range(r):  # run-wise design matrices
                X[:,:,j] = np.c_[np.zeros((n,o)), self.X_c[:,:,j], np.ones((n,1))]
        
        # specify further parameters
        mtr = EMPRISE.mtr       # microtime resolution (= bins per TR)
        mto = EMPRISE.mto       # microtime onset (= reference slice)
        del EMPRISE
        
        # part 2: prepare correlation matrix
        #---------------------------------------------------------------------#
        m, f, b, MLL1, MLL0, MLL00, corr_est = \
            self.estimate_MLE(avg, corr, order, Q_set=Q_set, \
                              mu_grid=np.array([1,5]), \
                              fwhm_grid=np.array([1,2]))
        # Explanation: To avoid code repetitions, "estimate_MLE" is called.
        del m, f, b, MLL1, MLL0, MLL00
        
        # extract estimated correlation structure
        V = corr_est['V']
        if avg[0]:
            P   = np.linalg.inv(V)
            ldV = np.linalg.slogdet(V)[1]
        else:
            P   = np.zeros((n,n,r))
            ldV = np.zeros(r)
            for j in range(r):
                P[:,:,j] = np.linalg.inv(V[:,:,j])
                ldV[j]   = np.linalg.slogdet(V[:,:,j])[1]
        
        # part 3: prepare grid search
        #---------------------------------------------------------------------#
        if mu_grid is None: # total range: 1 < mu < 5
            mu_grid   = [ 3.0, 1.0]
        if fwhm_grid is None:#total range: 0.1 < fwhm < 20.1
            fwhm_grid = [10.1, 5.0]
        
        # specify parameter grid
        m0 = mu_grid[0];   dm0 = mu_grid[1];
        f0 = fwhm_grid[0]; df0 = fwhm_grid[1];
        
        # initialize parameters
        mu_est   = m0*np.ones(v)
        fwhm_est = f0*np.ones(v)
        beta_est = np.zeros(v)
        
        # prepare maximum likelihood
        beta    = np.zeros(9)
        MLL     = np.zeros(9)
        MLL_est = np.zeros(v)
        
        # part 4: perform grid search
        #---------------------------------------------------------------------#
        print('\n-> Maximum likelihood estimation ({} runs, {} voxels, {} scans, '. \
              format(r, v, n_orig), end='')
        print('\n   {}averaging across runs, {}averaging across epochs):'. \
              format(['no ', ''][int(avg[0])], ['no ', ''][int(avg[1])]))
        
        # for all voxels
        for i in range(v):
            
            # display message
            print('   - voxel {} out of {}:'.format(i+1, v))
            
            # intialize step size
            dm = dm0; df = df0; s = 0;
            
            # as long as mu doesn't change by less than tolerance
            while dm > 0.01:
                
                # display message
                print('     - iteration {}: '.format(s+1), end='')
                
                # define current grid
                mus   = np.repeat(np.array([mu_est[i]-dm, mu_est[i], mu_est[i]+dm]), 3)
                fwhms = np.tile(np.array([fwhm_est[i]-df, fwhm_est[i], fwhm_est[i]+df]), 3)
                mu_log, sig_log = lin2log(mus, fwhms)
                
                # predict time courses
                Z, t = neuronal_signals(ons, dur, stim, self.TR, mtr, mu_log, sig_log)
                S, t = hemodynamic_signals(Z, t, n, mtr, mto, p=None, order=o)
                
                # cycle through grid
                for k in range(9):
                    
                    # generate design & estimate GLM
                    if avg[0]:
                        X[:,0:o] = S[:,k,:]
                        glm      = PySPM.GLM(Y[:,[i]], X, P=P, ldV=ldV)
                        MLL[k]   = glm.MLL()
                        beta[k]  = glm.OLS()[0,0]
                    else:
                        MLL[k]  = 0
                        beta[k] = 0
                        for j in range(r):
                            X[:,0:o,j] = S[:,k,:] # S[:,k,j]
                            glm        = PySPM.GLM(Y[:,[i],j], X[:,:,j], P=P[:,:,j], ldV=ldV[j])
                            MLL[k]     = MLL[k]  + glm.MLL()
                            beta[k]    = beta[k] + glm.WLS()[0,0]/r
                
                # find maximum likelihood estimates
                k_max       = np.argmax(MLL)
                mu_est[i]   = mus[k_max]
                fwhm_est[i] = fwhms[k_max]
                dm          = dm/2
                df          = df/2
                s           = s+1
                
                # display message
                print('mu = {:.2f}, fwhm = {:.2f}'.format(mu_est[i], fwhm_est[i]))
                
            # get maximum log-likelihood
            beta_est[i] = beta[k_max]
            MLL_est[i]  = MLL[k_max]
        
        # part 5: estimate reduced models
        #---------------------------------------------------------------------#
        m, f, b, MLL1, MLL_null, MLL_const, ce = \
            self.estimate_MLE(avg, corr, order, Q_set=Q_set, \
                              mu_grid=np.array([1,5]), \
                              fwhm_grid=np.array([1,2]))
        # Explanation: To avoid code repetitions, "estimate_MLE" is called.
        del m, f, b, MLL1, ce

        # return estimated parameters
        return mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est
    
    # function: number of free parameters
    #-------------------------------------------------------------------------#
    def free_parameters(self, avg=[False, False], corr='iid', order=1):
        """
        Number of Free Parameters for Maximum Likelihood Estimation
        k_est, k_null, k_const = ds.free_parameters(avg, corr, order)
            
            avg     - list of bool; see "estimate_MLE"
            corr    - string; see "estimate_MLE"
            order   - int; see "estimate_MLE"
            
            k_est   - int; number of free parameters for
                           model estimating tuning parameters
            k_null  - int; number of free parameters for
                           model using only confound and constant regressors
            k_const - int; number of free parameters for
                           model using only the constant regressor
        """
        
        # get data dimensions
        #---------------------------------------------------------------------#
        r = self.Y.shape[2]                 # number of runs
        c = self.X_c.shape[1]               # number of variables
        o = order                           # number of HRF regressors
        
        # calculate parameters
        #---------------------------------------------------------------------#
        r = [r,1][int(avg[0])]              # number of runs after averaging
        c = [c,0][int(np.max(avg))]         # number of confounds after averaging
        p = o + c + 1                       # number of regression coefficients
        k_est   = 2 + r*(p+1)               # 2+ -> tuning parameters
        k_null  =     r*(p-o+1)             # +1 -> noise variance
        k_const =     r*(p-o-c+1)           # r* -> per each run
                                            # -o -> w/o tuning regressors
        # return free parameters            # -c -> w/o confound regressors
        #---------------------------------------------------------------------#
        return k_est, k_null, k_const
    
    # function: visualize signals (single axis)
    #-------------------------------------------------------------------------#
    def plot_signals_axis(self, ax):
        """
        Plot Signals along with EMPRISE Task (single axis)
        ds.plot_task_signals(ax)
        
            ax - axis; into which the signals are plotted
            
        ds.plot_signals_axis(ax) plots fMRI signals Y at acquisition times t
        from the data set ds into into the axis ax and adds EMPRISE task
        markers according to the onsets, durations and stimuli in ds at the
        top of the plot.
        """
        
        # get dimensions
        n = self.Y.shape[0]         # number of observations
        r = self.Y.shape[1]         # number of signals
        b = len(self.ons[0])        # number of blocks
        
        # plot time series
        t = np.arange(0, n*self.TR, self.TR)
        for j in range(r):
            ax.plot(t, self.Y[:,j])
            
        # set axis limits
        Y_min = np.min(self.Y)
        Y_max = np.max(self.Y)
        Y_rng = Y_max-Y_min
        ax.axis([0, np.max(t), Y_min-(1/20)*Y_rng, Y_max+(3/20)*Y_rng])
        
        # plot task blocks
        ons = self.ons[0]; dur = self.dur[0]; stim = self.stim[0];
        for i in range(b):
            ax.plot(np.array([ons[i],ons[i]]), \
                    np.array([Y_max+(1/20)*Y_rng, Y_max+(3/20)*Y_rng]), '-k')
            ax.text(ons[i]+(1/2)*dur[i], Y_max+(2/20)*Y_rng, str(stim[i]), \
                    fontsize=16, horizontalalignment='center', verticalalignment='center')
        ax.plot(np.array([ons[-1]+dur[-1],ons[-1]+dur[-1]]), \
                np.array([Y_max+(1/20)*Y_rng, Y_max+(3/20)*Y_rng]), '-k')
        ax.plot(np.array([ons[0],ons[-1]+dur[-1]]), \
                np.array([Y_max+(1/20)*Y_rng, Y_max+(1/20)*Y_rng]), '-k')
        
        # set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.tick_params(axis='both', which='minor', labelsize=16)
    
    # function: visualize signals (full figure)
    #-------------------------------------------------------------------------#
    def plot_signals_figure(self, fig, mu=None, fwhm=None, avg=[False, False], xlabel='time [s]', ylabel='signal [a.u.]', title='EMPRISE BOLD signals'):
        """
        Plot Signals along with EMPRISE Task (full figure)
        ds.plot_task_signals(fig, mu, fwhm, avg, xlabel, ylabel, title)
        
            fig    - figure; into which the signals are plotted
            mu     - 1 x v vector; voxel-wise preferred numerosities (optional)
            fwhm   - 1 x v vector; voxel-wise tuning widths (optional)
            avg    - list of bool; indicating whether signals are averaged (see below)
            xlabel - string; label of the x-axis of the plot
            ylabel - string; label of the y-axis of the plot
            title  - string; title of the plot
            
        ds.plot_task_signals(fig, mu, fwhm, avg, xlabel, ylabel, title) plots
        fMRI signals Y at acquisition times t from the data set ds into the
        multiple axes of the figure fig and adds EMPRISE task markers according
        to onsets ons, durations dur and stimuli stim at the top of the plot.
        
        The input variable "avg" controls averaging. If the first entry of avg is
        true, then signals are averaged over runs. If the second entry of avg is
        true, then signals are averaged over epochs within runs. If both are
        true, then signals are first averaged over runs and then epochs. By
        default, both entries are false and no averaging takes place.
        
        The input variables "xlabel", "ylabel" and "title" control x-axis label,
        y-axis label and axis title. If left empty, no labels are displayed. If
        not specified, default labels are displayed (see above).
        """
        
        # average signals, if averaging over runs or epochs
        import EMPRISE
        Y, t = EMPRISE.average_signals(self.Y, t=None, avg=avg)
        v    = Y.shape[1]
        
        # correct onsets, if averaging over epochs
        ons = self.ons[0]; dur = self.dur[0]; stim = self.stim[0];
        if avg[1]:
            ons, dur, stim = EMPRISE.correct_onsets(ons, dur, stim)
        
        # plot signals (one subplot = one voxel)
        axs = fig.subplots(v,1)
        for j, ax in enumerate(axs):
            
            # select signals from this voxel
            if len(Y.shape) < 3:
                Y_ax = Y[:,[j]]
            else:
                Y_ax = Y[:,j,:]
            
            # plot into the present axis
            ds = DataSet(Y_ax, [ons], [dur], [stim], self.TR, self.X_c)
            ds.plot_signals_axis(ax)
            del ds
            
            # add labels, title etc.
            if j == len(axs)-1:
                ax.set_xlabel(xlabel, fontsize=16)
            if j >= 0:
                ax.set_ylabel(ylabel, fontsize=16)
            if j == 0:
                ax.set_title(title, fontsize=24)
            if mu is not None:
                ax.text(np.mean(t), np.min(Y_ax)-(1/2)*(1/20)*(np.max(Y_ax)-np.min(Y_ax)), \
                        'mu = {}, fwhm = {}'.format(mu[j], fwhm[j]), \
                        fontsize=16, horizontalalignment='center', verticalalignment='center')
        
        # show figure
        fig.show()

# test area / debugging section
#-----------------------------------------------------------------------------#
if __name__ == '__main__':
    
    # import packages
    import EMPRISE
    import matplotlib.pyplot as plt
    # enter "%matplotlib qt" in Spyder before
    
    # specify what to test
    what_to_test = 'corrtest'
    
    # test "lin2log" etc.
    if what_to_test == 'lin2log':
        a,b = lin2log(3,1.5)
        print(f_log(np.arange(1,6,1), a, b))
        print(f_lin(np.arange(1,6,1), 3, 1.5))
        print(lin2log(3,1.5))
        print(log2lin(1,0.1))
        print(log2lin(a,b))
        print(fwhm2sig(1.5))
    
    # test "MLL2Rsq"
    if what_to_test == 'MLL2Rsq':
        MLL1 = np.array([0, -50, -75, -87.5, -93.75, -100])
        MLL0 =-100
        n    = 50
        print(MLL2Rsq(MLL1, MLL0, n))
    
    # test "yp2Rsq"
    if what_to_test == 'yp2Rsq':
        yp = np.random.uniform(0, 10, size=(100,10))
        y  = np.zeros(yp.shape)
        for j in range(yp.shape[1]):
            y[:,j] = yp[:,j] + np.random.normal(0, j*0.3, size=100)
        print(yp2Rsq(y, yp))
    
    # test "corrtest"
    if what_to_test == 'corrtest':
        
        # calculate R-squared
        dr = 0.01
        n1 = 36
        n2 = 145
        r  = np.arange(dr, 1, dr)
        h, p1, stats = corrtest(r, n1)
        h, p2, stats = corrtest(r, n2)
        h, p1t,stats = corrtest(0.30, n1)
        h, p2t,stats = corrtest(0.25, n2)
        del h, stats
        
        # plot p-values
        fig = plt.figure(figsize=(16,9))
        ax  = fig.add_subplot(111)
        ax.plot(r, p1, '-r', label='n='+str(n1))
        ax.plot(r, p2, '-b', label='n='+str(n2))
        ax.plot(np.array([0,1]), np.array([0.05,0.05]), ':k', label='p=0.05')
        ax.plot(np.array([0,1]), np.array([0.01,0.01]), '--k', label='p=0.01')
        ax.plot(np.array([0,1]), np.array([0.001,0.001]), '-k', label='p=0.001')
        ax.axis([0, 1, 1e-17, 0.2])
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=16)
        ax.legend(loc='right', fontsize=16)
        ax.set_xlabel('correlation coefficient', fontsize=20)
        ax.set_ylabel('p-value', fontsize=20)
        ax.set_title('Statistical Significance of Pearson Correlation', fontsize=24)
        ax.text(dr, 1e-15, ' p(0.3, n=36) = {:1.2e}'.format(p1t), fontsize=16, \
                horizontalalignment='left', verticalalignment='center')
        ax.text(dr, 1e-16, ' p(0.25, n=145) = {:1.2e}'.format(p2t), fontsize=16, \
                horizontalalignment='left', verticalalignment='center')        
    
    # test "neuronal_signals"
    if what_to_test == 'neuronal_signals':
        mu_log, sig_log = lin2log(3,1.5)
        ons, dur, stim  = EMPRISE.Session('EDY7', 'visual').get_onsets()
        ons, dur, stim  = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
        Z, t = neuronal_signals(ons[0], dur[0], stim[0], EMPRISE.TR, EMPRISE.mtr, np.array([mu_log]), np.array([sig_log]))
        plt.plot(t, Z, '-b')
        plt.show()
        
    # test "hemodynamic_signals"
    if what_to_test == 'hemodynamic_signals':
        mu_log, sig_log = lin2log(3,1.5)
        ons, dur, stim  = EMPRISE.Session('EDY7', 'visual').get_onsets()
        ons, dur, stim  = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
        Z, t = neuronal_signals(ons[0], dur[0], stim[0], EMPRISE.TR, EMPRISE.mtr, np.array([mu_log]), np.array([sig_log]))
        S, t = hemodynamic_signals(Z, t, EMPRISE.n, EMPRISE.mtr, EMPRISE.mto, order=3)
        plt.plot(t, S[:,0,0], '-b')
        plt.plot(t, S[:,0,1], '-r')
        plt.plot(t, S[:,0,2], '-g')
        plt.show()
    
    # test "simulate"
    if what_to_test == 'simulate':
        
        # specify subject and session
        sub = 'EDY7'
        ses = 'visual'
        
        # get onsets and durations
        sess           = EMPRISE.Session(sub, ses)
        ons, dur, stim = sess.get_onsets()
        TR             = EMPRISE.TR
        
        # get confound variables
        labels = EMPRISE.covs
        X_c    = sess.get_confounds(labels)
        X_c    = EMPRISE.standardize_confounds(X_c)
        
        # specify tuning parameters
        mu   = np.array([1,2,3,4,5])
        fwhm = np.array([1,1,1,1,1])
        
        # specify scaling parameters
        mu_b = 2
        mu_c = 2
        s2_k = 1
        s2_j = 1
        s2_i = 1
        tau  = 0.1
        hrf  = None
        
        # simulate data
        ds = DataSet(0, ons, dur, stim, TR, X_c)
        Y, S, X, B = ds.simulate(mu, fwhm, mu_b, mu_c, s2_k, s2_j, s2_i, tau, hrf)
        
        # plot simulation (1)
        plt.rcParams.update({'font.size': 24})
        fig = plt.figure(figsize=(32,18))
        axs = fig.subplots(1,4)
        fig.suptitle('EMPRISE simulation')
        axs[0].imshow(Y[:,:,0], aspect='auto')
        axs[1].imshow(S[:,:,0], aspect='auto')
        axs[2].imshow(X[:,:,0], aspect='auto')
        axs[3].imshow(B[:,:,0], aspect='auto')
        fig.show()
        
        # plot simulation (2)
        fig = plt.figure(figsize=(32,18))
        axs = fig.subplots(mu.size,1)
        fig.suptitle('EMPRISE simulation')
        for k in range(mu.size):
            for j in range(Y.shape[2]):
                axs[k].plot(Y[:,k,j])
        fig.show()
    
    # test "estimate_MLE"
    if what_to_test == 'estimate_MLE':
        
        # generate signals
        ons, dur, stim = EMPRISE.Session('001', 'visual').get_onsets()
        ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
        X_c  = np.zeros((EMPRISE.n,0,len(ons)))
        mu   = np.arange(1.0, 6.0, 0.5)
        fwhm = np.repeat(np.arange(3, 18, 3), 2)
        ds   = DataSet(0, ons, dur, stim, EMPRISE.TR, X_c)
        Y, S, X, B = ds.simulate(mu, fwhm, 3, 3, 1, 0.1, 20, 0.5, None)
        
        # estimate parameters
        import time
        avgs  = [True, False]
        noise = 'iid'
        hrfs  = 1
        start_time = time.time()
        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est = \
            ds.estimate_MLE(avg=avgs, corr=noise, order=hrfs)
        k_est, k_null, k_const = ds.free_parameters(avgs, noise, hrfs)
        end_time   = time.time()
        print()
        print('-> time passed       : {:.2f} seconds'.format(end_time-start_time))
        print('-> r(mu,   mu_est)   = {:.4f}'.format(np.corrcoef(mu, mu_est)[0,1]))
        print('-> r(fwhm, fwhm_est) = {:.4f}'.format(np.corrcoef(fwhm, fwhm_est)[0,1]))
        print()
        print(mu_est)
        print(fwhm_est)
        print(beta_est)
        print(MLL_est)
        print(MLL_null)
        print(MLL_const)
        print(corr_est['h'])
        print([k_est, k_null, k_const])
        
    # test "estimate_MLE_rgs"
    if what_to_test == 'estimate_MLE_rgs':
        
        # generate signals
        ons, dur, stim = EMPRISE.Session('001', 'visual').get_onsets()
        ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
        X_c  = np.zeros((EMPRISE.n,0,len(ons)))
        mu   = np.arange(1.0, 6.0, 0.5)
        fwhm = np.repeat(np.arange(3, 18, 3), 2)
        ds   = DataSet(0, ons, dur, stim, EMPRISE.TR, X_c)
        Y, S, X, B = ds.simulate(mu, fwhm, 3, 3, 1, 0.1, 20, 0.5, None)
        
        # estimate parameters
        import time
        avgs  = [False, False]
        noise = 'ar1'
        hrfs  = 3
        start_time = time.time()
        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est = \
            ds.estimate_MLE_rgs(avg=avgs, corr=noise, order=hrfs)
        end_time   = time.time()
        print()
        print('-> time passed       : {:.2f} seconds'.format(end_time-start_time))
        print('-> r(mu,   mu_est)   = {:.4f}'.format(np.corrcoef(mu, mu_est)[0,1]))
        print('-> r(fwhm, fwhm_est) = {:.4f}'.format(np.corrcoef(fwhm, fwhm_est)[0,1]))
    
    # test "plot_signals_axis"
    if what_to_test == 'plot_signals_axis':
        
        # generate signals
        ons, dur, stim = EMPRISE.Session('EDY7', 'visual').get_onsets()
        ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
        X_c  = np.zeros((EMPRISE.n,0,len(ons)))
        mu   = np.array([1,3,5])
        fwhm = np.array([1,1,1])
        ds   = DataSet(0, ons, dur, stim, EMPRISE.TR, X_c)
        Y, S, X, B = ds.simulate(mu, fwhm, 10, 1, 1, 0.1, 0.25, 0.001)
        
        # visualize signals
        fig = plt.figure(figsize=(32,18))
        axs = fig.subplots(mu.size,1)
        for j, ax in enumerate(axs):
            title  = ''; xlabel = ''; ylabel = 'signal [a.u.]'
            if j == 0: title = 'EMPRISE BOLD signals'
            if j == len(axs)-1: xlabel = 'time [s]'
            Y_ax = np.squeeze(Y[:,j,:])
            ds   = DataSet(Y_ax, ons, dur, stim, EMPRISE.TR, X_c)
            ds.plot_signals_axis(ax)
            ax.set_xlabel(xlabel, fontsize=16)
            ax.set_ylabel(ylabel, fontsize=16)
            ax.set_title(title, fontsize=24)
        
        # show/save figure
        fig.show()
        fig.savefig('plot_signals_axis.png', dpi=150)
    
    # test "plot_signals_figure"
    if what_to_test == 'plot_signals_figure':
        
        # generate signals
        ons, dur, stim = EMPRISE.Session('EDY7', 'visual').get_onsets()
        ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
        X_c  = np.zeros((EMPRISE.n,0,len(ons)))
        mu   = np.array([1,3,5])
        fwhm = np.array([5,5,5])
        ds   = DataSet(0, ons, dur, stim, EMPRISE.TR, X_c)
        Y, S, X, B = ds.simulate(mu, fwhm, 10, 1, 1, 0.1, 0.25, 0.001)
        
        # visualize signals
        fig = plt.figure(figsize=(32,18))
        ds.plot_signals_figure(fig, mu, fwhm, avg=[False, False], xlabel='time [s]', ylabel='signal [a.u.]', title='EMPRISE BOLD signals')
        fig.savefig('plot_signals_figure.png', dpi=150)