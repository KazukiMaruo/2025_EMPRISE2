"""
PySPMs - an SPM clone in Python

Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
2023-06-15, 13:41: spm_hrf
2023-06-15, 15:37: get_des_mat
2023-07-13, 16:08: GLM, GLM.MLE, GLM.MLL
2023-08-14, 16:36: GLM, GLM.MLE, GLM.MLL
2023-08-17, 14:55: get_des_mat, documentation
2023-08-28, 15:25: spm_get_bf, spm_hrf
2023-08-31, 16:24: spm_reml
2023-09-07, 23:55: GLM.OLS, GLM.WLS
2023-09-08, 10:56: GLM.tcon, GLM.Fcon
2023-09-14, 14:10: GLM, GLM.MLL
2023-09-21, 10:28: GLM.regress
"""


# import packages
#-----------------------------------------------------------------------------#
import math
import numpy as np
import scipy as sp

# class: general linear model
#-----------------------------------------------------------------------------#
class GLM:
    """
    The GLM class allows to specify, estimate and assess univariate general
    linear models a.k.a. linear regression which is defined by an n x 1 data
    vector y, an n x p design matrix X and an n x n covariance matrix V.
    """
    
    # function: initialize GLM
    #-------------------------------------------------------------------------#
    def __init__(self, Y, X, V=None, P=None, ldV=None):
        """
        Initialize a General Linear Model
        glm = PySPM.GLM(Y, X, V)
        
            Y   - an n x v data matrix of measured signals
            X   - an n x p design matrix of predictor variables
            V   - an n x n covariance matrix specifying correlations (default: I_n)
            P   - an n x n precision matrix specifying correlations (default: inv(V))
            ldV - float; log-determinant of correlation matrix (default: log(det(V)))
        
            glm - a GLM object
            o Y - the n x v data matrix
            o X - the n x p design matrix
            o V - the n x n covariance matrix
            o P - the n x n precision matrix
            o n - the number of observations
            o v - the number of signals
            o p - the number of regressors
        """
        
        # store model specification
        self.Y = Y                          # data matrix
        self.X = X                          # design matrix
        
        # store covariance matrix
        if V is None:
            if P is None:
                self.V   = np.eye(Y.shape[0])#covariance matrix
                self.P   = self.V           # precision matrix
                self.iid = True             # errors are i.i.d.
            else:
                self.V   = V                # If P was supplied, leave V
                self.P   = P                # at None and use specified P.
                self.iid = False            # This avoid matrix inversion.
        else:
            self.V   = V                    # If V was supplied, use it
            self.P   = np.linalg.inv(V)     # and calculate P as inv(V).
            self.iid = np.all(V == np.eye(Y.shape[0]))
        
        # store log-determinant
        if ldV is None:
            if self.iid:                    # If errors are i.i.d.,
                self.ldV = 0                # log-determinant is zero.
            else:
                if self.V is None:          # Otherwise, calculate from V/P
                    self.ldV =-np.linalg.slogdet(self.P)[1]
                else:
                    self.ldV = np.linalg.slogdet(self.V)[1]
        else:                               # or use specified value,
            self.ldV = ldV                  # if ldV was supplied.
        
        # store model dimensions
        self.n = Y.shape[0]                 # number of observations
        self.v = Y.shape[1]                 # number of signals
        self.p = X.shape[1]                 # number of regressors
        
    # function: ordinary least squares
    #-------------------------------------------------------------------------#
    def OLS(self):
        """
        Ordinary Least Squares for General Linear Model
        B_est = glm.OLS()
        
            B_est - p x v matrix; estimated regression coefficients
        
        B_est = glm.OLS() computes ordinary least squares estimates for the
        regression coefficients beta, i.e. the values of those that minimize
        the residual sum of squares [1].
        
        [1] https://statproofbook.github.io/P/mlr-ols
        """
        
        # estimate model parameters
        B_cov = np.linalg.inv(self.X.T @ self.X)
        B_est = B_cov @ (self.X.T @ self.Y)
        
        # return parameter estimates
        return B_est
    
    # function: weighted least squares
    #-------------------------------------------------------------------------#
    def WLS(self):
        """
        Weighted Least Squares for General Linear Model
        B_est = glm.WLS()
        
            B_est - p x v matrix; estimated regression coefficients
        
        B_est = glm.WLS() computes weighted least squares estimates for the
        regression coefficients beta, i.e. the values of those that minimize
        the weighted residual sum of squares [1]
        
        [1] https://statproofbook.github.io/P/mlr-wls
        """
        
        # estimate model parameters (i.i.d.)
        if self.iid:
            B_cov  = np.linalg.inv(self.X.T @ self.X)
            B_est  = B_cov @ (self.X.T @ self.Y)
        
        # estimate model parameters (not i.i.d.)
        else:
            B_cov  = np.linalg.inv(self.X.T @ self.P @ self.X)
            B_est  = B_cov @ (self.X.T @ self.P @ self.Y)
        
        # return parameter estimates
        return B_est
    
    # function: maximum likelihood estimation
    #-------------------------------------------------------------------------#
    def MLE(self):
        """
        Maximum Likelihood Estimation for General Linear Model
        B_est, s2_est = glm.MLE()
        
            B_est  - p x v matrix; estimated regression coefficients
            s2_est - 1 x v vector; estimated residual variances
        
        B_est, s2_est = glm.MLE() computes maximum likelihood estimates for the
        regression coefficients beta and the error variance sigma^2, i.e. the
        values of those that maximize the log-likelihood function [1].
        
        [1] https://statproofbook.github.io/P/mlr-mle
        """
        
        # estimate model parameters (i.i.d.)
        if self.iid:
            B_cov  = np.linalg.inv(self.X.T @ self.X)
            B_est  = B_cov @ (self.X.T @ self.Y)
            E_est  = self.Y - self.X @ B_est
            s2_est = 1/self.n * np.sum(np.square(E_est), axis=0)
        
        # estimate model parameters (not i.i.d.)
        else:
            B_cov  = np.linalg.inv(self.X.T @ self.P @ self.X)
            B_est  = B_cov @ (self.X.T @ self.P @ self.Y)
            E_est  = self.Y - self.X @ B_est
            s2_est = np.zeros(self.v)
            for j in range(self.v):
                s2_est[j] = 1/self.n * (E_est[:,j].T @ self.P @ E_est[:,j])
        
        # return parameter estimates
        return B_est, s2_est
    
    # function: maximum log-likelihood
    #-------------------------------------------------------------------------#
    def MLL(self):
        """
        Maximum Log-Likelihood for General Linear Model
        LL_max = glm.MLL()
        
            LL_max - 1 x v vector; maximum log-likelihood values
            
        LL_max = glm.MLL() computes the maximum log-likelihood for GLM
        initialized via glm = GLM(Y, X, V), i.e. the values of the log-
        likelihood function at the maximum likelihood estimates [1].
        
        [1] https://statproofbook.github.io/P/mlr-mll
        """
        
        # obtain maximum likelihood estimates
        B_est, s2_est = self.MLE()
        
        # compute maximum log-likelihood
        LL_max = - self.n/2 \
                 - self.n/2 * np.log(s2_est) \
                 - self.n/2 * np.log(2*math.pi) \
                 -      1/2 * self.ldV
        
        # return maximum log-likelihood
        return LL_max
    
    # function: regress out variables
    #-------------------------------------------------------------------------#
    def regress(self):
        """
        Regress out Predictor Variables from Measured Signal
        E = glm.regress()
        
            E - n x v matrix; residuals after regression
        
        E_est = glm.regress() computes the OLS residual-forming matrix
            R = I_n - X (X^T X)^(-1) X^T
        and right-multiplies it with the matrix of measured signals [1]
            E = R Y
        to obtain the residuals after regressing out X from Y.
        
        [1] https://statproofbook.github.io/P/mlr-mat
        """
        
        # calculate residuals
        R = np.eye(self.n) - self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T
        E = R @ self.Y
        
        # return residuals
        return E
    
    # function: t-contrast inference
    #-------------------------------------------------------------------------#
    def tcon(self, c, alpha=0.05):
        """
        Contrast-Based Inference using t-Contrasts
        h, p, stats = glm.tcon(c, alpha)
        
            c       - p x 1 vector; contrast vector for t-contrast
            alpha   - float; significance level for the t-test
        
            h       - 1 x v vector; indicating rejectance of the null hypothesis
            p       - 1 x v vector; p-values computed under the null hypothesis
            stats   - dict; further information on statistical inference:
            o tstat - 1 x v vector; values of the t-statistic
            o df    - int; degrees of freedom of the t-statistic
        
        h, p, stats = glm.tcon(c, alpha) performs a t-test for the model glm,
        using contrast vector c and significance level alpha and returns a
        vector of logicals h indicating rejectance of the null hypothesis
            H0: c^T b = 0
        and the vector of p-values in favor of the alternative hypothesis
            H1: c^T b > 0
        as well as further information on the statistical test [1].
        
        [1] https://github.com/JoramSoch/MACS/blob/master/ME_GLM_con.m
        """
        
        # expand contrast, if necessary
        if len(c.shape) == 1:
            c = np.expand_dims(c,1)
        
        # estimate model parameters
        B_est, s2_est = self.MLE()
        s2_unb        =(self.n/(self.n-self.p)) * s2_est
        
        # compute beta covariance
        if self.iid:
            covB = np.linalg.inv(self.X.T @ self.X)
        else:
            covB = np.linalg.inv(self.X.T @ self.P @ self.X)
        
        # calculate t-statistics
        c_cov_c = c.T @ covB @ c
        con_est = np.squeeze(c.T @ B_est)
        den_est = np.zeros(self.v)
        for j in range(self.v):
            den_est[j] = np.sqrt(s2_unb[j] * c_cov_c)
        
        # calculate p-values
        stats = {'tstat': con_est/den_est, 'df': self.n-self.p}
        p     = 1 - sp.stats.t.cdf(stats['tstat'], stats['df'])
        h     = p < alpha
        
        # return test statistics
        return h, p, stats
    
    # function: F-contrast inference
    #-------------------------------------------------------------------------#
    def Fcon(self, C, alpha=0.05):
        """
        Contrast-Based Inference using F-Contrasts
        h, p, stats = glm.Fcon(C, alpha)
        
            C       - p x q matrix; contrast matrix for F-contrast
            alpha   - float; significance level for the F-test
        
            h       - 1 x v vector; indicating rejectance of the null hypothesis
            p       - 1 x v vector; p-values computed under the null hypothesis
            stats   - dict; further information on statistical inference:
            o Fstat - 1 x v vector; values of the F-statistic
            o df    - list of ints; degrees of freedom for the F-statistic
        
        h, p, stats = glm.Fcon(C, alpha) performs an F-test for the model glm,
        using contrast matrix C and significance level alpha and returns a
        vector of logicals h indicating rejectance of the null hypothesis
            H0: (C^T b)_1 = 0  and  ...  and  ... (C^T b)_q = 0
        and the vector of p-values in favor of the alternative hypothesis
            H1: (C^T b)_1 != 0  or  ...  or  ... (C^T b)_q != 0
        as well as further information on the statistical test [1].
        
        [1] https://github.com/JoramSoch/MACS/blob/master/ME_GLM_con.m
        """
        
        # expand contrast, if necessary
        if len(C.shape) == 1:
            C = np.expand_dims(C,1)
        q = C.shape[1]
        
        # estimate model parameters
        B_est, s2_est = self.MLE()
        s2_unb        =(self.n/(self.n-self.p)) * s2_est
        
        # compute beta covariance
        if self.iid:
            covB = np.linalg.inv(self.X.T @ self.X)
        else:
            covB = np.linalg.inv(self.X.T @ self.P @ self.X)
        
        # calculate t-statistics
        C_cov_C = C.T @ covB @ C
        inv_CcC = np.linalg.inv(C_cov_C)
        con_est = C.T @ B_est
        num_est = np.zeros(self.v)
        for j in range(self.v):
            num_est[j] = con_est[:,j].T @ inv_CcC @ con_est[:,j]
        
        # calculate p-values
        stats = {'Fstat': (1/q)*num_est/s2_unb, 'df': [q, self.n-self.p]}
        p     = 1 - sp.stats.f.cdf(stats['Fstat'], stats['df'][0], stats['df'][1])
        h     = p < alpha
        
        # return test statistics
        return h, p, stats

# function: hemodynamic response function
#-----------------------------------------------------------------------------#
def spm_hrf(dt=0.1, p=[6,16,1,1,6,0,32]):
    """
    Calculate Hemodynamic Response Function
    HRF = spm_hrf(dt, p)
        
        dt  - float; temporal resolution of the HRF (default: 0.1 sec)
        p   - list of floats; HRF parameters (default: see below)
        
        HRF - array of floats; the hemodynamic response function
        
    HRF = spm_hrf(dt, p) returns the hemodynamic response function (HRF) with
    temporal resolution dt and HRF parameters p. The default values of the HRF
    parameters are [1]:
        
        p[0] - delay of response (relative to onset)          6
        p[1] - delay of undershoot (relative to onset)       16
        p[2] - dispersion of response                         1
        p[3] - dispersion of undershoot                       1
        p[4] - ratio of response to undershoot                6
        p[5] - onset [sec]                                    0
        p[6] - length of kernel [sec]                        32
        
    The HRF is created as a maximum of two scaled probability density functions
    of the gamma distributions with parameters derived from p and dt. The
    default values lead to the first peak at around 6 seconds and post-stimulus
    undershoot between 10 and 20 seconds.
    
    Note: "spm_hrf.m" [1] specifies dt in terms of the repetition time (TR) and
    microtime resolution (T). When following this convention, call the present
    function via "spm_hrf(TR/T, ...)".
    
    [1] https://github.com/spm/spm12/blob/master/spm_hrf.m
    """
    
    #  create HRF as mixture of gammas
    #-------------------------------------------------------------------------#
    u   = np.arange(0,(math.ceil(p[6]/dt)+1)) - p[5]/dt
    hrf = sp.stats.gamma.pdf(u, p[0]/p[2], scale=p[2]/dt) - \
          sp.stats.gamma.pdf(u, p[1]/p[3], scale=p[3]/dt)/p[4]
    # Note: "spm_Gpdf.m" accepts *scale* parameters according to its help text,
    # but "spm_hrf.m" uses *rate* parameters dt/p[2] and dt/p[3] at this point.
    # Since sp.stats.gamma also takes scale keyword arguments, these ratios
    # have been reversed here.
    hrf = hrf[0:math.floor(p[6]/dt)]
    hrf = hrf/np.max(np.abs(hrf))
    # Note: "spm_hrf.m" divides by the vector sum at this point, but this was
    # changed to division by maximum here.
    return hrf

# function: hemodynamic basis functions
#-----------------------------------------------------------------------------#
def spm_get_bf(dt=0.1, name='HRF', p=None, order=1):
    """
    Get Hemodynamic Basis Functions
    BF = spm_get_bf(dt, name, p, order)
        
        dt    - float; temporal resolution of the HRF (default: 0.1 sec)
        name  - string; name of the basis set (default: "HRF")
        p     - list of floats; parameter values (default: see "spm_hrf")
        order - int; order of the basis set (default: 1; value: 0-3)
        
        BF    - time x order matrix; the hemodynamic basis functions
        
    BF = spm_get_bf(dt, name, p, order) returns hemodynamic basis functions [1]
    of specified order from the basis set name with parameters p at temporal
    resolution dt.
    
    Note: "spm_get_bf.m" [1] accepts a struct variable "xBF" as input which has
    fields "dt", "name", "order" and others. There are additional basis sets
    such as "Fourier", "Gamma" and "FIR". Currently, only "HRF" is implemented
    in the present function.
    
    [1] https://github.com/spm/spm12/blob/master/spm_get_bf.m
    """
    
    # get canonical HRF
    #-------------------------------------------------------------------------#
    if p is None: p = [6,16,1,1,6,0,32]
    hrf = spm_hrf(dt, p)
    bf  = np.zeros((hrf.size,order))
    
    # basis set "HRF"
    #-------------------------------------------------------------------------#
    if name == 'HRF':
        
        # canonical HRF
        if order > 0:
            bf[:,0] = spm_hrf(dt, p)
        
        # time derivative
        if order > 1:
            dp      = 1
            p[5]    = p[5] + dp
            bf[:,1] = (bf[:,0] - spm_hrf(dt, p))/dp
          # bf[:,1] = bf[:,1] / np.max(np.abs(bf[:,1]))
            p[5]    = p[5] - dp
        
        # dispersion derivative
        if order > 2:
            dp      = 0.01
            p[2]    = p[2] + dp
            bf[:,2] = (bf[:,0] - spm_hrf(dt, p))/dp
          # bf[:,2] = bf[:,2] / np.max(np.abs(bf[:,2]))
            p[2]    = p[2] - dp
    
    # unknwon basis set
    #-------------------------------------------------------------------------#
    else:
        err_msg = 'Unknown basis set: "{}". Basis set must be "HRF".'
        raise ValueError(err_msg.format(name))
    
    # return basis functions
    #-------------------------------------------------------------------------#
    return bf

# function: restricted maximum likelihood
#-----------------------------------------------------------------------------#
def spm_reml(Y, X, Q, N=1, K=32, R=4, hE=0, hP=np.exp(-8)):
    """
    Restricted Maximum Likelihood Estimation of Covariance Components
    V, Eh, Ph, F, Acc, Com = spm_reml(Y, X, Q, N, K, R, hE, hP)
        
        Y   - n x v matrix; data matrix
        X   - n x p matrix; design matrix (0, if n/a)
        Q   - list of arrays; covariance components
        N   - int; number of samples (default: 1)
        K   - int; number of iterations (default: 32)
        R   - float; regularization parameter (default: 4)
        hE  - float; prior expectations of hyperparameters (default: 0)
        hP  - float; prior precisions of hyperparameters (default: exp(-8))
        
        V   - n x n matrix; estimated covariance matrix
        Eh  - q x 1 vector; posterior expectations of hyperparameters
        Ph  - q x q matrix; posterior precisions of hyperparameters
        F   - float; free energy, ReML objective function
        Acc - float; model accuracy
        Com - float; model complexity (F = Acc - Com)
        
    V, Eh, Ph, F, Acc, Com = spm_reml(Y, X, Q, N, K, R, hE, hP) estimates
    the contribution of the covariance components Q in the errors of Y, after
    removing X [1]. If hyperparameter h[i] is the contribution of Q[i], then
    this routine solves the following equation system:
        Cov[Y|X,Q] = h[1]*Q[1] + ... + h[q]*Q[q] .
    The function returns the covariance matrix
        V = Eh[1]*Q[1] + ... + Eh[q]*Q[q]
    as well as posterior expectations Eh, posterior precision matrix Ph, the
    free energy F, model accuracy Acc and model complexity Com.
    
    The algorithm first estimates the sample covariance matrix YY and then
    performs a Fisher-scoring ascent on F to find ReML estimates of h [1].
    
    Note: "spm_reml.m" [1] uses YY as the first argument, lacks the input
    parameter K and calls the regularization parameter t.
    
    [1] https://github.com/spm/spm12/blob/master/spm_reml.m
    """
    
    # get dimensions
    #-------------------------------------------------------------------------#
    if type(X) != np.ndarray:
        X = np.zeros((Y.shape[0],0))
    n = Y.shape[0]
    v = Y.shape[1]
    p = X.shape[1]
    q = len(Q)
    
    # compute sample covariance of Y
    #-------------------------------------------------------------------------#
    YY = (1/v) * (Y @ Y.T)
    
    # compute orthonormal basis of X
    #-------------------------------------------------------------------------#
    if X.shape[1] > 0:
        X = sp.linalg.orth(X)
        
    # initialize hyperparameters
    #-------------------------------------------------------------------------#
    # B_est = GLM(Y, X).OLS()
    # E_est = Y - X @ B_est
    # EE    = (1/v) * (E_est @ E_est.T)
    # yE    = np.reshape(EE, (EE.size,1), order='F')
    # XE    = np.zeros((EE.size, q))
    # for i in range(q):
    #     XE[:,i] = np.reshape(Q[i], EE.size, order='F')
    # h = GLM(yE, XE).OLS()
    # del B_est, E_est, EE, yE, XE, s2
    
    # specify hyperpriors
    #-------------------------------------------------------------------------#
    h = np.zeros((q,1))
    d = np.zeros(n)
    for i in range(q):
        c      = np.diag(Q[i])
        h[i,0] = float(np.any(np.logical_and(c, np.logical_not(d))))
        d      = np.logical_or(d, c)
    hE = hE*np.ones((q,1))
    hP = hP*np.eye(q)
    dF = np.inf
    In = np.eye(n)
    Zn = np.zeros((n,n))
    
    # ReML via expectation maximization (EM) and Variational Bayes (VB)
    #-------------------------------------------------------------------------#
    for k in range(K):
    
        # I-step: initialize, compute current estimate of V
        #---------------------------------------------------------------------#
        C = Zn.copy()
        for i in range(q):
            C = C + h[i]*Q[i]
        
        # E-step: expectation, compute conditional covariance Cov[B|Y]
        #---------------------------------------------------------------------#
        iC  = np.linalg.inv(C)
        iCX = iC @ X
        if X.shape[1] > 0:
            Cq = np.linalg.inv(X.T @ iCX)
        else:
            Cq = np.zeros((p,p))
        
        # M-step: maximization, estimate hyperparameters h
        #---------------------------------------------------------------------#
        
        # first deriative: gradient dF/dh
        P    = iC - iCX @ Cq @ iCX.T
        U    = In - P @ YY
        PQ   = [[] for i in range(q)]
        dFdh = np.zeros((q,1))
        for i in range(q):
            PQ[i]   = P @ Q[i]
            dFdh[i] = -N/2*np.trace(PQ[i] @ U)
        
        # second deriative: expected curvature dF/dhh
        dFdhh = np.zeros((q,q))
        for i in range(q):
            for j in range(i,q):
                dFdhh[i,j] = -N/2*np.trace(PQ[i] @ PQ[j])
                dFdhh[j,i] = dFdhh[i,j]
        
        # U-step: update, compute new parameter estimates
        #---------------------------------------------------------------------#
        
        # add hyperpriors
        e     = h     - hE
        dFdh  = dFdh  - hP @ e
        dFdhh = dFdhh - hP
        
        # Fisher-scoring
        t  = np.exp(R - np.linalg.slogdet(dFdhh)[1]/q)
        # https://github.com/spm/spm12/blob/master/spm_dx.m#L70
        dh = (sp.linalg.expm(t*dFdhh) - np.eye(q)) @ np.linalg.inv(dFdhh) @ dFdh
        # https://github.com/spm/spm12/blob/master/spm_dx.m#L2
        h  = h + dh
        
        # C-step: convergence, check regularization and convergence
        #---------------------------------------------------------------------#
        
        # if predicted change in F increases, increase regularization
        pF = dFdh.T @ dh
        if pF > dF:
            R = np.max(np.array([R-1, -8]))
        else:
            R = np.min(np.array([R+1/4, 8]))
        
        # if near-phase transition, start again with more precise priors
        if not np.isfinite(float(pF)) or np.linalg.norm(dh, ord=np.inf) > 1e6:
            V, Eh, Ph, F, Acc, Com = spm_reml(Y, X, Q, N, K, R, 0, hP[1,1]*2)
            return V, Eh, Ph, F, Acc, Com
        else:
            dF = pF
        
        # if change in log-evidence is <1%, stop estimation
        print('   - ReML iteration {}: {:.4f} [{:.2f}]'.format(k+1, float(pF), t))
        if float(dF) < 1e-3: break
    
    # build final covariance
    #-------------------------------------------------------------------------#
    V = Zn
    for i in range(q):
        V = V + h[i,0]*Q[i]
    
    # collect posterior distribution
    #-------------------------------------------------------------------------#
    Eh = h
    Ph = -dFdhh
    
    # get free energy and partition
    #-------------------------------------------------------------------------#
    Ft  = np.trace(hP @ np.linalg.inv(Ph)) - q
    Acc = 1/2*Ft - np.trace(V @ P @ YY @ P) - N*n/2*np.log(2*math.pi) - N/2*np.linalg.slogdet(V)[1]
    # Accuracy = posterior E[...] of log p(Y|h)
    Com = 1/2*Ft + 1/2*e.T @ hP @ e + 1/2*np.linalg.slogdet(Ph @ np.linalg.inv(hP))[1]
    # Complexity = KL divergence of N(Eh,Ph) from N(hE,hP)
    F   = Acc - Com
    
    # return output values
    #-------------------------------------------------------------------------#
    return V, Eh, Ph, F, Acc, Com
    
# function: create design matrix
#-----------------------------------------------------------------------------#
def get_des_mat(names, onsets, durations, pmod=None, R=None, settings={'HRF': 'spm_hrf', 'mc': True}):
    """
    Create First-Level Design Matrix
    X, L = get_des_mat(names, onsets, durations, R, settings)
    
        names     - list of strings; experimental condition names
        onsets    - list of arrays of floats; onsets for each trial in each condition
        durations - list of arrays of floats; durations of each trial in each condition
        pmod      - list of dictionaries; parametric modulators for each condition:
        o name    - list of strings; names of the parametric modulator variables
        o vals    - list of vectors; values of the parametric modulator variables
        R         - n x r matrix; nuisance regressors, variables of no interest
        settings  - dictionary; settings for design matrix creation:
        o n       - int; number of scans
        o TR      - float; repetition time
        o mtr     - int; microtime resolution, i.e. number of HRF bins per TR
        o mto     - int; microtime onset, i.e. HRF bin to sample from
        o HRF     - string; method for creating the hemodynamic response function
        o mc      - bool; mean-centering of parametric modulators and additional regressors
        
        X         - n x p matrix; first-level fMRI design matrix
        L         - 1 x p list; labels for all regressors in X
        
    X, L = get_des_mat(names, onsets, durations, R, settings) returns a design
    matrix for first-level fMRI data analysis using names, onsets and durations
    and parametric modulators pmod in SPM format, additional regressors R as a
    matrix and user-specified settings as a dictionary [1].
    
    Note: In SPM, the field "vals" of "pmod" is called "param". Additionally,
    "pmod" has the field "poly" in SPM (referring the polynomial extension in
    which the parametric modulator variable is included, most often set to 1)
    which is ommitted here.
    
    [1] https://github.com/JoramSoch/ITEM/blob/master/ITEM_get_des_mat.m
    """
    
    # extract settings
    #-------------------------------------------------------------------------#
    n   = settings['n']
    TR  = settings['TR']
    mtr = settings['mtr']
    mto = settings['mto']
    dt  = TR/mtr
    
    # get hemodynamic response function
    #-------------------------------------------------------------------------#
    if settings['HRF'] == 'spm_hrf':
        HRF = spm_hrf(dt)
    elif settings['HRF'] == 'none':
        HRF = np.array(1)
    else:
        HRF = np.ones((1,mtr))
    
    # prepare parametric modulators
    #-------------------------------------------------------------------------#
    if pmod is None:
        pmod = [{} for i in range(len(names))]
    else:
        pmod = pmod + [{} for i in range(len(names)-len(pmod))]
    
    # create design matrix
    #-------------------------------------------------------------------------#
    X = np.zeros((n,0))
    L = []
    for i in range(len(names)):
        
        # calculate timing in microtime space
        #---------------------------------------------------------------------#
        o = np.round(np.array(onsets[i])/dt)
        d = np.round(np.array(durations[i])/dt)
        y = np.zeros(2*round((n*TR)/dt))

        # create condition stimulus function
        #---------------------------------------------------------------------#
        for k in range(len(o)):
            y[int(o[k]):(int(o[k])+int(d[k]))] = 1

        # convolve with HRF and sample
        #---------------------------------------------------------------------#
        x = np.convolve(y, HRF)
        x = x[np.arange(mto-1, n*mtr, mtr)]
        
        # add regressor to design matrix
        #---------------------------------------------------------------------#
        X = np.c_[X, x]
        L.append(names[i])
        
        # check for parametric modulators
        #---------------------------------------------------------------------#
        if bool(pmod[i]):
            for j in range(len(pmod[i]['name'])):
                
                # extract parametric modulator variable
                #-------------------------------------------------------------#
                y = np.zeros(2*round((n*TR)/dt))
                p = np.array(pmod[i]['vals'][j])
                if settings['mc']:
                    p = p - np.mean(p)
                
                # create parametric modulator function
                #-------------------------------------------------------------#
                for k in range(len(o)):
                    y[int(o[k]):(int(o[k])+int(d[k]))] = p[k]
                
                # convolve with HRF and sample
                #-------------------------------------------------------------#
                x = np.convolve(y, HRF)
                x = x[np.arange(mto-1, n*mtr, mtr)]
                
                # add regressor to design matrix
                #-------------------------------------------------------------#
                X = np.c_[X, x]
                L.append('{} x {}'.format(names[i], pmod[i]['name'][j]))
    
    # add additional regressors
    #-------------------------------------------------------------------------#
    if R is not None:
        if settings['mc']:
            R = R - np.tile(np.mean(R,0),(n,1))
        for i in range(R.shape[1]):
            X = np.c_[X, R[:,i]]
            L.append('R'+str(i+1))
    
    # return design matrix
    #-------------------------------------------------------------------------#
    X = X / np.tile(np.max(np.abs(X),0),(n,1))
    return X, L

# test area / debugging section
#-----------------------------------------------------------------------------#
if __name__ == '__main__':
    
    # import packages
    import matplotlib.pyplot as plt
    # enter "%matplotlib qt" in Spyder before
    
    # specify what to test
    what_to_test = 'spm_get_bf'
    
    # test "GLM"
    if what_to_test == 'GLM':
        n12 = 50
        y1  = np.random.normal(1, 3, size=n12)
        y2  = np.random.normal(0, 3, size=n12)
        y   = np.expand_dims(np.concatenate((y1, y2)), 1)
        X   = np.kron(np.eye(2), np.ones((n12,1)))
        V   = np.eye(2*n12)
        glm = GLM(y, X, V=None)
        print(glm.OLS())
        print(glm.WLS())
        print(glm.MLE())
        print(glm.MLL())
        print(glm.tcon(np.array([+1,-1]), alpha=0.01))
        print(glm.Fcon(np.array([+1,-1]), alpha=0.001))
    
    # test "spm_hrf"
    if what_to_test == 'spm_hrf':
        dt  = 0.1
        HRF = spm_hrf(dt)
        t   = np.arange(0,HRF.size*dt,dt)
        plt.plot(t, HRF)
    
    # test "spm_hrf"
    if what_to_test == 'spm_get_bf':
        dt = 0.1
        BF = spm_get_bf(dt, order=3)
        t  = np.arange(0,BF.shape[0]*dt,dt)
        plt.plot(t, BF[:,0])
        plt.plot(t, BF[:,1])
        plt.plot(t, BF[:,2])
    
    # test "spm_reml"
    if what_to_test == 'spm_reml':
        
        # generate design matrix
        names     = ['C1', 'C2']
        onsets    = [[10, 40, 50, 80], [20, 30, 60, 70, 90]]
        durations = [[2, 2, 2, 2], [2, 2, 2, 2, 2]]
        X, L      = get_des_mat(names, onsets, durations, \
                                settings={'n': 100, 'TR': 1, 'mtr': 25, 'mto': 13, 'HRF': 'spm_hrf'})
        
        # sample data matrix
        n = 100
        N = 1000
        h = [0.5, 1.5]
        Q = [np.eye(n), sp.linalg.toeplitz(np.power(0.1, np.arange(0,n)))]
        V = h[0]*Q[0] + h[1]*Q[1]
        b = np.array([[2,1]]).T
        Y = np.random.multivariate_normal(np.squeeze(X @ b), V, size=N).T
        
        # perform ReML estimation
        V_est, h_est, h_prec, F, Acc, Com = spm_reml(Y, X, Q)
        print(h_est[:,0])
    
    # test "get_des_mat"
    if what_to_test == 'get_des_mat':
        names     = ['C1', 'C2']
        onsets    = [[10, 40, 50, 80], [20, 30, 60, 70, 90]]
        durations = [[2, 2, 2, 2], [2, 2, 2, 2, 2]]
        pmod      = [{'name': ['M1a', 'M1b'], 'vals': [[1,2,3,4],[2,1,1,2]]}]
        R         = np.random.uniform(-1, +1, size=(100,6))
        settings  = {'n': 100, 'TR': 1, 'mtr': 25, 'mto': 13, 'HRF': 'spm_hrf', 'mc': True}
        X, L      = get_des_mat(names, onsets, durations, pmod, R, settings)
        plt.imshow(X)
        print(L)