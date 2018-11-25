"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Unscented Kalman Filter.
"""

import numpy as np
import scipy
from math import *

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation
from tools.task import get_prediction
from tools.task import wrap_angle


class UKF(LocalizationFilter):
    def __init__(self, *args, **kwargs):
        super(UKF, self).__init__(*args, **kwargs)
        # TODO add here specific class variables for the UKF
        self.n = 7
        self.X = np.zeros((self.n,2*self.n+1))
        self.X_x = self.X[:3,:]
        self.X_u = self.X[3:6,:]
        self.X_z = self.X[6,:]
        self.X_x_bar = np.zeros((3,2*self.n+1))
        lamda = 3-self.n # 1
        self.gamma = sqrt(self.n+lamda)
        self.w_m = np.concatenate([np.array([lamda/(self.n+lamda)]), 0.5/(self.n+lamda)*np.ones((2*self.n))])
        alpha = 1.000; beta = -0.01
        self.w_c = self.w_m; self.w_c[0] += 1-alpha**2+beta
        self.Z_bar  = np.zeros((2,2*self.n+1))
        self.S = 0
        self.Sigma_xz = np.zeros((3,1))


    def predict(self, u):
        # TODO Implement here the UKF, perdiction part
        M = get_motion_noise_covariance(u, self._alphas)
        # Q = np.array([[self._Q,0],[0,self._Q]])
        Q = self._Q
        self.mu_a = np.concatenate((self.mu, np.zeros(4)))
        self.Sigma_a = scipy.linalg.block_diag(self.Sigma, M, Q)
        L = scipy.linalg.cholesky(self.Sigma_a, lower=True)

        # Sigma-points generation
        self.X[:,0] = self.mu_a
        for i in range(1,len(L)+1):
            self.X[:,i] = self.mu_a + (self.gamma*L)[:,i-1]
        for i in range(len(L)+1,2*len(L)+1):
            self.X[:,i] = self.mu_a - (self.gamma*L)[:,i-len(L)-1]
        self.X_x = self.X[:3,:]; self.X_u = self.X[3:6,:]; self.X_z = self.X[6,:]

        # Pass sigma points through motion model and compute Gaussian statistics
        for i in range(2*self.n+1):
            self.X_x_bar[:,i] = get_prediction(self.X_x[:,i], u+self.X_u[:,i])
        self._state_bar.mu = (self.w_m * self.X_x_bar).sum(axis=1)[np.newaxis].T
        
        Sigma_bar = 0
        for i in range(2*self.n+1):
            Sigma_bar += self.w_c[i] * ( (self.X_x_bar[:,i]-self.mu_bar)[np.newaxis].T*(self.X_x_bar[:,i]-self.mu_bar).T )
        self._state_bar.Sigma = Sigma_bar
        

        # self._state_bar.mu = self._state.mu
        # self._state_bar.Sigma = self._state.Sigma



    def update(self, z):
        # TODO implement correction step
        # Predict observations at sigma points and compute Gaussian statistics
        for i in range(2*self.n+1):
            self.Z_bar[0,i] = get_observation(self.X_x_bar[:,i], int(z[1]))[0] + self.X_z[i]
            self.Z_bar[1,i] = z[1] 
        z_expected = ( self.w_m * self.Z_bar ).sum(axis=1)
        

        self.S = 0
        for i in range(2*self.n+1):
            self.S += self.w_c[i] * ( (self.Z_bar[0,i]-z_expected[0])**2 )

        self.Sigma_xz = 0
        for i in range(2*self.n+1):
            self.Sigma_xz += self.w_c[i] * ( (self.X_x_bar[:,i]-self.mu_bar)[np.newaxis].T*(self.Z_bar[0,i]-z_expected[0] ) )

        K =  self.Sigma_xz / (self.S)

        self._state.mu = self._state_bar.mu + K * (z[0]-z_expected[0])
        self._state.Sigma = self._state_bar.Sigma - np.dot(K*self.S, K.T)



        # self._state.mu = self._state_bar.mu
        # self._state.Sigma = self._state_bar.Sigma