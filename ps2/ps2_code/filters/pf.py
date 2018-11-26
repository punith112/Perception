
"""
Sudhanva Sreesha
ssreesha@umich.edu
28-Mar-2018

This file implements the Particle Filter.
"""

import numpy as np
from numpy.random import uniform, randn
from scipy.stats import norm as gaussian
from scipy.linalg import cholesky
import matplotlib.pyplot as plt

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import get_prediction
from tools.task import sample_from_odometry
from tools.task import wrap_angle


def resampling(X, w):
	# weights normalization:
	w /= sum(w)
	index = np.random.randint(0,len(w))
	betta = 0
	X_new = np.zeros_like(X)
	for i in range(len(w)):
		betta = betta + np.random.uniform(0, 2*max(w))
		while betta > w[index]:
			betta = betta - w[index]
			index = (index + 1) % len(w)
		X_new[:,i] = X[:,index]
	return X_new

# def pose_from_particles(X, w):
# 	w /= sum(w)
# 	x_est = 0
# 	y_est = 0
# 	theta_est = 0
# 	for i in range(len(X)):
# 		x_est = x_est + X[0,i]*w[i]
# 		y_est = y_est + X[1,i]*w[i]
# 		theta_est = theta_est + X[2,i]*w[i]
# 	return np.array([x_est, y_est, theta_est])

def pose_from_particles(X, w):
	return (X * w).sum(axis=1)

class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, beta)
        # TODO add here specific class variables for the PF
        self.Sigma0 = initial_state.Sigma
        self.M = num_particles * 5
        self.X_bar = np.ones((3,self.M)) * initial_state.mu
        self.X = np.ones((3,self.M)) * initial_state.mu
        self.w_bar = np.ones(self.M) / self.M
        self.w = np.ones(self.M) / self.M # uniform weighs distribution
        self.Z = np.zeros(self.M)

    def predict(self, u):
        # # TODO Implement here the PF, perdiction part
        for m in range(1,self.M):
        	self.X_bar[:,m] = sample_from_odometry(self.X[:,m-1], u, self._alphas)

        self.w_bar = self.w / sum(self.w) # keep previous weights
        updated_pose_bar = pose_from_particles(self.X_bar, self.w_bar)
        
        self._state_bar.mu = updated_pose_bar[np.newaxis].T
        self._state_bar.Sigma = get_gaussian_statistics(self.X_bar.T).Sigma

        # self._state_bar.mu = self._state.mu
        # self._state_bar.Sigma = self._state.Sigma

    def update(self, z):
        # TODO implement correction step
        for m in range(1,self.M):
        	self.Z[m] = get_observation(self.X[:,m],z[1])[0]
        # for m in range(1,self.M):
        # 	self.w[m]  = gaussian.pdf(z[1]-self.Z[m], 0, self._Q)
        self.X = resampling(self.X_bar, self.w)
        updated_pose = pose_from_particles(self.X, self.w)

        self._state.mu = updated_pose[np.newaxis].T
        self._state.Sigma = get_gaussian_statistics(self.X.T).Sigma




        # self._state.mu = self._state_bar.mu
        # self._state.Sigma = self._state_bar.Sigma