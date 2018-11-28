
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
from math import sqrt

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import get_prediction
from tools.task import sample_from_odometry
from tools.task import wrap_angle


def jacobian_H(lm_pose, mu):
    mx = lm_pose[0]
    my = lm_pose[1]
    q = (mx-mu[0])**2 + (my-mu[1])**2
    H = np.array([(my-mu[1])/q,       -(mx-mu[0])/q,      -1])
    return H

def low_variance_sampler(X, w):
    # w_new = w / sum(w)
    w_new = w
    X_new = np.zeros_like(X)
    M = len(w)
    r = uniform(0, 1/M)
    c = w[0]
    i = 0
    for m in range(1, M+1):
        U = r+(m-1)/M
        while(U>c):
            i = (i+1) % M
            c = c+w[i]
        X_new[:,m-1] = X[:,i]
        w_new[m-1] = 1 / M
    return X_new, w_new

def resampling(X, w):
    # weights normalization:
    w /= sum(w)
    M = len(w)
    index = np.random.randint(0,len(w))
    betta = 0
    X_new = np.zeros_like(X)
    for i in range(M):
        betta = betta + np.random.uniform(0, 2*max(w))
        while betta > w[index]:
            betta = betta - w[index]
            index = (index + 1) % len(w)
        X_new[:,i] = X[:,index]
    w = np.ones(M) / M
    return X_new, w

def pose_from_particles(X, w):
	return (X * w).sum(axis=1)

class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, beta)
        # Specific class variables for the PF
        self.Sigma0 = initial_state.Sigma
        self.M = num_particles * 10
        self.X_bar = np.ones((3,self.M)) * initial_state.mu
        self.X = np.ones((3,self.M)) * initial_state.mu
        self.w_bar = np.ones(self.M) / self.M
        self.w = np.ones(self.M) / self.M # uniform weighs distribution
        self.Z = np.zeros(self.M)

    def weights_update(self, z):        
        mx = self._field_map.landmarks_poses_x[int(z[1])]
        my = self._field_map.landmarks_poses_y[int(z[1])]
        H = jacobian_H([mx, my], self.mu)
        Q = self._Q
        S = np.dot(np.dot(H, self.Sigma), H.T) + Q
        for m in range(1,self.M):
            self.Z[m-1] = get_observation(self.X[:,m-1],z[1])[0]
            self.w[m-1] = gaussian.pdf(z[0]-self.Z[m-1], 0, sqrt(S))
        self.w /= sum(self.w)

    def predict(self, u):
        # PF, prediction part
        for m in range(self.M):
        	self.X_bar[:,m] = sample_from_odometry(self.X[:,m], u, self._alphas)
        self.w_bar = self.w / sum(self.w) # keep previous weights
        updated_pose_bar = pose_from_particles(self.X_bar, self.w_bar)
        
        self._state_bar.mu = updated_pose_bar[np.newaxis].T
        self._state_bar.Sigma = get_gaussian_statistics(self.X_bar.T).Sigma

    def update(self, z):
        # PF correction step
        self.weights_update(z)
        self.X, self.w = resampling(self.X_bar, self.w)
        # self.X, self.w = low_variance_sampler(self.X_bar, self.w)
        updated_pose = pose_from_particles(self.X, self.w)
        self._state.mu = updated_pose[np.newaxis].T
        self._state.Sigma = get_gaussian_statistics(self.X.T).Sigma
