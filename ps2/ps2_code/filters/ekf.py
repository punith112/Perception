"""
This file implements the Extended Kalman Filter.
"""

import numpy as np
from math import *
import matplotlib.pyplot as plt

from filters.localization_filter import LocalizationFilter
from tools.task import get_motion_noise_covariance
from tools.task import get_observation as get_expected_observation
from tools.task import get_prediction
from tools.task import wrap_angle


def jacobian_G(theta, u):
	drot1, dtrans, drot2 = u
	G = np.eye(3)
	G[0,2] = -dtrans * sin(theta+drot1)
	G[1,2] = dtrans * cos(theta+drot1)
	return G

def jacobian_V(theta, u):
	drot1, dtrans, drot2 = u
	V = np.array([[-dtrans*sin(theta+drot1), cos(theta+drot1), 0],
				  [dtrans*cos(theta+drot1),  sin(theta+drot1), 0],
				  [1,                        0,                1]])
	return V

def jacobian_H(lm_pose, mu):
    mx = lm_pose[0]
    my = lm_pose[1]
    q = (mx-mu[0])**2 + (my-mu[1])**2

    # H = np.array([[-(mx-mu[0])/sqrt(q), -(my-mu[1])/sqrt(q), 0],
    #               [ (my-mu[1])/q,       -(mx-mu[0])/q,      -1]])
    H = np.array([[(my-mu[1])/q,       -(mx-mu[0])/q,      -1],
    			  [0,				   0,                  0 ]])
    return H


class EKF(LocalizationFilter):
    def predict(self, u):
        # TODO Implement here the EKF, prediction part. HINT: use the auxiliary functions imported above from tools.task
        self.mu[2] = wrap_angle(self.mu[2])
        self._state_bar.mu = get_prediction(self.mu, u)[np.newaxis].T
        G = jacobian_G(self.mu[2], u)
        V = jacobian_V(self.mu[2], u)
        M = get_motion_noise_covariance(u, self._alphas)
        self._state_bar.Sigma = np.dot( np.dot(G,self.Sigma), G.T) + np.dot( np.dot(V,M), V.T)
        
        # self._state_bar.mu = self._state.mu
        # self._state_bar.Sigma = self._state.Sigma

    def update(self, z):
        # TODO implement correction step
        self.mu_bar[2] = wrap_angle(self.mu_bar[2])
        for lm_id in [int(z[1])]:
        	z_expected = get_expected_observation(self.mu_bar, lm_id)
        	mx = self._field_map.landmarks_poses_x[lm_id]
        	my = self._field_map.landmarks_poses_y[lm_id]
        	H = jacobian_H([mx, my], self.mu_bar)
        	# Q = np.array([[0,0],[0,self._Q]])
        	Q = np.array([[self._Q,0],[0,self._Q]])
        	S = np.dot(np.dot(H, self.Sigma_bar), H.T) + Q
        	K = np.dot( np.dot(self.Sigma_bar, H.T), np.linalg.inv(S) )
        	
	        self._state_bar.mu = self._state_bar.mu + np.dot(K, z[np.newaxis].T-z_expected[np.newaxis].T)
	        self._state_bar.Sigma = np.dot( np.eye(3) - np.dot(K,H), self.Sigma_bar )
        
        self._state.mu = self._state_bar.mu
        self._state.Sigma = self._state_bar.Sigma
