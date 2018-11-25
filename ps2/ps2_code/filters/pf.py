
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

from filters.localization_filter import LocalizationFilter
from tools.task import get_gaussian_statistics
from tools.task import get_observation
from tools.task import sample_from_odometry
from tools.task import wrap_angle


def resampling(particleList, w):
	# weights normalization:
	w /= sum(w)
	index = np.random.randint(0,len(w))
	betta = 0
	newParticleList = []
	for i in range(len(w)):
		betta = betta + np.random.uniform(0, 2*max(w))
		while betta > w[index]:
			betta = betta - w[index]
			index = (index + 1) % len(w)
		newParticleList.append(particleList[index])
	particleList = newParticleList
	return particleList

def pose_from_particles(particleList, w):
	x_est = 0
	y_est = 0
	theta_est = 0
	for i in range(len(particleList)):
		x_est = x_est + particleList[i][0]*w[i]
		y_est = y_est + particleList[i][1]*w[i]
		theta_est = theta_est + particleList[i][2]*w[i]
	return np.array([x_est, y_est, theta_est])

class PF(LocalizationFilter):
    def __init__(self, initial_state, alphas, beta, num_particles, global_localization):
        super(PF, self).__init__(initial_state, alphas, beta)
        # TODO add here specific class variables for the PF
        self.X = []
        self.x0 = initial_state.mu.T[0]
        self.X.append( self.x0 )
        self.Sigma0 = initial_state.Sigma
        self.M = num_particles
        self.w = np.ones(self.M) / self.M # uniform weighs distribution

    def predict(self, u):
        # TODO Implement here the PF, perdiction part
        for m in range(1,self.M):
        	self.X.append( sample_from_odometry(self.X[-1], u, self._alphas) )

        # self._state_bar.mu = self._state.mu
        # self._state_bar.Sigma = self._state.Sigma

    def update(self, z):
        # TODO implement correction step
        for m in range(1,self.M):
        	z_particle = get_observation(self.X[m],z[1])
        	self.w[m] = gaussian.pdf(z_particle[0], z[0], self._Q)
        self.X = resampling(self.X, self.w)
        updated_pose = pose_from_particles(self.X, self.w)

        self._state.mu = updated_pose[np.newaxis].T
        self._state.Sigma = self._state_bar.Sigma
        # self._state.Sigma = get_gaussian_statistics(np.array(self.X)).Sigma
