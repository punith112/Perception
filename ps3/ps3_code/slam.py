"""
Sudhanva Sreesha
ssreesha@umich.edu
24-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018
"""

import numpy as np
from math import *
from scipy.linalg import block_diag

from tools.task import get_prediction, wrap_angle
from tools.plot import plot2dcov

from slam_folder.slamBase import SlamBase
from field_map import FieldMap


class SimulationSlamBase(SlamBase):
    def __init__(self, slam_type, data_association, update_type, args, initial_state):
        super(SimulationSlamBase, self).__init__(slam_type, data_association, update_type, np.array(args.beta))
        self.iR = 3 # [x, y, theta] of the robot
        self.iM = 0 # [mx my] for each landmark from N
        self.params = args
        self.state = initial_state
        self.state_bar = self.state
        self.R = np.zeros_like(self.state.Sigma)
        self.G = np.zeros_like(self.state.Sigma)
        self.Q = np.diag([self.params.beta[0]**2, self.params.beta[1]**2])
        self.lm_seq = [] # sequence of lms
        self.N = 8 # number of all lms
        self.m = np.empty((self.N, 2))
        for i in range(self.N):
            self.m[i,:] = np.array([None, None])

    def initialize_new_landmark(self, lm_id, z, batch_i):
        self.mu_bar[2] = wrap_angle(self.mu_bar[2])
        self.mu[2] = wrap_angle(self.mu[2])
        iR = self.iR
        iM = self.iM
        
        self.lm_seq.append(lm_id)
        self.G = block_diag(self.G, np.eye(2))
        self.R = block_diag(self.R, np.zeros((2,2)))
        r = z[batch_i,0]
        phi = wrap_angle(z[batch_i,1])
        theta =  wrap_angle(self.mu_bar[2])
        m_new = self.mu_bar[:2] + np.array([r*cos(wrap_angle(phi+theta)), r*sin(wrap_angle(phi+theta))])
        self.m[int(lm_id),:] = m_new
        self.state.mu = np.append(self.mu, m_new)[np.newaxis].T # append new lm coords to the state vector
        L = self.get_jacobian_L(r, phi, theta)
        W = self.get_jacobian_W(r, phi, theta)
        Sx = self.Sigma_bar[:iR,:iR]; Sxm = self.Sigma_bar[:iR, iR:iR+iM]
        Smx = self.Sigma_bar[iR:iR+iM, :iR]; Sm = self.Sigma_bar[iR:iR+iM, iR:iR+iM]
        # new Sigma sub-matrices
        Sur = Sx @ L.T; Sr = Smx @ L.T; Sbr = L@Sx@L.T + W@self.Q@W.T
        Sbl = L @ Sx; Sb = L @ Sxm
        # if len(self.lm_ids)==0:
        #     self.state_bar.Sigma = np.block([[Sx,  Sur],
        #                                      [Sbl, Sbr]])
        self.state_bar.Sigma = np.block([[Sx, Sxm, Sur],
                                         [Smx, Sm, Sr],
                                         [Sbl, Sb, Sbr]])

        self.mu_bar[2] = wrap_angle(self.mu_bar[2])
        self.mu[2] = wrap_angle(self.mu[2])
        

    def predict(self, u, dt=None):
        self.mu_bar[2] = wrap_angle(self.mu_bar[2])
        self.mu[2] = wrap_angle(self.mu[2])

        iR = self.iR # Robot indexes
        self.iM = 2+2*len(self.lm_seq)
        iM = self.iM # Map indexes
        mu_r = self.mu[:iR]

        G_x = self.get_g_prime_wrt_state(mu_r, u)
        V_x = self.get_g_prime_wrt_motion(mu_r, u)
        M_t = self.get_motion_noise_covariance(u)

        R_t = V_x @ M_t @ V_x.T
        self.R[:R_t.shape[0], :R_t.shape[1]] = R_t
        self.G[:G_x.shape[0], :G_x.shape[1]] = G_x

        # EKF prediction of the state mean.
        self.state_bar.mu[:iR] = get_prediction(mu_r, u)[np.newaxis].T
        self.state_bar.mu[2] = wrap_angle(self.mu_bar[2])
        # EKF prediction of the state covariance.
        self.state_bar.Sigma[:iR,:iR] = G_x @ self.Sigma[:iR, :iR] @ G_x.T + R_t
        if iM > 0:
            self.state_bar.Sigma[:iR, iR:iR+iM] = G_x @ self.Sigma[:iR, iR:iR+iM]
            self.state_bar.Sigma[iR:iR+iM, :iR] = self.state.Sigma[iR:iR+iM, :iR] @ G_x.T

        self.mu_bar[2] = wrap_angle(self.mu_bar[2])
        self.mu[2] = wrap_angle(self.mu[2])

        return self.mu_bar, self.Sigma_bar


    def update(self, z):
        self.mu_bar[2] = wrap_angle(self.mu_bar[2])
        self.mu[2] = wrap_angle(self.mu[2])
        for lm_id in z[:,2]: # for all observed features
            batch_i = np.where(z==lm_id)[0][0] # 0th or 1st of observed lms in the batch
            if (lm_id not in self.lm_seq): # lm never seen before
                self.initialize_new_landmark(lm_id, z, batch_i)
            delta = np.array(self.m[int(lm_id)] - self.mu_bar[:2])
            q = np.dot( delta, delta.T )
            z_expected = np.array([ sqrt(q), wrap_angle( atan2(delta[1],delta[0])-self.mu_bar[2] ) ])
            H = self.get_jacobian_H(q, delta, int(lm_id))
            S = (H @ self.Sigma_bar) @ H.T + self.Q
            K = (self.Sigma_bar @ H.T) @ np.linalg.inv(S)

            Z = z[batch_i,:2]
            self.state_bar.mu = self.state_bar.mu + K @ (Z - z_expected)[np.newaxis].T
            self.state_bar.Sigma = (np.eye(K.shape[0]) - (K @ H)) @ self.Sigma_bar

        self.mu_bar[2] = wrap_angle(self.mu_bar[2])
        self.mu[2] = wrap_angle(self.mu[2])

        self.state.mu = self.state_bar.mu
        self.state.Sigma = self.state_bar.Sigma

        return self.mu, self.Sigma


    def get_motion_noise_covariance(self, motion):
        """
        :param motion: The motion command at the current time step (format: [drot1, dtran, drot2]).
        :return: The covariance of the motion noise (in motion space).
        """
        drot1, dtran, drot2 = motion
        a1, a2, a3, a4 = self.params.alphas

        return np.diag([a1 * drot1 ** 2 + a2 * dtran ** 2,
                        a3 * dtran ** 2 + a4 * (drot1 ** 2 + drot2 ** 2),
                        a1 * drot2 ** 2 + a2 * dtran ** 2])

    @staticmethod
    def get_g_prime_wrt_state(state, motion):
        """
        :param state: The current state mean of the robot (format: np.array([x, y, theta])).
        :param motion: The motion command at the current time step (format: np.array([drot1, dtran, drot2])).
        :return: Jacobian of the state transition matrix w.r.t. the state.
        """
        drot1, dtran, drot2 = motion

        return np.array([[1, 0, -dtran * np.sin(state[2] + drot1)],
                         [0, 1, dtran * np.cos(state[2] + drot1)],
                         [0, 0, 1]])

    @staticmethod
    def get_g_prime_wrt_motion(state, motion):
        """
        :param state: The current state mean of the robot (format: np.array([x, y, theta])).
        :param motion: The motion command at the current time step (format: np.array([drot1, dtran, drot2])).
        :return: Jacobian of the state transition matrix w.r.t. the motion command.
        """
        drot1, dtran, drot2 = motion

        return np.array([[-dtran * np.sin(state[2] + drot1), np.cos(state[2] + drot1), 0],
                         [dtran * np.cos(state[2] + drot1), np.sin(state[2] + drot1), 0],
                         [1, 0, 1]])

    def get_jacobian_H(self, q, delta, lm_id):
        Hx = (1/q) * np.array([ [-sqrt(q)*delta[0], -sqrt(q)*delta[1], 0],
                                  [delta[1],        -delta[0],        -q] ])
        Hj = (1/q) * np.array([ [sqrt(q)*delta[0],  sqrt(q)*delta[1]],
                                [-delta[1],         delta[0]       ]])
        n_lms = len(self.lm_seq)
        H = np.zeros((2, 3+2*n_lms))
        H[:2,:3] = Hx
        column = np.where(np.array(self.lm_seq)==lm_id)[0][0]
        H[:,3+2*column:3+2*column+2] = Hj
        return H

    @staticmethod
    def get_jacobian_L(r, phi, theta):
        return np.array([[1, 0, -r*sin(wrap_angle(phi+theta))],
                         [0, 1, r*cos(wrap_angle(phi+theta)) ]])

    @staticmethod
    def get_jacobian_W(r, phi, theta):
        return np.array([[cos(phi+theta), -r*sin(wrap_angle(phi+theta))],
                         [sin(phi+theta), r*cos(wrap_angle(phi+theta))]])

    @property
    def mu_bar(self):
        """
        :return: The state mean after the update step.
        """
        return self.state_bar.mu.T[0]

    @property
    def Sigma_bar(self):
        """
        :return: The state covariance after the update step (shape: 3x3).
        """
        return self.state_bar.Sigma


    @property
    def mu(self):
        """
        :return: The state mean after the update step.
        """
        return self.state.mu.T[0]

    @property
    def Sigma(self):
        """
        :return: The state covariance after the update step (shape: 3x3).
        """
        return self.state.Sigma

