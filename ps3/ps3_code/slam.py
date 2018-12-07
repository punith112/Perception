"""
Sudhanva Sreesha
ssreesha@umich.edu
24-Apr-2018

Gonzalo Ferrer
g.ferrer@skoltech.ru
26-Nov-2018
"""

import numpy as np

from tools.task import get_prediction
from slam_folder.slamBase import SlamBase
from field_map import FieldMap


class SimulationSlamBase(SlamBase):
    def __init__(self, slam_type, data_association, update_type, args, initial_state):
        super(SimulationSlamBase, self).__init__(slam_type, data_association, update_type, np.array(args.beta))
        self.iR = 3 # [x, y, theta] of the robot
        num_landmarks_per_side = 4
        self.field_map = FieldMap(num_landmarks_per_side)
        self.N = 2*num_landmarks_per_side
        self.iM = 3*self.N # [mx my s] for each landmark from N
        self.params = args
        self.state_bar = initial_state
        self.state = initial_state
        mx = self.field_map._landmark_poses_x
        my = self.field_map._landmark_poses_y
        self.state.mu = np.vstack([initial_state.mu, np.zeros((self.iM,1))])
        i = 0
        for m in range(self.iR,self.iM,3):
            self.state.mu[m] = mx[i]
            self.state.mu[m+1] = my[i]
            self.state.mu[m+2] = i+1
            i+=1
        self.state.Sigma = np.pad(initial_state.Sigma,[(0,self.iM-self.iR),(0,self.iM-self.iR)], mode='constant', constant_values=0)
        self.R = np.zeros_like(self.state.Sigma)
        self.G = np.zeros_like(self.state.Sigma)

    def predict(self, u, dt=None):
        iR = self.iR  # Robot indexes
        iM = self.iM  # Map indexes
        mu_r = self.mu[:iR]

        G_x = self.get_g_prime_wrt_state(mu_r, u)
        V_x = self.get_g_prime_wrt_motion(mu_r, u)
        M_t = self.get_motion_noise_covariance(u)

        R_t = V_x @ M_t @ V_x.T
        self.R[:R_t.shape[0], :R_t.shape[1]] = R_t
        self.G[:G_x.shape[0], :G_x.shape[1]] = G_x

        # EKF prediction of the state mean.
        self.state_bar.mu = get_prediction(mu_r, u)[np.newaxis].T

        # EKF prediction of the state covariance.
        self.state_bar.Sigma[:iR,:iR] = G_x @ self.Sigma_bar[:iR, :iR] @ G_x.T + R_t
        if iM > 0:
            self.state.Sigma[:iR, iR:iM] = G_x @ self.Sigma[:iR, iR:iM]
            self.state.Sigma[iR:iM, :iR] = self.state.Sigma[iR:iM, :iR] @ G_x.T
        Sigma = self.state.Sigma
        return mu_r, Sigma

    def update(self, z):
        # self.mu_bar[2] = wrap_angle(self.mu_bar[2])
        for lm_id in [int(z[1])]:
            z_expected = get_expected_observation(self.mu_bar, lm_id)
            mx = self._field_map.landmarks_poses_x[lm_id]
            my = self._field_map.landmarks_poses_y[lm_id]
            q = (mx-self.mu[0])**2 + (my-self.mu[1])**2
            H = np.array([(my-mu[1])/q, -(mx-mu[0])/q, -1])
            Q = np.diag(self.params.beta)
            S = np.dot(np.dot(H, self.Sigma_bar), H.T) + Q
            K = np.dot( np.dot(self.Sigma_bar, H.T), np.linalg.inv(S) )
            
            self.state_bar.mu = self.state_bar.mu + np.dot(K, z[np.newaxis].T-z_expected[np.newaxis].T)
            self.state_bar.Sigma = np.dot( np.eye(3) - np.dot(K,H), self.Sigma_bar )
        
        self.state.mu = self._state_bar.mu
        self.state.Sigma = self._state_bar.Sigma
        return self.mu

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
    @property
    def mu_bar(self):
        """
        :return: The state mean after the update step.
        """
        # return self.state_bar.mu
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
        # return self.state.mu
        return self.state.mu.T[0]

    @property
    def Sigma(self):
        """
        :return: The state covariance after the update step (shape: 3x3).
        """
        return self.state.Sigma

