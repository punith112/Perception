import numpy as np
from math import *
import matplotlib.pyplot as plt
import scipy.linalg as la
from tools.objects import Gaussian
import sys
from tools.task import get_prediction, wrap_angle


class SAM():
	def __init__(self, initial_state, args):
		self.iR = 3
		self.iM = 0
		self.state = initial_state
		self.state_bar = self.state
		self.params = args
		self.R = np.zeros_like(self.state.Sigma)
		self.G = np.zeros_like(self.state.Sigma)
		self.N = 0
		self.M = 0 # number of all lms
		self.observed_lms = 2
		self.K = self.M*self.observed_lms # number of measurements Z
		M = self.M # number of landmarks M
		N = 0 # number of poses X
		K = self.K
		dx = len(initial_state.mu); dz = 2; dm = 2;
		self.A = np.zeros((N*dx+K*dz,N*dx+M*dm))
		self.b = np.random.rand(self.A.shape[0])[np.newaxis].T
		self.G = []
		self.H = []
		self.J = []
		self.lm_seq = [] # sequence of lms
		self.js = []
		self.x_traj = initial_state.mu
		self.lm_poses = np.array([])
		
	def predict(self, u, dt=None):
		iR = self.iR # Robot indexes
		iM = self.iM # Map indexes
		mu_r = self.mu[:iR]

		G_x = self.get_g_prime_wrt_state(mu_r, u)
		V_x = self.get_g_prime_wrt_motion(mu_r, u)
		M_t = self.get_motion_noise_covariance(u)

		R_t = V_x @ M_t @ V_x.T

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


	def update(self, u, z):
		for batch_i in range(z.shape[0]): # for all observed features
			self.N += 1
			self.M = len(self.lm_seq)
			self.K = self.N * z.shape[0]
			lm_id = z[batch_i,2]

			if (lm_id not in self.lm_seq): # lm never seen before
				self.initialize_new_landmark(lm_id, z, batch_i)

			# G, H, J calculation
			j = int(np.where(self.lm_seq==lm_id)[0] + 1) # number of ID in landmark array
			self.js.append(j)
			delta = self.mu_bar[self.iR+2*j-2 : self.iR+2*j] - self.mu_bar[:2]
			q = np.dot( delta, delta.T )
			z_expected = np.array([ sqrt(q), wrap_angle( atan2(delta[1],delta[0])-self.mu_bar[2] ) ])
			self.H.append( self.get_jacobian_Hx(q, delta) )
			self.J.append( self.get_jacobian_J(q, delta) )
			self.G.append( self.get_g_prime_wrt_state(self.mu[:self.iR], u) )
			# self.H = self.get_jacobian_Hx(q, delta)
			# self.J = self.get_jacobian_J(q, delta)
			# self.G = self.get_g_prime_wrt_state(self.mu[:self.iR], u)
			
			self.x_traj = np.vstack((self.x_traj, self.state.mu[:3]))

			A = self.adjacency_matrix(lm_id)
			# x0 = self.mu_bar # linearization point
			# a = x0 - get_prediction(self.mu, u)
			# c = z[batch_i,:2] - z_expected
			b = np.random.rand(A.shape[0]) # TODO: should consist of a-s and c-s
			delta = self.QR_factorization(A,b)

			# self.state_bar.mu[:3] = (self.mu_bar[:3] + delta[:3])[np.newaxis].T
			# print(len(self.x_traj),  len(delta[:-len(self.lm_seq)*2]))

		# self.state.mu = self.state_bar.mu

		self.mu_bar[2] = wrap_angle(self.mu_bar[2])
		self.mu[2] = wrap_angle(self.mu[2])
		self.state.mu = self.state_bar.mu
		self.state.Sigma = self.state_bar.Sigma

		return self.mu, self.Sigma


	def initialize_new_landmark(self, lm_id, z, batch_i):        
		self.lm_seq.append(lm_id)
		self.iM += 2

		r = z[batch_i,0]
		phi = wrap_angle(z[batch_i,1])
		theta = wrap_angle(self.mu_bar[2])
		ang = wrap_angle(phi+theta)

		mu_new = self.mu[:2] + np.array([r*cos(ang), r*sin(ang)]) # position of new landmark
		self.state.mu = np.append(self.mu, mu_new)[np.newaxis].T
		self.lm_poses = self.state.mu[3:]


	# def adjacency_matrix(self, G,H,J, number_of_lms, visualize=False):
	def adjacency_matrix(self, lm_id):
		j = np.where(self.lm_seq==lm_id)[0][0]
		js0 = np.array(self.js)-self.js[0]
		self.M = int( max(self.lm_seq) ); M = self.M # number of landmarks M
		N = self.N
		K = self.N * self.observed_lms # number of measurements Z
		observed_lms = self.observed_lms # number of measurements from 1 pose
		G = self.G[-1]; H = self.H[-1]; J = self.J[-1]
		dx = G.shape[0]; dz = H.shape[0]; dm = J.shape[0]
		A = np.zeros((N*dx+K*dz,N*dx+M*dm))
		I = np.eye(dx)
		for i in range(N):
		    A[dx*i:dx*(i+1),dx*i:dx*(i+1)] = I
		for g in range(N-1):
		    A[dx*(g+1):(g+2)*dx, dx*g:dx*(g+1)] = self.G[g]

		dM = 2
		for h in range(N):
		    for batch in range(observed_lms):
		        hr = dx*N+(batch+observed_lms*h)*dz
		        hc = dx*h
		        A[hr:hr+dz, hc:hc+dx] = self.H[h-1]
		        jr = hr
		        jc = dx*N+ ( dm*js0[h] %(M*dm) )
		        A[jr:jr+dz, jc:jc+dm] = self.J[h-1]
		self.A = A
		return A

	# def adjacency_matrix(self):
	# 	dx = 3; dz = 2; dm = 2
	# 	A00 = - np.eye(self.N * dx);     A01 = np.zeros((self.N *dx,self.M*dm))
	# 	A10 = np.zeros((self.K*dz, self.N*dx)); A11 = np.zeros((self.K*dz,self.M*dm))

	# 	if A00.shape[0]>3:
	# 		A00[-3-self.G.shape[0]:-3 ,-3-self.G.shape[1]:-3] = self.G
	# 	# print(A00[-3-self.G.shape[0]:-3 ,-3-self.G.shape[1]:-3].shape)

	# 	A = np.block([[A00, A01],
	# 	              [A10, A11]])

	# 	self.A = A
	# 	return A


	@staticmethod
	def back_substitution(R, b):
		# solving Rx = b
	    n = b.size
	    x = np.zeros_like(b)

	    for i in range(n-1, 0, -1):
	        x[i] = R[i, i]/b[i]
	        for j in range (i-1, 0, -1):
	            R[i, i] += R[j, i]*x[i]
	    return x

	def QR_factorization(self, A, b):
		Q, R_ = la.qr(A,mode='full')
		b_new = Q.T @ b
		R = R_[:A.shape[0],:]
		d = b_new[:A.shape[1]]
		# R delta = d
		delta = self.back_substitution(R,d)
		return delta




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

	@staticmethod
	def get_jacobian_Hx(q, delta):
		return (1/q) * np.array([ [-sqrt(q)*delta[0], -sqrt(q)*delta[1], 0],
		                          [delta[1],          -delta[0],        -q] ])

	@staticmethod
	def get_jacobian_J(q, delta):
		return (1/q) * np.array([ [sqrt(q)*delta[0],  sqrt(q)*delta[1]],
		                          [-delta[1],         delta[0]       ]])

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


# from run import get_cli_args
# args = get_cli_args()

# mean_prior = np.array([180., 50., 0.])
# Sigma_prior = 1e-12 * np.eye(3, 3)
# initial_state = Gaussian(mean_prior, Sigma_prior)

# # sam object initialization
# number_of_lms = 16
# sam = SAM(initial_state, args)

# # Adjacency matrix for random Jacobians
# # M=N = 8, K = 8*2=16
# A = sam.adjacency_matrix(sam.G,sam.H,sam.J, number_of_lms, visualize=True)
# b = np.random.rand(A.shape[0])
# print('\ndelta_x=', sam.QR_factorization(A,b)[:3])