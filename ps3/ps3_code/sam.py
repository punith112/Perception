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
		self.M = 0 # number of all lms
		self.batches = 2
		self.K = self.M*self.batches # number of measurements Z
		M = self.M # number of landmarks M
		N = M # number of poses X
		K = self.K
		dx = len(initial_state.mu); dz = 2; dm = 2;
		self.A = np.zeros((N*dx+K*dz,N*dx+M*dm))
		self.b = np.random.rand(self.A.shape[0])[np.newaxis].T
		self.G = np.random.rand(dx,dx)
		self.H = np.random.rand(dz,dx)
		self.J = np.random.rand(dz,dm)
		# self.G = np.zeros((dx,dx))
		# self.H = np.zeros((dz,dx))
		# self.J = np.zeros((dz,dm))
		self.lm_seq = [] # sequence of lms
		
	def predict(self, u, dt=None):
		iR = self.iR # Robot indexes
		iM = self.iM # Map indexes
		mu_r = self.mu[:iR]

		G_x = self.get_g_prime_wrt_state(mu_r, u)
		V_x = self.get_g_prime_wrt_motion(mu_r, u)
		M_t = self.get_motion_noise_covariance(u)
		self.G = G_x

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
		for lm_id in z[:,2]: # for all observed features
			batch_i = np.where(z==lm_id)[0][0] # 0th or 1st of observed lms in the batch

			r = z[batch_i,0]
			phi = wrap_angle(z[batch_i,1])
			theta =  wrap_angle(self.mu_bar[2])
			self.m_new = self.mu_bar[:2] + np.array([r*cos((phi+theta)), r*sin((phi+theta))])

			delta = np.array(self.m_new - self.mu_bar[:2])
			q = np.dot( delta, delta.T )
			self.H = self.get_jacobian_Hx(q, delta)
			self.J = self.get_jacobian_J(q, delta)
			if (lm_id not in self.lm_seq): # lm never seen before
				self.initialize_new_landmark(lm_id, z, batch_i)
			number_of_lms = len(self.lm_seq)
			A = self.adjacency_matrix(self.G, self.H, self.J, number_of_lms)
			x0 = self.mu_bar

		self.mu_bar[2] = wrap_angle(self.mu_bar[2])
		self.mu[2] = wrap_angle(self.mu[2])
		self.state.mu = self.state_bar.mu
		self.state.Sigma = self.state_bar.Sigma

		return self.mu, self.Sigma


	def initialize_new_landmark(self, lm_id, z, batch_i):        
		self.lm_seq.append(lm_id)

	def adjacency_matrix(self, G,H,J, number_of_lms, visualize=False):
		self.M = number_of_lms
		K = self.M*self.batches # number of measurements Z
		M = self.M # number of landmarks M
		N = M # number of poses X
		batches = self.batches # number of measurements from 1 pose
		dx = G.shape[0]; dz = H.shape[0]; dm = J.shape[1]
		self.A = np.zeros((N*dx+K*dz,N*dx+M*dm))
		A = self.A
		I = np.eye(dx)
		for i in range(N):
		    A[dx*i:dx*(i+1),dx*i:dx*(i+1)] = I
		for g in range(N-1):
		    A[dx*(g+1):(g+2)*dx, dx*g:dx*(g+1)] = G

		dM = 2
		for h in range(N):
		    for batch in range(batches):
		        hr = dx*N+(batch+batches*h)*dz
		        hc = dx*h
		        A[hr:hr+dz, hc:hc+dx] = H
		        jr = hr
		        jc = dx*N+dm*(h+batch+dM)%(M*dm)
		        A[jr:jr+dz, jc:jc+dm] = J

		if visualize:
			plt.figure(1, figsize=(10,10))
			plt.spy(A, marker='o', markersize=5)
			plt.title('Adjacency matrix $A$')

			print('non zero elements in A = ',np.count_nonzero(A))
			plt.figure(2)
			Lambda = np.transpose(A) @ A # + np.eye(N+M)*0.001
			plt.spy(Lambda, marker='o', markersize=5)
			plt.title('Information matrix $\Lambda$')

			plt.figure(3)
			plt.spy(np.linalg.inv(Lambda), marker='o', markersize=5)
			plt.title('Inverse of $\Lambda$: dense')
			plt.show()
		return A

	@staticmethod
	def back_substitution(A, b):
		# solving Rx = b
	    n = b.size
	    x = np.zeros_like(b)

	    for i in range(n-1, 0, -1):
	        x[i] = A[i, i]/b[i]
	        for j in range (i-1, 0, -1):
	            A[i, i] += A[j, i]*x[i]
	    return x

	def QR_factorization(self, A, b):
		Q, R_ = la.qr(A,mode='full')
		b_new = Q.T @ b
		R = R_[:A.shape[0],:]
		d = b_new[:A.shape[1]]
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


# mean_prior = np.array([180., 50., 0.])
# Sigma_prior = 1e-12 * np.eye(3, 3)
# initial_state = Gaussian(mean_prior, Sigma_prior)

# # sam object initialization
# number_of_lms = int( sys.argv[1] )
# sam = SAM(initial_state)

# # Adjacency matrix for random Jacobians
# # M=N = 8, K = 8*2=16
# A = sam.adjacency_matrix(sam.G,sam.H,sam.J, number_of_lms, visualize=False)
# b = np.random.rand(A.shape[0])
# print('\ndelta_x=', sam.QR_factorization(A,b)[:3])