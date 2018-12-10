import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from tools.objects import Gaussian
import sys


class SAM():
	def __init__(self, initial_state, number_of_lms):
		self.state = initial_state
		self.M = number_of_lms # number of all lms
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
		

	def get_delta(self, G, H, J):
		A = self.adjacency_matrix(G,H,J)
		b = self.b
		delta = self.QR_factorization(A, b)
		return delta

	def adjacency_matrix(self, G,H,J, dM=2, visualize=False):
		K = self.M*self.batches # number of measurements Z
		M = self.M # number of landmarks M
		N = M # number of poses X
		batches = self.batches # number of measurements from 1 pose
		dx = G.shape[0]; dz = H.shape[0]; dm = J.shape[1]
		A = self.A
		I = np.eye(dx)
		for i in range(N):
		    A[dx*i:dx*(i+1),dx*i:dx*(i+1)] = I
		for g in range(N-1):
		    A[dx*(g+1):(g+2)*dx, dx*g:dx*(g+1)] = G

		dM = dM
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

			# print('non zero elements in A = ',np.count_nonzero(A))
			# plt.figure(2)
			# Lambda = np.transpose(A) @ A # + np.eye(N+M)*0.001
			# plt.spy(Lambda, marker='o', markersize=5)
			# plt.title('Information matrix $\Lambda$')

			# plt.figure(3)
			# plt.spy(np.linalg.inv(Lambda), marker='o', markersize=5)
			# plt.title('Inverse of $\Lambda$: dense')
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

mean_prior = np.array([180., 50., 0.])
Sigma_prior = 1e-12 * np.eye(3, 3)
initial_state = Gaussian(mean_prior, Sigma_prior)

# sam object initialization
number_of_lms = int( sys.argv[1] )
sam = SAM(initial_state, number_of_lms)

# Adjacency matrix for random Jacobians
# M=N = 8, K = 8*2=16
sam.adjacency_matrix(sam.G,sam.H,sam.J, dM=2, visualize=True)
print('\ndelta_x=', sam.get_delta(sam.G,sam.H,sam.J)[:3].T[0])