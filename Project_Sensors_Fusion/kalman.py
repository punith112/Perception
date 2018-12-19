import numpy as np
from math import *

import matplotlib.pyplot as plt


def transition_matrix(T):
	"""
	input: T - timestep
	output: G - 6x6 transition matrix, propagating position and velocity
			accoring to the motion model:
			x_i = x_i-1 + v_i*T + ui*T^2 / 2
			v_i = v_i-1 + ui* T
			ui  = a_i + eps_i
			for each of 3 coordinates x = [x, y, z], v = [vx, vy, vz],
			where a_i is random acceleration
	"""
	G = np.eye(6);
	G[0,1] = T;
	G[2,3] = T;
	G[4,5] = T;
	return G # 6x6

def odometry_covariance_matrix(T, sigmaA):
	"""
	input: T - timestep, sigmaA - acceleration a_i std deviation
	F describes random acceleration terms in motion model
	"""
	F = np.array([[T**2/2, 0, 0], [T, 0, 0], [0, T**2/2, 0], [0, T, 0], [0, 0, T**2/2], [0, 0, T]])
	R = F @ F.T * (sigmaA**2)
	return R # 6x6 - odometry noise matrix

def observation_matrix(T):
	"""
	input: T - timestep
	output: H - 3x6 measurement matrix, choosing zx, zy, zz from Xpr
	"""
	H = np.zeros((3,6))
	H[0,0] = 1
	H[1,2] = 1
	H[2,4] = 1
	return H # 3x6


def kalman_filter(X0, P0, z, T):
	"""
	input: X0 - initial state
		   z - measurements vector (noisy trajectory)
		   T - timestep
	output: Xfl - filtered trajectory
			Pfl - array of covariences on each timestep
	"""
	N = z.shape[1]
	Xpr = np.zeros((6,N))
	Xpr[:,0] = X0
	Xfl = np.zeros((6,N))
	Xfl[:,0] = X0

	Ppr = [P0]
	Pfl = [P0]

	K = [np.zeros((6,3))]

	G = transition_matrix(T)
	R = odometry_covariance_matrix(T, sigmaA=0.4)
	H = observation_matrix(T)
	sigmaN = 0.1
	Q = np.diag(sigmaN**2 * np.ones(3))
	# print('Q=\n', Q)

	for i in range(1,N):
	    # prediction
	    Xpr[:,i] = G @ Xfl[:,i-1]
	    Ppr.append(G @ Pfl[i-1] @ (G.T) + R)

	    # correction
	    K.append( Ppr[i] @ H.T @ np.linalg.inv(H @ Ppr[i] @ H.T + Q) )
	    Xfl[:,i] = Xpr[:,i] + K[i] @ (z[:,i] - H @ Xpr[:,i])
	    Pfl.append( (np.eye(6) - K[i] @ H) @ Ppr[i] )

	    # Xfl = Xpr; Pfl = Ppr

	return np.array(Xfl), np.array(Pfl)


visualize = 1

T = 0.05
# G = transition_matrix(T)
# R = odometry_covariance_matrix(T, sigmaA=4)
# H = observation_matrix(T)

# print('G = \n', G)
# print('R = \n', R)
# print('H = \n', H)

N = 200
t = T * np.linspace(1, N, N)
x = np.cos(t); y = np.sin(t); z = 2*np.ones(N)
sigma_noise = 0.3
zx = x + sigma_noise*np.random.randn(N)
zy = y + sigma_noise*np.random.randn(N)
zz = z + sigma_noise*np.random.randn(N)

Z = np.vstack((zx,zy,zz))
X0 = np.array([1,0.1, 0,0.1, 2,0]) # x,vx, y,vy, z,nz
P0 = 0.1 * np.eye(6)
Xfl, Pfl = kalman_filter(X0, P0, Z, T)

xfl = Xfl[0,:]
yfl = Xfl[2,:]
zfl = Xfl[4,:]


if visualize:
	# plt.figure()
	# plt.plot(Pfl[:,0,0])
	# plt.title('Sigma_X')

	# plt.figure()
	# plt.plot(Pfl[:,2,2])
	# plt.title('Sigma_Y')

	# plt.figure()
	# plt.plot(Pfl[:,4,4])
	# plt.title('Sigma_Z')

	# plt.figure()
	# plt.plot(zx, '.')
	# plt.plot(xfl)
	# plt.plot(x)
	# plt.title('X')
	# plt.legend(['measurements', 'filtered', 'real'])

	# plt.figure()
	# plt.plot(zy, '.')
	# plt.plot(yfl)
	# plt.plot(y)
	# plt.title('Y')
	# plt.legend(['measurements', 'filtered', 'real'])

	plt.figure()
	plt.plot(zz, '.')
	plt.plot(zfl, 'ro')
	plt.plot(z)
	plt.title('Z')
	plt.legend(['measurements', 'filtered', 'real'])

	plt.figure()
	plt.plot(zx, zy, '.')
	plt.plot(xfl, yfl, 'ro')
	plt.plot(x, y)
	plt.title('XY-plane')
	plt.legend(['measurements', 'filtered', 'real'])

	plt.show()