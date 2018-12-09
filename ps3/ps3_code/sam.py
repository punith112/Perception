import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

dx = 3; dz = 2; dm = 2;
G = np.random.rand(dx,dx)
H = np.random.rand(dz,dx)
J = np.random.rand(dz,dm)

def adjacency_matrix(G,H,J, dM=2, visualize=False):
	K = 16 # number of measurements Z
	M = 8 # number of landmarks M
	N = M # number of poses X
	batches = 2 # number of measurements from 1 pose
	dx = G.shape[0]; dz = H.shape[0]; dm = J.shape[1]
	A = np.zeros((N*dx+K*dz,N*dx+M*dm))
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
		plt.figure(1)
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

def back_substitution(A, b):
	# solving Rx = b
    n = b.size
    x = np.zeros_like(b)

    for i in range(n-1, 0, -1):
        x[i] = A[i, i]/b[i]
        for j in range (i-1, 0, -1):
            A[i, i] += A[j, i]*x[i]
    return x

def QR_factorization(A, b):
	Q, R_ = la.qr(A,mode='full')
	b_new = Q.T @ b
	R = R_[:A.shape[0],:]
	d = b_new[:A.shape[1]]
	delta = back_substitution(R,d)
	return delta

A = adjacency_matrix(G, H, J, visualize=False)

# b = np.random.rand(A.shape[0])[np.newaxis].T
b = np.ones(A.shape[0])[np.newaxis].T
delta = QR_factorization(A, b)
