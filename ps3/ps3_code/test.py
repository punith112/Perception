import numpy as np
from math import *
from tools.task import wrap_angle
import sys


def get_jacobian_L(r, phi, theta):
    # L = dh^(-1) / dy, y = state
    return np.array([[1, 0, -r*sin(wrap_angle(phi+theta))],
                     [0, 1, r*cos(wrap_angle(phi+theta)) ]])

def get_jacobian_W(r, phi, theta):
    # W = dh^(-1) / dz, z = [r, phi] - lm measurement wrt robot's pose
    return np.array([[cos(wrap_angle(phi+theta)), -r*sin(wrap_angle(phi+theta))],
                     [sin(wrap_angle(phi+theta)), r*cos(wrap_angle(phi+theta))]])



def expandSigma(Sigma, L, W, Q):
    iR = 3 #self.iR
    iM = Sigma.shape[0]-3 #self.iM
    Sx = Sigma[:iR,:iR];        Sxm = Sigma[:iR, iR:iR+iM]
    Smx = Sigma[iR:iR+iM, :iR]; Sm = Sigma[iR:iR+iM, iR:iR+iM]
    # new Sigma sub-matrices
    Sur = Sx @ L.T; Sr = Smx @ L.T; Sbr = L@Sx@L.T + W@Q@W.T
    Sbl = L @ Sx; Sb = L @ Sxm

    Sigma = np.block([[Sx, Sxm, Sur],
                     [Smx, Sm, Sr],
                     [Sbl, Sb, Sbr]])
    return Sigma

Q  = np.diag([100, 100])
L = get_jacobian_L(1, 0, pi/4)
W = get_jacobian_W(1, 0, pi/4)


input_size = int( sys.argv[1] )
Sigma = np.eye(input_size)

print(expandSigma(Sigma, L, W, Q).shape)