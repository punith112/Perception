import numpy as np
from math import *

from field_map import *

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

fm = FieldMap()
mx_list = fm._landmark_poses_x
my_list = fm._landmark_poses_y

mu0 = [180,50]
for m in range(len(mx_list)):
	print(jacobian_H([mx_list[m], my_list[m]], mu0))
	print('\n')
