import numpy as np
from math import *

from field_map import *

def measurement_jacobian(lm_pose, mu):
    mx = lm_pose[0]
    my = lm_pose[1]
    q = (mx-mu[0])**2 + (my-mu[0])**2

    H = np.array([[-(mx-mu[0])/sqrt(q), -(my-mu[1])/sqrt(q), 0],
                  [ (my-mu[1])/q,       -(mx-mu[0])/q,       1]])
    return H

fm = FieldMap()
mx_list = fm._landmark_poses_x
my_list = fm._landmark_poses_y

mu0 = [180,50]
for m in range(len(mx_list)):
	print(measurement_jacobian([mx_list[m], my_list[m]], mu0))
	print('\n')