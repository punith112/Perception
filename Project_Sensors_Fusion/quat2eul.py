from tf.transformations import *
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

filename = sys.argv[1]

data = pd.read_csv(filename)
roll = np.zeros_like(data.x); pitch = np.zeros_like(data.x); yaw = np.zeros_like(data.x)
for i in range(len(data.x)):
	q = np.array([data.x[i], data.y[i], data.z[i], data.w[i]])
	roll[i], pitch[i], yaw[i] = euler_from_quaternion(q)

plt.plot(roll)
plt.plot(pitch)
plt.plot(yaw)
plt.legend(['roll', 'pitch', 'yaw'])
plt.show()

# example launch
# python quat2eul.py ~/Desktop/Perception/Project_Sensors_Fusion/2D_trajectory/6th/2018-12-18-16-03-11/_slash_vicon_slash_obstacle1_slash_obstacle1.csv