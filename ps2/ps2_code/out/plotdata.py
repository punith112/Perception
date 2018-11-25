import matplotlib.pyplot as plt
import numpy as np
from math import *

data_input = np.load('input_data.npy')
x_in = data_input['real_robot_path'][:,0]
y_in = data_input['real_robot_path'][:,1]
theta_in = data_input['real_robot_path'][:,2]

def plotoutput(file_name):
	data_out = np.load(file_name)
	x = data_out['mean_trajectory'][:,0]
	y = data_out['mean_trajectory'][:,1]
	theta = data_out['mean_trajectory'][:,2]

	cov = data_out['covariance_trajectory']
	x_sigma = cov[0,0,:]
	y_sigma = cov[1,1,:]
	theta_sigma = cov[2,2,:]
	print(cov)

	if 'ekf' in file_name:
		title = 'EKF'
	elif 'ukf' in file_name:
		title = 'UKF'
	plt.figure()
	plt.plot(x_in,y_in)
	plt.plot(x,y)
	plt.legend(['real_robot_path', 'filtered_path'])
	plt.title(title)

	plt.figure()
	plt.plot(x-x_in)
	plt.plot(y-y_in)
	#plt.plot(x_sigma)
	#plt.plot(-x_sigma)
	plt.legend(['x-x_real, m', 'y-y_real, m'])
	plt.title(title)

	plt.figure()
	plt.plot(y-y_in)
	#plt.plot(y_sigma)
	#plt.plot(-y_sigma)
	#plt.legend(['y-y_real, m', '3S', '-3S'])
	plt.title(title + ': y-y_real, m')

	plt.figure()
	plt.plot(theta-theta_in)
	#plt.plot(theta_sigma)
	#plt.plot(-theta_sigma)
	plt.title(title+': theta-theta_real')
	#plt.legend(['theta-theta_real', '3S', '-3S'])
	plt.show()


file_name = 'output_ukf.npy'
plotoutput(file_name)