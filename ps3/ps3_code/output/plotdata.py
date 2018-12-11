import matplotlib.pyplot as plt
import numpy as np
from math import *

data_input = np.load('input_data.npy')
x_ideal = data_input['noise_free_robot_path'][:,0]
y_ideal = data_input['noise_free_robot_path'][:,1]
x_in = data_input['real_robot_path'][:,0]
y_in = data_input['real_robot_path'][:,1]
theta_in = data_input['real_robot_path'][:,2]

def plotoutput(file_name):
	data_out = np.load(file_name)
	x = data_out['mean_trajectory'][:200,0]
	y = data_out['mean_trajectory'][:200,1]
	theta = data_out['mean_trajectory'][:200,2]

	cov = data_out['covariance_trajectory']
	x_sigma = cov[:,0,0]**0.5
	y_sigma = cov[:,1,1]**0.5
	theta_sigma = cov[:,2,2]**0.5

	if 'ekf' in file_name:
		title = 'EKF_SLAM'

	plt.figure()
	plt.plot(x_in,y_in)
	plt.plot(x,y)
	plt.plot(x_ideal,y_ideal)
	plt.legend(['real_robot_path', 'filtered_path', 'ideal_path'])
	plt.title(title)
	plt.savefig(title+'_paths.png')

	plt.figure()
	plt.plot(x-x_in)
	plt.plot(3*x_sigma)
	plt.plot(-3*x_sigma)
	plt.legend(['x-x_real, m', '3S', '-3S'])
	plt.title(title + ': x-x_real, m')
	plt.savefig(title+'_x.png')

	plt.figure()
	plt.plot(y-y_in)
	plt.plot(3*y_sigma)
	plt.plot(-3*y_sigma)
	plt.legend(['y-y_real, m', '3S', '-3S'])
	plt.title(title + ': y-y_real, m')
	plt.savefig(title+'_y.png')

	plt.figure()
	plt.plot(theta-theta_in)
	plt.plot(3*theta_sigma)
	plt.plot(-3*theta_sigma)
	plt.title(title+': theta-theta_real')
	plt.legend(['theta-theta_real', '3S', '-3S'])
	plt.savefig(title+'_theta.png')
	
	plt.show()


file_name = 'output_ekf_slam.npy'
plotoutput(file_name)
