# R/W Images
from scipy.misc import imread, imsave

# Processing
import numpy
import math

import timeit

# For kmeans
from sklearn.cluster import KMeans

# Graphs
import matplotlib.pyplot as plt

num_segments = 50
img_len = 0
img_width = 0

def get_img_from_arr(img):
	my_img = numpy.zeros((img_len, img_width, 3))

	for i in range(img_len):
		for j in range(img_width):
			my_img[i][j][0] = img[i*img_width + j][0]
			my_img[i][j][1] = img[i*img_width + j][1]
			my_img[i][j][2] = img[i*img_width + j][2]

	return my_img

def get_array_from_img(filename):
	global img_len, img_width

	img = imread(filename)

	img_len = img.shape[0]
	img_width = img.shape[1]

	# Convert to a 2-dimensional array
	img_arr = numpy.zeros((img.shape[0]*img.shape[1], img.shape[2]))

	# Row major order
	for i in range(img_len):
		for j in range(img_width):
			img_arr[i*img_width + j][0] = img[i][j][0]	# Red
			img_arr[i*img_width + j][1] = img[i][j][1]	# Green
			img_arr[i*img_width + j][2] = img[i][j][2]	# Blue

	return img_arr

def get_initial_model_parameters(x):
	
	mu = numpy.zeros((num_segments, 3))
	pi = numpy.zeros((num_segments))

	kmeans_mu = KMeans(n_clusters=num_segments).fit(x)
	labels = kmeans_mu.labels_

	# Initial Probabilites for each segment
	for i in range(len(labels)):
		pi[labels[i]] += 1.0

	# Pixel Probabilites
	for i in range(len(labels)):
		mu[labels[i]] += x[i]
	for i in range(len(labels)):
		mu[labels[i]] /= pi[labels[i]]

	pi = pi/len(labels)

	# Normalize to ensure no zeroes
	mu[mu == 0] = 1

	for i in range(len(mu)):
		s = sum(mu[i])
		mu[i] /= s

	assert pi.any() != 0
	assert mu.any() != 0

	return mu, pi

def e_step_2(mu, pi, x):
	mu [mu == 0] = 0.0001

	# Compute z vectors
	z = numpy.dot(x, numpy.transpose(numpy.log(mu)))

	for i in range(len(z)):
		for j in range(len(z[0])):
			z[i][j] = z[i][j] + numpy.log(pi[j])

	w = numpy.copy(z)

	# Compute w matrix
	for i in range(len(w)):
		w[i] = numpy.exp(w[i] - max(w[i]))

		assert (w[i].any() == 1)

		w[i] = w[i]/float(sum(w[i]))

	# Compute the Q value
	q = numpy.sum(numpy.multiply(z, w))

	return mu, pi, w, q

def e_step(mu, pi, x):
	num_pixels = img_len * img_width
	num_clusters = num_segments
	result = []
	for i in range(num_pixels):
		temp = []
		for j in range(num_clusters):
			temp.append( -0.5 * (numpy.dot(x[i] - mu[j], x[i] - mu[j])) + numpy.log(pi[j]) )
		result.append(temp)

	z_matrix = numpy.array(result)
	w = numpy.copy(z_matrix)

	for i in range(num_pixels):
		w[i] = numpy.exp(w[i] - numpy.max(w[i]))
		w[i] = w[i] / float(numpy.sum(w[i]))

	q = numpy.sum(numpy.multiply(z_matrix, w))

	return mu, pi, w, q

def m_step(mu, pi, w, x):

	x_sums = numpy.sum(x, axis = 1)

	w_t = numpy.transpose(w)

	mu = numpy.dot(w_t, x)

	fac = numpy.dot(w_t, x_sums)

	for j in range(len(mu)):
		mu[j] = mu[j]/fac[j]

	# Recompute pi vectors
	pi = numpy.sum(w, axis = 0) / len(x)

	return mu, pi

def get_closest_idx(x, sample):
	
	min_d = 10000
	min_idx = -1

	for i in range(len(sample)):
		d = numpy.linalg.norm(x - sample[i])

		if d < min_d:
			min_d = d
			min_idx = i

	assert min_idx != -1

	return min_idx

def replace_img(img, mu):
	
	for i in range(len(img)):
		min_idx = get_closest_idx(img[i], mu)

		img[i] = mu[min_idx]

	return img

def main():
	img = get_array_from_img("Sunset.jpg")

	mu, pi = get_initial_model_parameters(img)

	# Compute Labels (E step)
	mu, pi, w, q_old = e_step(mu, pi, img)

	# Recompute parameters (M Step)
	mu, pi = m_step(mu, pi, w, img)

	num_iterations = 0

	# Let it converge
	while True:
		num_iterations += 1
		print "\n-------------- X -----------"
		print "Iteration: ", num_iterations

		e_start = timeit.default_timer()
		mu, pi, w, q_new = e_step(mu, pi, img)
		e_stop = timeit.default_timer()
		print "Q: ", q_new
		print "E Step Time: ", (e_stop - e_start), " seconds"

		m_start = timeit.default_timer()
		mu, pi = m_step(mu, pi, w, img)
		m_stop = timeit.default_timer()
		print "M Step Time: ", (m_stop - m_start), " seconds"
		print "-------------- X -----------\n"

		# Check for convergence
		delta_q = abs(float(q_new) - q_old)#/q_old
		if  delta_q <= 1:
			break

		q_old = q_new

	# Replace pixels with closest cluster centers
	new_img = replace_img(img, mu)

	# Convert Image to 3-dimesions
	another_img = get_img_from_arr(new_img)

	imsave("Sunset_clustered.png", another_img)


if __name__ == '__main__':
	main()