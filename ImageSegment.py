# R/W Images
from scipy.misc import imread

# Processing
import numpy
import math

# For kmeans
from sklearn.cluster import KMeans

# Graphs
import matplotlib.pyplot as plt

num_segments = 20

def get_array_from_img(filename):
	img = imread(filename)

	# Convert to a 2-dimensional array
	img_arr = numpy.zeros((img.shape[0]*img.shape[1], img.shape[2]))

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			img_arr[i*img.shape[1] + j][0] = img[i][j][0]	# Red
			img_arr[i*img.shape[1] + j][1] = img[i][j][1]	# Green
			img_arr[i*img.shape[1] + j][2] = img[i][j][2]	# Blue

	# Scale to 0-1 range
	img_arr = img_arr / 255

	return img_arr

def get_initial_model_parameters(x):
	
	#mu = numpy.zeros(())
	#pi = 

	pass
	#return mu, pi	

def main():
	img = get_array_from_img("Sunset.jpg")
	print img[0]
	#mu, pi = get_initial_model_parameters(img)

if __name__ == '__main__':
	main()