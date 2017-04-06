import numpy
import pandas
import math

# For kmeans
from sklearn.cluster import KMeans

num_documents = 1500
num_words = 12419

# Number of clusters
num_topics = 30

def get_model_parameters(docs):
	p = numpy.array([numpy.array([float(0) for j in range(num_words+1)]) for i in range(num_topics)])
	pi = numpy.array([float(0) for i in range(num_topics)])

	kmeans_p = KMeans(n_clusters=num_topics).fit(docs)
	labels =  kmeans_p.labels_

	# Normalize to make elements probabilities
	for k in labels:
		pi[k] = pi[k] + 1
	for k in range(len(pi)):
		pi[k] = float(pi[k])/len(labels)
	
	# Add the vectors for each topic, and avoid 0's
	for x in range(len(labels)):
		p[labels[x]] = p[labels[x]] + docs[x]
	
	p [p == 0] = 1
	assert (p.any() == 0) == False

	# Normalize p to make each sample a probability
	for i in range(len(p)):
		s = float(sum(p[i]))
		for j in range(len(p[i])):
			p[i][j] = float(p[i][j])/s
			assert p[i][j] != 0

	return p, pi

def e_step(p, pi, docs):
	# Compute z vectors
	z = numpy.dot(docs, numpy.transpose(p))

	for i in range(len(z)):
		for j in range(len(z[0])):
			z[i][j] = z[i][j] + numpy.log(pi[j])

	w = numpy.copy(z)

	# Compute w matrix
	for i in range(len(w)):
		m = max(w[i])
			
		w[i] = numpy.power(w[i] - m, math.e)
		w[i] = w[i]/float(sum(w[i]))

	# Compute the Q value
	print numpy.sum(numpy.multiply(z, w))

	return p, pi, w

def m_step(p, pi, w, docs):
	# Recompute p vectors

	for i in range(num_documents+1)

def main():
	# Read in the vociabulary
	vocab = pandas.read_csv("vocab.nips.txt", header = None)

	# Read in the bag of words representation
	words = pandas.read_csv("docword.nips.txt", skiprows = [0, 1, 2], sep = ' ', dtype = numpy.int32).as_matrix()

	# Store each document as a vector of words, these are our data points
	docs = numpy.array([(numpy.array([0 for y in range(num_words+1)])) for x in range(num_documents+1)])
	for point in words:
		docs[int(point[0])][int(point[1])] = float(point[2])

	# Estimate Model Parameters

	# p[i][j] represents the probability that word j is in topic i
	# pi[i] represent the probability a given document is in topic i
	p, pi = get_model_parameters(docs)
	
	# Compute Labels (E step)
	p, pi, w = e_step(p, pi, docs)

	m_step(p, pi, w, docs)

	# Recompute parameters (M Step)
	
	
if __name__ == '__main__':
	main()