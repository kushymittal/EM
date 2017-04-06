import numpy
import pandas
import math

# For kmeans
from sklearn.cluster import KMeans

# Metrics
from scipy.spatial import distance
import timeit

# Graphs
import matplotlib.pyplot as plt

num_documents = 1500
num_words = 12419

# Number of clusters
num_topics = 30

def get_initial_model_parameters(docs):
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
		w[i] = numpy.exp(w[i] - max(w[i]))

		assert (w[i].any() == 1)

		w[i] = w[i]/float(sum(w[i]))

	# Compute the Q value
	q = numpy.sum(numpy.multiply(z, w))

	return p, pi, w, q

def m_step(p, pi, w, docs):
	
	# Recompute p vectors
	for j in range(num_topics):
		
		#temp = numpy.copy(p[j])
		
		p[j][p[j] != 0] = 0
		assert (p[j].all() == 0)
		
		d = 0
		
		for i in range(num_documents+1):
			p[j] = p[j] + (w[i][j] * docs[i])
			d = d + sum(docs[i])*w[i][j]

		p[j] = p[j] / d

		#print "P difference: ", distance.euclidean(temp, p[j])

	# Recompute pi vectors
	pi = numpy.sum(w, axis = 0) / (num_documents + 1)

	return p, pi

def print_frequent_words_idx(p):
	# Read in the vociabulary
	vocab = pandas.read_csv("vocab.nips.txt", header = None)

	for j in range(num_topics):
		
		words = (-p[j]).argsort()[:10]

		print "\n-------------- X -----------"
		print "Topic: ", j

		for k in range(10):
			print vocab[0][words[k]], 

		print "\n-------------- X -----------\n"

	return
	
def topic_prob_graph(pi):
	topics = list(range(num_topics))

	plt.plot(topics, pi, "o", color="Teal")
	plt.xlabel("Topics")
	plt.ylabel("Probabilities")
	plt.show()

	return

def main():
	p_start = timeit.default_timer()

	# Read in the bag of words representation
	words = pandas.read_csv("docword.nips.txt", skiprows = [0, 1, 2], sep = ' ', dtype = numpy.int32).as_matrix()

	# Store each document as a vector of words, these are our data points
	docs = numpy.array([(numpy.array([float(0) for y in range(num_words+1)])) for x in range(num_documents+1)])
	for point in words:
		docs[int(point[0])][int(point[1])] = float(point[2])

	# Estimate Model Parameters

	# p[i][j] represents the probability that word j is in topic i
	# pi[i] represent the probability a given document is in topic i
	p, pi = get_initial_model_parameters(docs)
	
	# Compute Labels (E step)
	p, pi, w, q_old = e_step(p, pi, docs)

	# Recompute parameters (M Step)
	p, pi = m_step(p, pi, w, docs)
	
	num_iterations = 0

	print "\n-------------- X -----------"
	print "Iteration: ", num_iterations
	print "Q: ", q_old
	print "-------------- X -----------\n"

	while True:
		num_iterations += 1
		print "\n-------------- X -----------"
		print "Iteration: ", num_iterations

		e_start = timeit.default_timer()
		p, pi, w, q_new = e_step(p, pi, docs)
		e_stop = timeit.default_timer()
		print "Q: ", q_new
		print "E Step Time: ", (e_stop - e_start), " seconds"

		m_start = timeit.default_timer()
		p, pi = m_step(p, pi, w, docs)
		m_stop = timeit.default_timer()
		print "M Step Time: ", (m_stop - m_start), " seconds"
		print "-------------- X -----------\n"

		# Check for convergence
		delta_q = abs(float(q_new) - q_old)/q_old
		if  delta_q <= 0.01:
			break

		q_old = q_new

	# Print common words for each topic
	print_frequent_words_idx(p)

	# Make a graph for the prob for each topic
	topic_prob_graph(pi)

	p_stop = timeit.default_timer()
	print "Program Runtime: ", p_stop - p_start, " seconds \n\n"

	
if __name__ == '__main__':
	main()