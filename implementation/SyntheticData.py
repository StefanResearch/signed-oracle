import numpy

import Graph

'''
	Generates a signed SBM with n vertices and 2*k equally sized bi-clusters.
	Intra-cluster (+,+)-edges are inserted with probability pIntra and
	intra-cluster (+,-)-edges are inserted with probability pCross. The edges
	have the ``correct'' sign with probability pSign. Inter-cluster edges are
	inserted with probability q and have positive sign with probability
	posInter.

	Returns the graph and the list of planted clusters.
'''
def signedSBM(n, k, pIntra, pCross, q, pSign, posInter):
	print('Creating a random graph with the following parameters:')
	print(f'\tn: {n}, k: {k}')
	print(f'\tpIntra: {pIntra} , pCross: {pCross}, q: {q}')
	print(f'\tpSign: {pSign}, posInter: {posInter}\n')

	G = Graph.Graph()
	
	biclusters = partitionVertices(n,k)
	' add edges inside the biclusters '
	for bicluster in biclusters:
		for i in range(len(bicluster)):
			u = bicluster[i]
			for j in range(i+1,len(bicluster)):
				v = bicluster[j]

				randomlyAddEdge(G,u,v,pIntra,pSign)

	' add edges between the biclusters '
	for i in range(k):
		for u in biclusters[2*i]:
			for v in biclusters[2*i+1]:
				randomlyAddEdge(G,u,v,pCross,1-pSign)

	' add noise edges '
	for i in range(2*k):
		for j in range(i+1,2*k):
			' skip if bicluster1 and bicluster2 form a bicluster '
			if i%2 == 0 and j==i+1:
				continue

			bicluster1 = biclusters[i]
			bicluster2 = biclusters[j]

			for u in bicluster1:
				for v in bicluster2:
					randomlyAddEdge(G,u,v,q,posInter)

	' create the clustering '
	clusters = []
	for i in range(k):
		cluster = [u for u in biclusters[2*i]] + [u for u in biclusters[2*i+1]]
		clusters.append(cluster)

	return G, clusters, biclusters

'''
	Returns a random partitioning of n vertices into 2*k equally sized
	bi-clusters (the last cluster might contain a few more elements due to
	rounding issues). The returned object is a list of lists. A bi-cluster is
	formed by the elements at positions 2*i and 2*i+1.
'''
def partitionVertices(n, k):
	clusterSize = int(numpy.floor( n/(2*k) ))
	
	permutation = numpy.arange(n)

	clusters = []
	for i in range(2*k-1):
		cluster = permutation[(i*clusterSize):((i+1)*clusterSize)].tolist()
		cluster = [c+1 for c in cluster] # make sure that indices start at 1
		clusters.append(cluster)
	
	' the final cluster contains the rest of the vertices and might have more than clusterSize elements '
	cluster = permutation[ ((2*k-1)*clusterSize) : n ].tolist()
	cluster = [c+1 for c in cluster] # make sure that indices start at 1
	clusters.append(cluster)

#	print(f'ground truth: {clusters}')
	return clusters

'''
	Adds edge (u,v) to graph G with probability probExists. The probability that
	the edge has positive sign (if it exists) is probPositive.
'''
def randomlyAddEdge(G, u, v, probExists, probPositive):
	exists = numpy.random.binomial(1,probExists)

	if exists==1:
		positive = numpy.random.binomial(1,probPositive)
		sign = 2*positive - 1
		G.addEdge(u,v,sign)

