import random
import numpy
import networkx as nx

import time

class Graph:
	'''
		The graph is stored as a dict over unordered lists. Because of how the
		graph is stored, the vertex ids start at 1 (and not at 0).

		For positive edges, the list contains the index of the vertex, and for
		negative edges, the list contains the negative index of the vertex.
		For example, if vertex 102 has neighbors 103 with a positive edge and
		105 with a negative edge, then edges[102] = [103,-105].
	'''
	def __init__(self):
		self.edges = dict()

		self.numVertices = 0
		self.numEdges = 0
		self.numPositiveEdges = 0
		self.numNegativeEdges = 0

	'''
		u and v should be integers,
		sign should be +1 or -1.

		Does not ensure consistency
		(e.g., there could be multiple edges of different signs).
	'''
	def addEdge(self, u, v, sign):
		if u not in self.edges:
			self.addVertex(u)
		if v not in self.edges:
			self.addVertex(v)

		self.edges[u].append(sign*v)
		self.edges[v].append(sign*u)

		self.numEdges += 1
		if sign >= 0:
			self.numPositiveEdges += 1
		else:
			self.numNegativeEdges += 1

	def addVertex(self, u):
		self.edges[u] = []
		self.numVertices += 1

	'''
		Returns a uniformly random neighbor of vertex u
		together with the sign of the edge.
	'''
	def randomNeighbor(self, u):
		signedNeighbor = random.choice(self.edges[u])
		
		sign = 1 if signedNeighbor >= 0 else -1
		unsignedNeighbor = abs(signedNeighbor)

		return unsignedNeighbor, sign

	'''
		Returns beta_G(V1,V2), i.e., the signed bipartiteness ratio of the
		bicluster (V1,V2).
	'''
	def signedBipartitenessRatio(self, V1, V2):
		numerator = self.numNegativeEdgesInsideSubgraph(V1)
		numerator += self.numNegativeEdgesInsideSubgraph(V2)
		numerator += 2*self.numPositiveEdgesBetween(V1,V2)
		numerator += self.numEdgesLeavingSet(V1.union(V2))
		
		vol = self.volume(V1.union(V2))

		if vol == 0:
			return 0
		else:
			return numerator/vol

	'''
		Returns the number of negative edges inside G[S].
		Assumes that S is a set.
	'''
	def numNegativeEdgesInsideSubgraph(self, S):
		numNegativeEdges = 0
		for u in S:
			if u not in self.edges.keys():
				print(f'could not find {u}')
				continue

			for neighbor in self.edges[u]:
				if neighbor<0 and -neighbor in S:
					numNegativeEdges += 1

		return numNegativeEdges

	'''
		Returns the number of positive edges from V1 to V2.
		Assumes that V1 and V2 are both sets.
	'''
	def numPositiveEdgesBetween(self, V1, V2):
		numPositiveEdges = 0
		for u in V1:
			if u not in self.edges.keys():
				print(f'could not find {u}')
				continue

			for neighbor in self.edges[u]:
				if neighbor>0 and neighbor in V2:
					numPositiveEdges += 1

		return numPositiveEdges

	'''
		Returns the number of edges from G[S] to G[V\S].
	'''
	def numEdgesLeavingSet(self, S):
		numEdgesLeaving = 0
		for u in S:
			for neighbor in self.edges[u]:
				if neighbor not in S and -neighbor not in S:
					numEdgesLeaving += 1

		return numEdgesLeaving

	'''
		Returns the volume of a set of vertices S.
	'''
	def volume(self, S):
		vol = 0
		for s in S:
			vol += self.degree(s)

		return vol

	'''
		Returns the degree of vertex u.
	'''
	def degree(self, u):
		if u in self.edges.keys():
			return len(self.edges[u])
		else:
			return 0

	def getNetworkxGraph(self):
		' we will subtract 1 for all vertex ids to make sure that the graph is 0-indexed '
		G = nx.Graph()
		for vertex in self.edges.keys():
			G.add_node(vertex-1)
		
		for vertex,neighbors in self.edges.items():
			for neighbor in neighbors:
				sign = -1 if neighbor < 0 else 1
				G.add_edge(vertex-1, abs(neighbor)-1)
				G[vertex-1][abs(neighbor)-1]['sign'] = sign

		return G

	def writeToSparseFormatForMatlab(self, filename):
		with open(filename,'w') as f:
			for vertex,neighbors in self.edges.items():
				for neighbor in neighbors:
					sign = -1 if neighbor < 0 else 1
					f.write(f'{vertex}\t{abs(neighbor)}\t{sign}\r\n')

	def writeSparseMatrixToTmp(self):
		timestring = time.time()
		graphfile = f'/tmp/thegraph_{timestring}.txt'
		self.writeToSparseFormatForMatlab(graphfile)

		return graphfile

