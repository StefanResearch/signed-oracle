import numpy

import Graph

'''
	Reads a sparse csv file and returns a Graph object.

	Assumes that the csv-file has the following format:
	u,v,edgeWeight
	where u and v are integers and edgeWeight is a float
'''
def graphFromSparseCSV(inputfile, separator=',', skipHeader=False, inputIsZeroIndexed=False):
	'''
		since the input file might contain some edges multiple times with
		different weights, we start by summing over all of these weights
	'''
	neighbors = dict()

	with open(inputfile) as fp:
		line = fp.readline()
		if skipHeader:
			line = fp.readline()

		while line:
			lineSplit = str.split(line, separator)

			u = int(lineSplit[0])
			v = int(lineSplit[1])
			edgeWeight = float(lineSplit[2]) if len(lineSplit) > 2 else 1

			if inputIsZeroIndexed:
				u += 1
				v += 1

			if u not in neighbors:
				neighbors[u] = dict()
			if v not in neighbors:
				neighbors[v] = dict()

			if v not in neighbors[u]:
				neighbors[u][v] = 0
			if u not in neighbors[v]:
				neighbors[v][u] = 0

			neighbors[u][v] += edgeWeight
			neighbors[v][u] += edgeWeight

			line = fp.readline()

	' Create the signed graph by taking the sign of the edge weights. '
	G = Graph.Graph()
	for u in neighbors.keys():
		for v in neighbors[u].keys():
			' make sure that each edge is only added once '
			if u < v:
				edgeWeight = neighbors[u][v]
				sign = 1 if edgeWeight >= 0 else -1
				G.addEdge(u,v,sign)

	return G

