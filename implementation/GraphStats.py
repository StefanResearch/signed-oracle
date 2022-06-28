import numpy

import Graph

def printGraphStats(G):
	print(f'n:\t\t {G.numVertices}')
	print(f'|E|:\t\t {G.numEdges}')
	print(f'|E+|:\t\t {G.numPositiveEdges}')
	print(f'|E-|:\t\t {G.numNegativeEdges}')
	print(f'|E-|/|E|:\t {G.numNegativeEdges / G.numEdges}')
	print(' ')

	degrees = []
	for u in G.edges.keys():
		degrees.append(len(G.edges[u]))
	
	print(f'deg avg:\t {numpy.average(degrees)}')
	print(f'deg std:\t {numpy.std(degrees)}')
	print(f'deg max:\t {numpy.max(degrees)}')
	print(f'deg min:\t {numpy.min(degrees)}')
	print(' ')
