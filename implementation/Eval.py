import numpy
import random

import os
import sys
import time

import networkx as nx
from networkx.algorithms import bipartite

import GraphReader
import GraphStats
import SyntheticData

import OracleModule

sys.path.insert(1, 'include/signed-local-community-master')
from core import query_graph_using_sparse_linear_solver, sweep_on_x_fast
from helpers import get_v1

algos = ['seeded','random','useeded','urandom','FOCG','polar']

quantities = ['accuracy','time','time-norm']

numRepetitions = 5
experimentsOutputFilePath = '../results/results.csv'

def runAllSyntheticExperiments():
	global algos

	algos = ['seeded','useeded','random','urandom','FOCG','polar']

	runSyntheticExperiments('k', [4,6,8,10,12], biclustering=True)
	runSyntheticExperiments('numWalks', [200,400,600,800], biclustering=True)
	runSyntheticExperiments('numSteps', [1,2,3,4,5], biclustering=True)
	runSyntheticExperiments('n', [500,1000,2000,3000,4000], biclustering=True)

	runSyntheticExperiments('k', [4,8,12,16], biclustering=False)
	runSyntheticExperiments('numWalks', [50,100,200,400], biclustering=False)
	runSyntheticExperiments('numSteps', [1,2,3,4,5], biclustering=False)
	runSyntheticExperiments('n', [500,1000,2000,3000,4000], biclustering=False)


def runSyntheticExperiments(paramKey,
							paramValues,
							biclustering=False):
	paramsList = []

	for paramValue in paramValues:
		params = getStandardParamsSynthetic(biclustering=biclustering)
		params[paramKey] = paramValue

		paramsList.append(params)

	runExperimentsVaryParam(paramKey,
							paramsList,
							biclustering=biclustering)
	
def getStandardParamsSynthetic(biclustering=False):
	params = dict()

	' parameters for random graph '
	params['n'] = 2000
	params['k'] = 6

	params['method'] = 'randomGraph'
	params['pIntra'] = 0.8
	params['pCross'] = 0.4
	params['q'] = 0.05
	params['pSign'] = 0.8
	params['posInter'] = 0.9

	' parameters for algorithms '
	params['numSeedsPerBicluster'] = 3
	if biclustering:
		params['numWalks'] = 600
		params['numSteps'] = 2
	else:
		params['numWalks'] = 400
		params['numSteps'] = 2

	return params

'''
	Expects a list paramsList.
	Each entry in the list should contain a dict specifying all parameters for
	running the algorithms and running the data. For each parameter set,
	numRepetitions trials are run (this is a global parameter).
	paramKey should contain the key for the parameter that is being varied.
	Only one parameter is allowed to be varied.
	
	These parameters to be set in each dict are the following:
		' algo parameters for oracle algorithms '
		numSteps
		numWalks

		method -- specifices where the ground truth comes from,
					either randomGraph, wikiSmall, wikiMedium or wikiLarge
		k

		' parameters for random graph '
		n
		pIntra
		pCross
		q
		pSign
		posInter

		' we are using wiki data '
		inputfile
'''
def runExperimentsVaryParam(paramKey,
							paramsList,
							biclustering=False):
	print(f'running {numRepetitions} repetitions')
	for repetition in range(numRepetitions):
		for params in paramsList:
			params['paramKey'] = paramKey
			paramValue = params[paramKey]
			print(f'-----------------------------------------------------------------------------')
			print(f'Repetition {repetition}: Starting {params["method"]} experiments for {paramKey}={paramValue}.')

			G,_,plantedClusters = getGroundTruthCommunities(params)

			' build seed sets for those algorithms that need it '
			verticesToCluster = []
			clusteredSeedVertices = [] # contains seeds for each U_i = V_2i + V_(2i+1)
			polarSeedVertices = [] # contains seeds for each V_i
			numSeedsPerBicluster = params['numSeedsPerBicluster']
			k = params['k']
			for i in range(k):
				verticesToCluster.extend(plantedClusters[2*i])
				verticesToCluster.extend(plantedClusters[2*i+1])

				V1 = random.sample(plantedClusters[2*i],numSeedsPerBicluster)
				V2 = random.sample(plantedClusters[2*i+1],numSeedsPerBicluster)

				if biclustering:
					clusteredSeedVertices.append(V1)
					clusteredSeedVertices.append(V2)
				else:
					U = V1 + V2
					clusteredSeedVertices.append(U)

				''' The polarSeedVertices are only used by polar and polar
					uses 0-indexed vertex ids. Thus, we need to subtract 1 from
					the vertex ids. '''
				polarSeedVertices.append([v-1 for v in V1])
				polarSeedVertices.append([v-1 for v in V2])

			' update the params and inputs for the algorithms '
			params['G'] = G
			params['plantedClusters'] = plantedClusters
			params['verticesToCluster'] = verticesToCluster
			params['polarSeedVertices'] = polarSeedVertices
			params['clusteredSeedVertices'] = clusteredSeedVertices

			' run the algos '
			for algo in algos:
				algoLines = runAlgo(algo, params, biclustering=biclustering)

				with open(experimentsOutputFilePath, 'a') as experimentsOutputFile:
					for algoLine in algoLines:
						method = params['method']
						line = f'{method},{paramKey},{paramValue},{algoLine}\n'
						experimentsOutputFile.write(line)

			' clean up the params '
			del params['G']
			del params['plantedClusters']
			del params['verticesToCluster']
			del params['polarSeedVertices']
			del params['clusteredSeedVertices']


def runAlgo(algoName, params, biclustering=False):
	print(f'\tRunning {algoName} (biclustering = {biclustering}).')

	' get other parameters '
	n = params['n']
	k = 2*params['k'] if biclustering else params['k']
	G = params['G']
	plantedClusters = params['plantedClusters']

	' set algo parameters '
	numSteps = params['numSteps']
	numWalks = params['numWalks']

	' do some more preprocessing such that the timing is not affected '
	if algoName == 'polar':
		Gnx = G.getNetworkxGraph() # this also makes the graph 0-indexed
		l1, v1 = get_v1(Gnx)
		Gnx.graph['lambda1'] = l1
		Gnx.graph['v1'] = v1

	graphfile = G.writeSparseMatrixToTmp()
	
	numEstNorm = 1 if biclustering else 1
	verticesToCluster = params['verticesToCluster']

	t = time.time()

	clustering = []
	if algoName == 'random':
		s = params['numSeedsPerBicluster']*k
		clustering = OracleModule.classifyGivenVertices(verticesToCluster, graphfile, numSteps, numWalks, k=k, s=s, numEstNorm=numEstNorm, biclustering=biclustering)
	elif algoName == 'urandom':
		s = params['numSeedsPerBicluster']*k
		clustering = OracleModule.classifyGivenVertices(verticesToCluster, graphfile, numSteps, numWalks, k=k, s=s, unsigned=True, numEstNorm=numEstNorm, biclustering=biclustering)
	elif algoName == 'seeded':
		clustering = OracleModule.classifyGivenVertices(verticesToCluster, graphfile, numSteps, numWalks, seedClusters=params['clusteredSeedVertices'], numEstNorm=numEstNorm, biclustering=biclustering)
	elif algoName == 'useeded':
		clustering = OracleModule.classifyGivenVertices(verticesToCluster, graphfile, numSteps, numWalks, seedClusters=params['clusteredSeedVertices'], unsigned=True, numEstNorm=numEstNorm, biclustering=biclustering)
	elif algoName == 'FOCG':
		' restart timing inside subroutine to not penalize the graph I/O time '
		if biclustering:
			_,clustering,t = runFOCG(G)
		else:
			clustering,_,t = runFOCG(G)
	elif algoName == 'polar':
		biclusterSeeds = params['polarSeedVertices']

		kPrime = int(k/2) if biclustering else k
		for i in range(kPrime):
			seeds1 = biclusterSeeds[2*i]
			seeds2 = biclusterSeeds[2*i+1]

			x, obj_val = query_graph_using_sparse_linear_solver(Gnx, [seeds1, seeds2], kappa=0.9, verbose=0, ub=max(0,Gnx.graph['lambda1']))
			C1, C2, C, best_t, best_sbr, ts, sbr_list = sweep_on_x_fast(Gnx, x, top_k=100)
			
			' it is important that we do not use C here because C often contains the full graph '
			' also, since polar uses 0-indexed graphs, we have to change the indices '
			if biclustering:
				cluster1 = [x+1 for x in list(C1)]
				cluster2 = [x+1 for x in list(C2)]
				clustering.append(cluster1)
				clustering.append(cluster2)
			else:
				cluster = [x+1 for x in list(C1)+list(C2)]
				cluster.sort()
				clustering.append(cluster)
		Gnx = None
	else:
		print('runAlgo: Got unknown algorithm. Stopping.')
		return

	totalTime = time.time() - t
	print(f'	- finished after {totalTime} seconds')

	paramKey = params['paramKey']
	paramValue = params[paramKey]

	stats = dict()
	stats['time'] = totalTime
	stats['time-norm'] = totalTime/len(verticesToCluster)

	mergeBiclusters = False if biclustering else True
	quality = computeQualityOfClustering(plantedClusters,
										 clustering,
										 mergeBiclusters=mergeBiclusters,
										 verticesToCluster=verticesToCluster)
	stats['accuracy'] = quality

	lines = []
	for quantity in quantities:
		stat = stats[quantity]
		line = f'{algoName},{biclustering},{quantity},{stat}'
		lines.append(line)

	os.remove(graphfile)

	return lines
	
def computeQualityOfClustering(plantedBiclusters,
							   clustering,
							   mergeBiclusters=True,
							   verticesToCluster=None):
	''' 
		we do not want to penalize the algorithms if they find some vertices
		that are not part of the planted communities, so we remove them from the
		algorithm output
	'''
	prunedClustering = []
	for cluster in clustering:
		prunedCluster = []
		for plantedBicluster in plantedBiclusters:
			plantedBiclusterSet = set(plantedBicluster)
			for u in cluster: # include u if it is contained in any of the planted biclusters
				if u in plantedBiclusterSet:
					prunedCluster.append(u)
		prunedClustering.append(prunedCluster)

	'''
		if we only wanted to cluster the vertices in verticesToCluster,
		then we have to remove all other vertices from the plantedBiclusters
		that are not contained in verticesToClusters
	'''
	plantedBiclustersPruned = plantedBiclusters
	if verticesToCluster != None:
		plantedBiclustersPruned = []
		
		for plantedBicluster in plantedBiclusters:
			intersection = set(plantedBicluster).intersection(set(verticesToCluster))
			plantedBiclustersPruned.append(list(intersection))

	return computeQualityOfClusteringMatching(plantedBiclustersPruned, prunedClustering, mergeBiclusters=mergeBiclusters)

def computeQualityOfClusteringMatching(plantedBiclusters,
									   clustering,
									   mergeBiclusters=True):
	n = numpy.sum( [len(cluster) for cluster in plantedBiclusters] )
	
	' merge corresponding biclusters into clusters '
	if mergeBiclusters:
		plantedClusters = []
		k = int(len(plantedBiclusters)/2)
		for i in range(k):
			V1 = plantedBiclusters[2*i]
			V2 = plantedBiclusters[2*i+1]
			A = set(V1 + V2)

			plantedClusters.append(A)
	else:
		k = len(plantedBiclusters)
		plantedClusters = [set(c) for c in plantedBiclusters]

	' make sure both clusterings have same length '
	for i in range(max(len(plantedClusters),len(clustering))):
		if i >= len(plantedClusters):
			plantedClusters.append([])
		if i >= len(clustering):
			clustering.append([])

	G = nx.Graph()
	leftNodes = [f'l{i}' for i in range(k)]
	rightNodes = [f'r{j}' for j in range(len(clustering))]
	G.add_nodes_from(leftNodes,bipartite=0)
	G.add_nodes_from(rightNodes,bipartite=1)

	for i in range(k):
		A = plantedClusters[i]

		for j in range(len(clustering)):
			cluster = clustering[j]
			B = set(cluster)

			weight = -len(A.intersection(B))

			G.add_edge(f'l{i}',f'r{j}',weight=weight)

	matching = bipartite.matching.minimum_weight_full_matching(G,leftNodes,'weight')

	value = 0
	for li in leftNodes:
		rj = matching[li]

		i = int(li[1:])
		j = int(rj[1:])

		A = plantedClusters[i]
		B = set(clustering[j])

		value += len(A.intersection(B))
		print(f'	got accuracy {len(A.intersection(B))/len(A)} for cluster with {len(A)} vertices')

	return value/n

'''
	Assumes that A and B are sets and returns the size of their symmetric
	difference.
'''
def symmetricDifference(A, B):
	diff = len(A.difference(B))
	diff += len(B.difference(A))

	return diff

def runFOCG(G):
	matlabBin = '/Applications/MATLAB_R2021a.app/bin/matlab'
	pathToFOCG = 'include/KOCG.SIGKDD2016-master'

	Gpath = f'{pathToFOCG}/tmp.dat'
	outputPath = f'{pathToFOCG}/FOCG.out'

	G.writeToSparseFormatForMatlab(Gpath)

	' start timing now to not penalize the graph I/O time '
	t = time.time()
	command = f'{matlabBin} -nodesktop -nosplash'
	command += f' -r "run {pathToFOCG}/RunFromPython; exit"'
	command += f' -logfile "{pathToFOCG}/FOCG.log"'
	os.system(command)

	biclustering = []
	with open(outputPath) as fp:
		line = fp.readline()
		while line:
			cluster = [int(x) for x in line.split(' ') if x not in ['','\n']]
			cluster = [x for x in cluster if x in G.edges.keys()]
			biclustering.append(cluster)

			line = fp.readline()

	clustering = []
	for i in range(int(len(biclustering)/2)):
		cluster = biclustering[2*i] + biclustering[2*i+1]
		clustering.append(cluster)
	
	os.remove(Gpath)
	os.remove(outputPath)

	return clustering, biclustering, t

def runAllRealWorldExperiments():
	global numRepetitions
	global algos
	numRepetitions = 5

	algos = ['seeded','useeded','random','urandom','FOCG','polar']

	' wiki small clustering and biclustering '
	for biclustering in [False,True]:
		runRealWorldExperiments('numWalks', [500,1000,1500,2000], biclustering=biclustering, size='small')
		runRealWorldExperiments('numSteps', [12,16,20,24], biclustering=biclustering, size='small')

	algos = ['seeded','useeded','FOCG','polar']

	' wiki medium experiments ' 
	runRealWorldExperiments('numWalks', [1000, 2500, 5000], biclustering=False, size='medium')
	runRealWorldExperiments('numWalks', [1000, 2500, 5000, 10000], biclustering=True, size='medium')

	' wiki large experiments ' 
	runRealWorldExperiments('numWalks', [1000, 2500, 5000], biclustering=False, size='large')
	runRealWorldExperiments('numWalks', [5000,10000,2500], biclustering=True, size='large')

def runRealWorldExperiments(paramKey,
							paramValues,
							biclustering=False,
							size='small'):
	paramsList = []

	for paramValue in paramValues:
		params = dict()
		params = getStandardParamsWiki(biclustering=biclustering, size=size)

		params[paramKey] = paramValue

		paramsList.append(params)

	runExperimentsVaryParam(paramKey,
							paramsList,
							biclustering=biclustering)

def getStandardParamsWiki(biclustering=False, size='small'):
	params = dict()
	params['numSeedsPerBicluster'] = 5

	if size == 'small':
		params['inputfile'] = '../data/wiki_small'
		params['n'] = 9211
		params['k'] = 5
		params['numWalks'] = 1000
		params['numSteps'] = 20
		params['method'] ='wikiSmall'
	elif size == 'medium':
		params['inputfile'] = '../wiki/wiki_medium'
		params['n'] = 34404
		params['k'] = 5
		params['numWalks'] = 5000 if biclustering else 2500
		params['numSteps'] = 24
		params['method'] = 'wikiMedium'
	elif size == 'large':
		params['inputfile'] = '../wiki/wiki_large'
		params['n'] = 258259
		params['k'] = 5
		params['numWalks'] = 10000 if biclustering else 2500
		params['numSteps'] = 20
		params['method'] = 'wikiLarge'
	else:
		print(f'No wiki dataset found for size {size}.')

	return params

def getGroundTruthCommunities(params):
	G = None
	clustering = []
	biclustering = []

	method = params['method']
	k = params['k']
	inputfile = params['inputfile'] if 'inputfile' in params else ''

	if method == 'randomGraph':
		n = params['n']
		pIntra = params['pIntra']
		pCross = params['pCross']
		q = params['q']
		pSign = params['pSign']
		posInter = params['posInter']
		G, clustering, biclustering = SyntheticData.signedSBM(n, k, pIntra, pCross, q, pSign, posInter)
	elif 'wiki' in method:
		graphfile = f'{inputfile}/edges.txt'
		G = GraphReader.graphFromSparseCSV(graphfile, separator='#')
		
		partyToId = {}
		partiesfile = f'{inputfile}/partyToId.txt'
		with open(partiesfile) as fp:
			while True:
				line = fp.readline()
				if not line:
					break

				lineSplit = str.split(line, '#')
				partyToId[int(lineSplit[0].strip())] = lineSplit[1].strip()

		numClusters = max(list(partyToId.keys()))
		clustering = [[] for i in range(numClusters)]
		biclustering = [[] for i in range(2*numClusters)]

		verticesfile = f'{inputfile}/vtxToPage.txt'
		with open(verticesfile) as fp:
			while True:
				line = fp.readline()
				if not line:
					break

				lineSplit = str.split(line, '#')
				if lineSplit[2].strip() == '':
					continue

				partyId = int(lineSplit[2].strip())
				vtxId = int(lineSplit[0].strip())

				clusterIndex = abs(partyId)-1
				clustering[clusterIndex].append(vtxId)

				biclusterIndex = 2*clusterIndex if partyId > 0 else 2*clusterIndex+1
				biclustering[biclusterIndex].append(vtxId)
	else:
		print('Unknown method for running real-world experiments.')
		return None, [], []
	
	GraphStats.printGraphStats(G)

	''' pick the k clusters with the most vertices '''
	k = min(k,len(clustering)) # make k smaller if the algorithm found fewer communities

	print(f'cluster sizes before picking {[len(c) for c in clustering]}')

	clustersizes = [len(cluster) for cluster in clustering]
	largestIndices = numpy.argsort(clustersizes)[-k:]
	clustering = [clustering[i] for i in largestIndices]

	''' computing the bipartiteness ratios '''
	signedBipartitenessRatios = []
	for i in range(len(clustering)):
		V1 = set(biclustering[2*i])
		V2 = set(biclustering[2*i+1])
		signedBipartitenessRatio = G.signedBipartitenessRatio(V1,V2)
		if signedBipartitenessRatio == 0:
			signedBipartitenessRatio = numpy.Inf
		signedBipartitenessRatios.append(signedBipartitenessRatio)

	print(f'cluster sizes: {[len(cluster) for cluster in clustering]}')
	print(f'signed bipartiteness ratios: {[signedBipartitenessRatios[x] for x in range(k)]}')

	return G, clustering, biclustering

