import os

def getInitCommand(inputfile,
				   numSteps,
				   numWalks, 
				   seedClusters=None, 
				   k=None, 
				   s=None, 
				   numEstNorm=1, 
				   unsigned=False,
				   biclustering=False):
	command = './oracle '
	command += f'{inputfile} '
	command += f'{str(numSteps)} '
	command += f'{str(numWalks)} '

	command += f'--numEstNorm {numEstNorm} '

	if unsigned:
		command += '--unsigned '

	if biclustering:
		command += '--biclustering '

	if seedClusters == None:
		command += f'--randomSeeds {k} {s} '
	else:
		' tell the algorithm about the seedClusters '
		for seedCluster in seedClusters:
			command += '--seedCluster '
			for u in seedCluster:
				command += f'{str(u)} '
			command += 'endSeedCluster '

	return command

def classifyAllVertices(inputfile,
						numSteps, 
						numWalks, 
						seedClusters=None, 
						k=None, 
						s=None, 
						numEstNorm=1, 
						unsigned=False,
						biclustering=False):
	command = getInitCommand(inputfile, numSteps, numWalks, seedClusters, k, s, numEstNorm, unsigned=unsigned, biclustering=biclustering)

	command += '--clusterAll '

	return computeVertexClusteringWithCommand(command)

def classifyGivenVertices(verticesToClassify,
						  inputfile, 
						  numSteps, 
						  numWalks, 
						  seedClusters=None, 
						  k=None, 
						  s=None, 
						  numEstNorm=1, 
						  unsigned=False, 
						  biclustering=False):
	command = getInitCommand(inputfile, numSteps, numWalks, seedClusters, k, s, numEstNorm, unsigned=unsigned, biclustering=biclustering)

	command += '--clusterVertices '
	for vtx in verticesToClassify:
		command += f'{str(vtx)} '
	command += 'endClusterVertices '

	return computeVertexClusteringWithCommand(command)

def computeVertexClusteringWithCommand(command):
	stream = os.popen(command)

	clustering = []

	lines = stream.readlines()
	for line in lines:
		line.strip()
		cluster = [u.strip() for u in line.split(' ')]
		cluster = [int(u) for u in cluster if u != '']
		clustering.append(cluster)

	return clustering

'''
	Returns a dict queryAnswers, where the keys are strings of format
	"u-v" and the values are 0 or 1 based on the query answer.
	E.g., if u=153 and v=39 then the key is '153-39'.
'''
def runQueries(inputfile,
			   numSteps,
			   numWalks, 
			   queries, 
			   seedClusters=None, 
			   k=None, 
			   s=None, 
			   numEstNorm=1, 
			   unsigned=False,
			   biclustering=False):
	command = getInitCommand(inputfile, numSteps, numWalks, seedClusters, k, s, numEstNorm, unsigned=unsigned, biclustering=biclustering)

	for query in queries:
		u = query[0]
		v = query[1]
		command += f'--sameCluster {u} {v} '

#	print(command)
	stream = os.popen(command)

	queryAnswers = {}
	lines = stream.readlines()
	for line in lines:
		linesplit = line.strip().split(' ')
		u = int(linesplit[0])
		v = int(linesplit[1])
		answer = int(linesplit[2])

		queryAnswers[f'{u}-{v}'] = answer

	return queryAnswers
		
