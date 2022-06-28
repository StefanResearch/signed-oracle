/*
 * compile with
 * 		g++-11 -O3 -Wall -fopenmp -std=c++11 oracle.cpp -o oracle
 */
#include <fstream>
#include <iostream>

#include <map>
#include <string>
#include <tuple>
#include <queue>
#include <vector>

#include <algorithm>
#include <random>

#include <cstring>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <omp.h>

#include "DisjointSet.cpp"

typedef enum initializationOptions {
	INIT_NOT_SET,
	INIT_RANDOM,
	INIT_SEEDS
} Init_Type;

typedef long vertexId;
typedef std::tuple<double, vertexId, vertexId> weightedEdge;
typedef std::pair<vertexId, vertexId> unweightedEdge;
typedef std::map<vertexId, std::vector<vertexId> > Graph;

Graph readGraph(std::string graphfile);
void addSignedEdge(Graph& G, vertexId u, vertexId v, int sign);
void addVertexIfNotExisting(Graph& G, vertexId u);

long numSteps;
long numWalks;
long numEstNorm;
double estNorm(Graph& G, vertexId u, vertexId v);
double estNorm(Graph& G, vertexId u, vertexId v, long numRepetitions);
double estDotProd(Graph& G, vertexId u, vertexId v); // returns the absolute value of the dot product
std::map<vertexId, double> performRandomWalks(Graph& G, vertexId u);

bool biclustering;
std::vector<vertexId> seedNodes;
std::map<vertexId,long> seedToCluster;
long k = -1;
std::vector< std::vector<vertexId> > seedClusters; // used internally for initializing with seeds
bool initializedSeeds;
long s; // used internally for random initialization
void initWithClusteredSeedVertices(std::vector< std::vector<vertexId > > seedClusters);
void initWithRandomSeeds(Graph& G, long k, long s);
void initSeeds(Graph& G, Init_Type initiliazationOption);

long whichCluster(Graph& G, vertexId u);
bool sameCluster(Graph& G, vertexId u, vertexId v);

std::vector<std::vector<vertexId> > computeClusteringOfAllVertices(Graph &G);
std::vector<std::vector<vertexId> > computeClusteringOfVertices(Graph &G, std::vector<vertexId> vertices);

bool ignoreSigns;

int
main(int argc, char** argv) {
	// perform some initialization
	srand(time(NULL));
	Init_Type initializedSeedClusters = INIT_NOT_SET;
	numEstNorm = 1;
	initializedSeeds = false;
	ignoreSigns = false;
	biclustering = false;

	bool answerSameClusterQueries = false;
	std::vector< unweightedEdge > queries;

	// parse the command line input
	std::string inputfilepath(argv[1]);
	numSteps = strtol(argv[2],NULL,10);
	numWalks = strtol(argv[3],NULL,10);

	Graph G = readGraph(inputfilepath);

	int i=4;
	while (i<argc) {
		if (strcmp(argv[i], "--seedCluster") == 0) {
			initializedSeedClusters = INIT_SEEDS;

			i++; // go to the next argument, as the current one is "--seedCluster"

			std::vector<vertexId> cluster;
			while (strcmp(argv[i], "endSeedCluster") != 0) {
				vertexId u = (vertexId)strtol(argv[i],NULL,10);
				cluster.push_back(u);

				i++;
			}

			seedClusters.push_back(cluster);
		} else if (strcmp(argv[i], "--randomSeeds") == 0) {
			initializedSeedClusters = INIT_RANDOM;

			k = strtol(argv[i+1],NULL,10);
			s = strtol(argv[i+2],NULL,10);

			i += 2;
		} else if (strcmp(argv[i], "--clusterAll") == 0) {
			if (! initializedSeeds) {
				initSeeds(G, initializedSeedClusters);
			}

			std::vector< std::vector<vertexId> > clustering;
			clustering = computeClusteringOfAllVertices(G);

			for (auto c : clustering) {
				for (auto u : c) {
					std::cout << u << " ";
				}
				std::cout << std::endl;
			}
		} else if (strcmp(argv[i], "--clusterVertices") == 0) {
			i++; // go to the next argument, as the current one is "--clusterVertices"

			std::vector<vertexId> vertices;
			while (strcmp(argv[i], "endClusterVertices") != 0) {
				vertexId u = (vertexId)strtol(argv[i],NULL,10);
				vertices.push_back(u);

				i++;
			}

			if (! initializedSeeds) {
				initSeeds(G, initializedSeedClusters);
			}

			std::vector< std::vector<vertexId> > clustering;
			clustering = computeClusteringOfVertices(G, vertices);

			for (auto c : clustering) {
				for (auto u : c) {
					std::cout << u << " ";
				}
				std::cout << std::endl;
			}
		} else if (strcmp(argv[i], "--numEstNorm") == 0) {
			numEstNorm = strtol(argv[i+1],NULL,10);
			i += 1;
		} else if (strcmp(argv[i], "--unsigned") == 0) {
			ignoreSigns = true;
		} else if (strcmp(argv[i], "--biclustering") == 0) {
			biclustering = true;
		} else if (strcmp(argv[i], "--sameCluster") == 0) {
			answerSameClusterQueries = true;

			vertexId u = strtol(argv[i+1],NULL,10);
			vertexId v = strtol(argv[i+2],NULL,10);
			queries.push_back(std::make_pair(u,v));

			i += 2;
		} else {
			std::cout << "The argument " << argv[i] << " is not supported." << std::endl;
		}
		
		i++;
	}

	// Answer the sameCluster queries that we got in parallel.
	// We initialize and output in a single thread and compute the query
	// solutions in parallel.
	if (answerSameClusterQueries) {
		std::map<unweightedEdge,long> answers;
		for (auto query : queries) {
			answers[query] = 0;
		}

		if (! initializedSeeds) {
			initSeeds(G, initializedSeedClusters);
		}

		#pragma omp parallel for
		for (auto query : queries) {
			vertexId u = query.first;
			vertexId v = query.second;
			answers[query] = (int)sameCluster(G,u,v);
		}

		for (auto query : queries) {
			vertexId u = query.first;
			vertexId v = query.second;
			long answer = answers[query];

			std::cout << u << " " << v << " " << answer << std::endl;
		}
	}

	return 0;
}

Graph readGraph(std::string graphfile) {
	Graph G;

	std::ifstream inputfile(graphfile);
	if (inputfile.fail()) {
		std::cout << "ERROR: Could not open input file: " << graphfile << std::endl;
		return G;
	}

	vertexId u, v;
	int sign;
	while (inputfile >> u >> v >> sign) {
		addSignedEdge(G, u, v, sign);
	}

	return G;
}

void addSignedEdge(Graph& G, vertexId u, vertexId v, int sign) {
	addVertexIfNotExisting(G, u);
	addVertexIfNotExisting(G, v);

	G[u].push_back(v * sign);
	G[v].push_back(u * sign);
}

void addVertexIfNotExisting(Graph& G, vertexId u) {
	if (G.find(u) == G.end()) {
		G[u] = std::vector<vertexId>();
	}
}

void initWithClusteredSeedVertices(std::vector< std::vector<vertexId > > seedClusters) {
	k = seedClusters.size();

	seedNodes = std::vector<vertexId>();
	seedToCluster = std::map<vertexId,long>();

	for (size_t i=0; i<seedClusters.size(); i++) {
		std::vector<vertexId> cluster = seedClusters[i];

		for (size_t j=0; j<cluster.size(); j++) {
			vertexId seed = cluster[j];
			seedNodes.push_back(seed);
			seedToCluster[seed] = i;
		}
	}
}

void initWithRandomSeeds(Graph& G, long k, long s) {
	if (k > s) {
		std::cout << "Cannot initialize oracle. We need k <= s." << std::endl;
	}

	// randomly sample s seed nodes
	seedNodes = std::vector<vertexId>();
	for (long i=0; i<s; i++) {
		auto it = G.begin();
		std::advance(it, rand() % G.size());

		vertexId seed = it->first;
		seedNodes.push_back(seed);
	}

	// estimate the pair-wise distances between all seed nodes
	std::vector<weightedEdge> edgeWeights;
	for (long i=0; i<s; i++) {
		vertexId u = seedNodes[i];

		for (long j=i+1; j<s; j++) {
			vertexId v = seedNodes[j];

			// we make the norms negative so that the heap orders them from
			// small to larger
			double norm = -estNorm(G, u, v, 5);
			weightedEdge edge {norm, u, v};
			edgeWeights.push_back(edge);
		}
	}

	// keep on merging seed nodes until we have k clusters
	DisjointSet unionFind(seedNodes);
	std::priority_queue<weightedEdge> heap(edgeWeights.begin(), edgeWeights.end());
	while (unionFind.numComponents > k) {
		// merge the components of u and v
		vertexId u = std::get<1>(heap.top());
		vertexId v = std::get<2>(heap.top());
		unionFind.union_set(u,v);

		heap.pop();
	}

	// we can now initialize seedToCluster but we have to make sure that the
	// clusterIds are in the range 0...k-1.
	seedToCluster = std::map<vertexId,long>();

	long numAssignedClusters = 0;
	std::map<long,long> clusterToClusterId;
	for (vertexId seed : seedNodes) {
		long parent = unionFind.find_set(seed);

		if (clusterToClusterId.find(parent) == clusterToClusterId.end()) {
			clusterToClusterId[parent] = numAssignedClusters;
			numAssignedClusters++;
		}

		seedToCluster[seed] = clusterToClusterId[parent];
	}
}

void initSeeds(Graph& G, Init_Type initOption) {
	if (initOption == INIT_RANDOM) {
		initWithRandomSeeds(G, k, s);
	} else if (initOption == INIT_SEEDS) {
		initWithClusteredSeedVertices(seedClusters);
	} else {
		std::cout << "Please specify how the seed nodes should be initialized." << std::endl;
		return;
	}

	initializedSeeds = true;
}

std::vector<std::vector<vertexId> >
computeClusteringOfAllVertices(Graph &G) {
	std::vector< vertexId > vertices;
	for (Graph::iterator it = G.begin(); it != G.end(); it++) {
		vertexId u = it->first;

		vertices.push_back(u);
	}

	return computeClusteringOfVertices(G, vertices);
}

std::vector<std::vector<vertexId> > 
computeClusteringOfVertices(Graph &G,
							std::vector<vertexId> vertices) {
	std::map< vertexId, long > vertexToCluster;
	for (auto u : vertices) {
		vertexToCluster[u] = -1;
	}

	#pragma omp parallel for
	for (size_t i=0; i<vertices.size(); i++) {
		vertexId u = vertices[i];
		long clusterId = whichCluster(G, u);
		vertexToCluster[u] = clusterId;
	}

	std::vector<std::vector<vertexId > > clusters;
	for (long i=0; i<k; i++) {
		clusters.push_back(std::vector<vertexId>());
	}

	for (size_t i=0; i<vertices.size(); i++) {
		vertexId u = vertices[i];
		long clusterId = vertexToCluster[u];
		clusters[clusterId].push_back(u);
	}
	
	return clusters;
}

long whichCluster(Graph& G, vertexId u) {
	double minDistance = std::numeric_limits<double>::max();
	vertexId minSeed = -1;

	for (size_t i=0; i<seedNodes.size(); i++) {
		vertexId seed = seedNodes[i];
		double norm = estNorm(G, u, seed);

		if (norm < minDistance) {
			minDistance = norm;
			minSeed = seed;
		}
	}

	return seedToCluster[minSeed];
}

bool sameCluster(Graph& G, vertexId u, vertexId v) {
	return (whichCluster(G,u) == whichCluster(G,v));
}

double estNorm(Graph &G, vertexId u, vertexId v) {
	return estNorm(G, u, v, numEstNorm);
}

/**
  * Estimates the norm numReptitions times and returns the median.
  * If numRepetitions is even, we use numRepetitions/2-1 as median element.
  */
double estNorm(Graph &G, vertexId u, vertexId v, long numRepetitions) {
	std::vector<double> norms;

	for (int i=0; i<numRepetitions; i++) {
		double norm = estDotProd(G, u, u);
		norm -= 2*estDotProd(G, u, v);
		norm += estDotProd(G, v, v);

		norms.push_back(norm);
	}

	std::sort(norms.begin(), norms.end());
	// for the medianIndex we need to subtract 1 because the vector is 0-indexed
	long medianIndex = numRepetitions/2 + (numRepetitions%2 == 1) - 1;
	double norm = norms[medianIndex];

	return norm;
}

// returns the absolute value of the dot product
double estDotProd(Graph& G, vertexId u, vertexId v) {
	double dotProduct = 0;

	std::map<vertexId, double> weightsU = performRandomWalks(G, u);
	std::map<vertexId, double> weightsV = performRandomWalks(G, v);

	// we only have to iterate over the keys of walksU because if w is not in
	// weightsU but in weightsV, then the product of weightsU[w]*weightsV[w] is 0.
	for (std::map<vertexId,double>::iterator it = weightsU.begin(); it != weightsU.end(); it++) {
		vertexId w = it->first;
		if (weightsV.find(w) != weightsV.end()) {
			dotProduct += it->second * weightsV[w];
		}
	}

	return dotProduct;
}

std::map<vertexId, double> performRandomWalks(Graph& G, vertexId u) {
	std::map<vertexId, double> weights;

	std::default_random_engine generator;
	std::geometric_distribution<int> distribution(0.5);

	// perform all random walks
	for (long i=0; i<numWalks; i++) {
		vertexId currentVertex = u;
		int walkSign = 1;
		long remainingSteps = numSteps;

		// perform the random walk
		do {
			// perform the self-loop steps
			long numSelfLoops = distribution(generator);
			remainingSteps -= numSelfLoops;

			// now go to a neighbor
			remainingSteps -= 1;
			long numNeighbors = G[currentVertex].size();
			if (numNeighbors == 0) {
				// we have found an isolated vertex
				remainingSteps = 0;
				break;
			}

			long randomNeighborIndex = rand() % numNeighbors;

			// if we take a negative edge and we are using the signs, 
			// then flip the sign of the walk
			if (G[currentVertex][randomNeighborIndex] < 0 && (!ignoreSigns)) {
				walkSign *= -1;
			}

			// walk to the neighbor
			currentVertex = abs( G[currentVertex][randomNeighborIndex] );
		} while (remainingSteps > 0);

		// we have finished the walk
		if (weights.find(currentVertex) == weights.end()) {
			weights[currentVertex] = 0.0;
		}
		weights[currentVertex] += walkSign / (double)numWalks;
	}

	// we normalize by the root of the vertex degree,
	// if we do not compute a biclustering, we also take absolute values
	for (std::map<vertexId,double>::iterator it = weights.begin(); it != weights.end(); it++) {
		weights[it->first] = it->second / sqrt(G[it->first].size());
		if (! biclustering) {
			weights[it->first] = fabs(weights[it->first]);
		}
	}

	return weights;
}

