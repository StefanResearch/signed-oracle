# signed-oracle

This is the implementation for the paper
>	Sublinear-Time Clustering Oracle for Signed Graphs
by Stefan Neumann and Pan Peng, ICML'22.

## Oracle Implementation
The implementation of our signed oracle can be found in folder "implementation".
To use our oracle, you first need to compile it, e.g., using:
>	g++-11 -O3 -Wall -fopenmp -std=c++11 oracle.cpp -o oracle
To compile it, you need a compiler that is compliant with c++11 and you need
OpenMP if you want to use parallelization.

To run the oracle, use the following call:
>	./oracle inputFilePath numSteps numWalks

The parameters of the oracle are as follows:
* *inputFilePath*:
	The path to the file that stores the graph.
	We assume that the graph is stored using a sparse format, in which each edge
	has the format "u#v#sign" where u and v are the vertex IDs and sign is the
	edge sign. We assume that sign is either 1 or -1.
* *numSteps*:
	The number of random walk steps to be used by the algorithm.
* *numWalks*:
	The number of random walks to be used by the algorithm.

### Optional Oracle Parameters
Additionally, the oracle accepts the following optional parameters:
* *--seedCluster ... endSeedCluster*:
	Allows to specify a new ground-truth seed cluster.
	For example, "--seedCluster 1 30 48 70 --endSeedCluster" creates a
	ground-truth cluster of seed nodes containing vertices 1, 30, 48, 70.
	It is important that the cluster is ended with "endSeedCluster".
* *--randomSeeds s*:
	Tells the oracle to randomly sample s seed nodes and to preprocess them as
	described in the paper.
* *--clusterAll*:
	Returns a clustering of all vertices in the graph.
	Calls the WhichCluster-procedure for each vertex in the graph.
* *--clusterVertices ... endClusterVertices*:
	Returns a clustering of the given vertices.
	For example, "--clusterVertices 5 10 39 4588 endClusterVertices" returns a
	clustering of the vertices 5, 10, 39, 4588.
	It is important that to include "endClusterVertices".
* *--numEstNorm r*:
	Specifies how many times the algorithms runs the estimateNorm-procedure.
	Default: r=1.
	If r>1, the algorithm runs estimateNorm() r times and then return the
	median.
* *--unsigned*:
	Runs the unsigned version of the oracle (see paper).
* *--biclustering*:
	Runs the biclustering version of the oracle (see paper).
* *--sameCluster u v*:
	Returns whether vertices u and v are in the same cluster.

## Running the Competing Methods
To run the competing method polar by Xiao et al. (WebConf'20), download their
code and put it into /include/signed-local-community-master. The code can be
downloaded from the following link:
>	https://github.com/xiaohan2012/signed-local-community
	
To run the competing method FOCG by Chu et al. (KDD'16), download their
code and put it into /include/KOCG.SIGKDD2016-master. The code can be downloaded
from the following link:
>	https://github.com/lingyangchu/KOCG.SIGKDD2016

## Running the Experiments
To run the experiments, execute implementation/main.py.

