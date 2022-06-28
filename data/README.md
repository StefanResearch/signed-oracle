This directory contains three datasets:
	- wiki_large containing the dataset Wiki-L from the paper.
	- wiki_medium containing the dataset Wiki-M from the paper.
	- wiki_small containing the dataset Wiki-S from the paper.

Each of the three subdirectory contains three files:
	- partyToId.txt:
		This is a csv-file with separator '#'.
		* In the first column, it contains the ID of a party.
		* In the second column, it contains the name of the party.
		
		Note that party IDs can be positive and negative.
		When two party distinct IDs have the same absolute value this indicates
		that these parties are from the same country.
		For example, Democrats have party ID 5 while the Republicans (the
		opponents of the Democrats in the US) have party ID -5.

	- vtxToPage.txt:
		This is a csv-file with separator '#'.
		* In the first column, it contains the ID of the node. This is a
		  positive integer.
		* In the second column, it contains the name of the corresponding
		  Wikipedia page.
		* In the third column, it contains the party ID of the corresponding
		  page if it is available. If no party ID is available, the third column
		  is left empty. This corresponds to the ground-truth labels of the
		  vertices.

	- edges.txt:
		This is a csv-file with separator '#'.
		The file encodes the edges (u,v) of the graph.
		* In the first column, it contains the node ID of vertex u.
		* In the second column, it contains the node ID of vertex v.
		* In the third column, it contains the edge sign (either 1 or -1).

