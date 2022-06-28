/*
 * This implementation of a union--find data structure is taken from
 * https://stackoverflow.com/questions/43561722/is-the-union-find-or-disjoint-set-data-structure-in-stl
 * and has been slightly modified.
 */

#include <vector>

struct DisjointSet {
	std::map<long,long> parent;
	std::map<long,long> size;
	long numComponents;

    DisjointSet(std::vector<long> elements) {
		numComponents = elements.size();

        for (size_t i = 0; i < elements.size(); i++) {
			long element = elements[i];
            parent[element] = element;
            size[element] = 1;
        }
    }

    int find_set(long v) {
        if (v == parent[v]) {
            return v;
		}

		parent[v] = find_set(parent[v]);
        return parent[v];
    }

    void union_set(long a, long b) {
        a = find_set(a);
        b = find_set(b);
        if (a != b) {
			// make sure that size[a] >= size[b]
            if (size[a] < size[b]){
				// swap a and b
				long z = a;
				a = b;
				b = z;
			}

            parent[b] = a;
            size[a] += size[b];
			numComponents -= 1;
        }
    }
};

