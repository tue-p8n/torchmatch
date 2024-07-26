// C++ wrapper for computing the transportation cost between two vectors
// given a cost matrix.

#ifndef EMD_H
#define EMD_H

#include <cstdint>
#include <iostream>
#include <vector>

typedef unsigned int node_id_type;

enum ProblemType { INFEASIBLE, OPTIMAL, UNBOUNDED, MAX_ITER_REACHED };

int EMD_wrap(int n1, int n2, double *X, double *Y, double *D, double *G, double *alpha,
             double *beta, double *cost, uint64_t maxIter, double *alpha_init,
             double *beta_init);

#endif
