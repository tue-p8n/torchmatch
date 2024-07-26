// JV (rectangular) solver: public entry.
//
// Cost matrix is row-major flat; outputs are int64 row-to-col / col-to-row
// mappings of length n. The algorithm is the Jonker-Volgenant
// shortest-augmenting-path procedure.

#pragma once
#include <cstdint>

namespace match_jonker::detail {

// Returns 0 on success, -1 on allocation failure.
int jonker_dense_solve_square(const double *cost_flat, int n, int64_t *row_to_col,
                              int64_t *col_to_row);

int jonker_dense_solve_square_f32(const float *cost_flat, int n, int64_t *row_to_col,
                                  int64_t *col_to_row);

} // namespace match_jonker::detail
