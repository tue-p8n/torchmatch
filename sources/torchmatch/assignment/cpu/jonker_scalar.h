#pragma once

#include <cstdint>

inline constexpr int RECTANGULAR_LSAP_INFEASIBLE = -1;
inline constexpr int RECTANGULAR_LSAP_INVALID = -2;

extern "C" int solve_rectangular_linear_sum_assignment(intptr_t nr, intptr_t nc,
                                                       double *input_cost,
                                                       bool maximize, int64_t *a,
                                                       int64_t *b);
