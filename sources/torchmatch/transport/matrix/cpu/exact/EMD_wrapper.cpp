#include "EMD.h"
#include "network_simplex_simple.h"
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace {

struct SetupPolicy {
    bool full_support;
    bool use_arc_mixing;
    bool use_dense_cost_pointer;
};

inline SetupPolicy make_setup_policy(uint64_t n, uint64_t m, int n1, int n2,
                                     bool dense_cost_pointer_supported) {
    SetupPolicy policy;
    policy.full_support =
        (n == static_cast<uint64_t>(n1)) && (m == static_cast<uint64_t>(n2));
    policy.use_arc_mixing = !policy.full_support;
    policy.use_dense_cost_pointer = dense_cost_pointer_supported && policy.full_support;
    return policy;
}

template <typename NetType, typename DigraphType>
inline void setup_explicit_arc_costs(NetType &net, DigraphType &di, const double *D,
                                     int n2, const std::vector<uint64_t> &indI,
                                     const std::vector<uint64_t> &indJ, uint64_t n,
                                     uint64_t m) {
    int64_t idarc = 0;
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < m; ++j) {
            net.setCost(di.arcFromId(idarc), D[indI[i] * n2 + indJ[j]]);
            ++idarc;
        }
    }
}

template <typename NetType>
inline void
setup_warmstart_potentials(NetType &net, const double *alpha_init,
                           const double *beta_init, const std::vector<uint64_t> &indI,
                           const std::vector<uint64_t> &indJ, uint64_t n, uint64_t m) {
    if (alpha_init == nullptr || beta_init == nullptr)
        return;
    std::vector<double> alpha_compressed(n);
    std::vector<double> beta_compressed(m);
    for (uint64_t i = 0; i < n; ++i)
        alpha_compressed[i] = alpha_init[indI[i]];
    for (uint64_t j = 0; j < m; ++j)
        beta_compressed[j] = beta_init[indJ[j]];
    net.setWarmstartPotentials(&alpha_compressed[0], &beta_compressed[0], (int)n,
                               (int)m);
}

template <typename NetType>
inline void extract_dense_full_support(const NetType &net, const double *D, double *G,
                                       double *alpha, double *beta, double *cost,
                                       uint64_t n, uint64_t m) {
    const int node_total = net.nodeNum();
    const int pi_base = node_total - 1;

    for (uint64_t ii = 0; ii < n; ++ii) {
        alpha[ii] = -net._pi[pi_base - static_cast<int>(ii)];
    }
    for (uint64_t jj = 0; jj < m; ++jj) {
        beta[jj] = net._pi[pi_base - static_cast<int>(n + jj)];
    }

    // Only write non-zero entries. G is already zero-initialized in Python.
    const int64_t arc_total = net.arcNum();
    for (int64_t a = 0; a < arc_total; ++a) {
        const double flow = net._flow[a];
        if (flow == 0.0)
            continue;
        const int64_t d_idx = arc_total - a - 1; // row-major index in D/G
        *cost += flow * D[d_idx];
        G[d_idx] = flow;
    }
}

template <typename NetType, typename DigraphType, typename InvalidType>
inline void
extract_compressed_support(const NetType &net, DigraphType &di, InvalidType invalid,
                           const double *D, double *G, double *alpha, double *beta,
                           double *cost, const std::vector<uint64_t> &indI,
                           const std::vector<uint64_t> &indJ, uint64_t n, int n2) {
    for (uint64_t ii = 0; ii < n; ++ii) {
        alpha[indI[ii]] = -net.potential(ii);
    }
    for (uint64_t jj = 0; jj < indJ.size(); ++jj) {
        beta[indJ[jj]] = net.potential(jj + n);
    }

    uint64_t i, j;
    typename DigraphType::Arc a;
    di.first(a);
    for (; a != invalid; di.next(a)) {
        i = di.source(a);
        j = di.target(a);
        const double flow = net.flow(a);
        *cost += flow * D[indI[i] * n2 + indJ[j - n]];
        G[indI[i] * n2 + indJ[j - n]] = flow;
    }
}

} // namespace

int EMD_wrap(int n1, int n2, double *X, double *Y, double *D, double *G, double *alpha,
             double *beta, double *cost, uint64_t maxIter, double *alpha_init,
             double *beta_init) {
    using namespace lemon;
    uint64_t n, m, cur;

    typedef FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(Digraph);

    n = 0;
    for (int i = 0; i < n1; i++) {
        double val = *(X + i);
        if (val > 0) {
            n++;
        } else if (val < 0) {
            return INFEASIBLE;
        }
    }
    m = 0;
    for (int i = 0; i < n2; i++) {
        double val = *(Y + i);
        if (val > 0) {
            m++;
        } else if (val < 0) {
            return INFEASIBLE;
        }
    }

    std::vector<uint64_t> indI(n), indJ(m);
    std::vector<double> weights1(n), weights2(m);
    Digraph di(n, m);
    const SetupPolicy policy = make_setup_policy(n, m, n1, n2, true);
    NetworkSimplexSimple<Digraph, double, double, node_id_type> net(
        di, policy.use_arc_mixing, (int)(n + m), n * m, maxIter);

    cur = 0;
    for (uint64_t i = 0; i < n1; i++) {
        double val = *(X + i);
        if (val > 0) {
            weights1[cur] = val;
            indI[cur++] = i;
        }
    }

    // Demand is negative supply in the network-simplex convention.
    cur = 0;
    for (uint64_t i = 0; i < n2; i++) {
        double val = *(Y + i);
        if (val > 0) {
            weights2[cur] = -val;
            indJ[cur++] = i;
        }
    }

    net.supplyMap(&weights1[0], (int)n, &weights2[0], (int)m);

    if (policy.use_dense_cost_pointer) {
        net.setDenseCostMatrix(D, n2);
    } else {
        setup_explicit_arc_costs(net, di, D, n2, indI, indJ, n, m);
    }
    setup_warmstart_potentials(net, alpha_init, beta_init, indI, indJ, n, m);

    int ret = net.run();

    if (ret == (int)net.OPTIMAL || ret == (int)net.MAX_ITER_REACHED) {
        *cost = 0;
        if (policy.full_support) {
            extract_dense_full_support(net, D, G, alpha, beta, cost, n, m);
        } else {
            extract_compressed_support(net, di, INVALID, D, G, alpha, beta, cost, indI,
                                       indJ, n, n2);
        }
    }
    return ret;
}
