#pragma once

// VP-tree with the paper's "vps^b" ingredients (Yianilos 1993):
//   1. four bounds per node (lo/hi of left AND right subspace) instead of just mu
//   2. ancestor history: every node/point remembers its distance range to its ancestors'
//      vantage points, so whole subtrees / single points can be rejected WITHOUT computing
//      a new distance to them
//   3. bucketed leaves
// Replaces VP_tree.h, VP_node.h and VP_point.h.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <thread>
#include <utility>
#include <vector>

class VPTree
{
  public:
    // How many (topmost) ancestor vantage points each node / bucket point remembers.
    // Memory cost is ~ n * kHistory floats. More = better pruning, up to a point.
    static constexpr size_t kHistory = 8;
    // Max points in a leaf bucket (must be >= 2).
    static constexpr size_t kBucketSize = 16;

    explicit VPTree(uint32_t seed = 12345u)
        : m_gen(seed)
    {
    }

    void create_VP_tree(const std::vector<std::vector<float>>& points)
    {
        m_nodes.clear();
        m_bounds.clear();
        m_bucket_ids.clear();
        m_bucket_hist.clear();
        m_n = points.size();
        if (m_n == 0) return;
        m_dim = points[0].size();

        // flat, contiguous copy of the data (one cache-friendly block instead of n heap vectors)
        m_data.resize(m_n * m_dim);
        for (size_t i = 0; i < m_n; ++i)
            std::copy(points[i].begin(), points[i].end(), m_data.begin() + static_cast<std::ptrdiff_t>(i * m_dim));

        m_hist.assign(m_n * kHistory, 0.0f);
        m_scratch.assign(m_n, 0.0f);
        m_ids.resize(m_n);
        std::iota(m_ids.begin(), m_ids.end(), uint32_t{0});
        m_nodes.reserve(2 * m_n / kBucketSize + 16);

        build(0, m_n, 0);

        // build-only memory
        std::vector<float>().swap(m_hist);
        std::vector<float>().swap(m_scratch);
    }

    // k nearest neighbours of database point q (q itself excluded), nearest first.
    [[nodiscard]] std::vector<size_t> find_kNN(size_t q, size_t k) const
    {
        std::vector<size_t> out;
        if (m_nodes.empty() || k == 0 || q >= m_n) return out;

        Search s(*this, point(static_cast<uint32_t>(q)), static_cast<uint32_t>(q), k);
        s.visit(0, 0);

        out.resize(s.heap.size());
        for (size_t i = out.size(); i-- > 0;)
        {
            out[i] = s.heap.top().second;
            s.heap.pop();
        }
        return out;
    }

    // Same search, but writes ids and (non-squared) Euclidean distances, nearest first, into
    // caller-owned buffers: no allocation per query and no need to recompute distances afterwards.
    void find_kNN(size_t q, size_t k, std::vector<uint32_t>& ids, std::vector<float>& dists) const
    {
        ids.clear();
        dists.clear();
        if (m_nodes.empty() || k == 0 || q >= m_n) return;

        Search s(*this, point(static_cast<uint32_t>(q)), static_cast<uint32_t>(q), k);
        s.visit(0, 0);

        ids.resize(s.heap.size());
        dists.resize(s.heap.size());
        for (size_t i = ids.size(); i-- > 0;)
        {
            dists[i] = s.heap.top().first;
            ids[i] = s.heap.top().second;
            s.heap.pop();
        }
    }

    // Point ids in tree order. Querying in this order keeps consecutive searches in similar
    // parts of the tree (better cache behaviour).
    [[nodiscard]] const std::vector<uint32_t>& tree_order() const { return m_ids; }

    // kNN for every point, computed in parallel with std::jthread.
    // Searches are const and independent, so the only shared state is an atomic work counter
    // and each thread writes to different res[q] slots. n_threads == 0 -> use all hardware threads.
    // Points are handed out in tree order (in chunks), so consecutive queries touch similar memory.
    [[nodiscard]] std::vector<std::vector<size_t>> find_kNN_all(size_t k, size_t n_threads = 0) const
    {
        std::vector<std::vector<size_t>> res(m_n);
        if (m_n == 0) return res;

        constexpr size_t chunk = 256;
        if (n_threads == 0) n_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
        n_threads = std::min(n_threads, (m_n + chunk - 1) / chunk);

        std::atomic<size_t> next{0};
        auto worker = [&]
        {
            for (;;)
            {
                const size_t begin = next.fetch_add(chunk, std::memory_order_relaxed);
                if (begin >= m_n) return;
                const size_t end = std::min(begin + chunk, m_n);
                for (size_t j = begin; j < end; ++j)
                {
                    const size_t q = m_ids[j];
                    res[q] = find_kNN(q, k);
                }
            }
        };

        {
            std::vector<std::jthread> threads;
            threads.reserve(n_threads - 1);
            for (size_t i = 1; i < n_threads; ++i)
                threads.emplace_back(worker);
            worker(); // the calling thread works too
        } // jthreads join automatically here

        return res;
    }

  private:
    static constexpr float kInf = std::numeric_limits<float>::infinity();

    struct Node
    {
        // internal node: vantage point + bounds of the distances d(vp, x) of each child
        uint32_t vp{};
        uint32_t left{};    // internal: left child.  leaf: bucket begin
        uint32_t right{};   // internal: right child. leaf: bucket end
        float ll{}, lh{};   // [min,max] of d(vp,x) over left subtree
        float rl{}, rh{};   // [min,max] of d(vp,x) over right subtree
        uint32_t bnd_off{}; // offset into m_bounds: [lo,hi] per remembered ancestor, whole subtree
        bool is_leaf{false};
    };

    std::vector<Node> m_nodes;
    std::vector<float> m_bounds;        // ancestor ranges of every node
    std::vector<uint32_t> m_bucket_ids; // leaf points
    std::vector<float> m_bucket_hist;   // kHistory floats per leaf point: d(ancestor_l, point)
    std::vector<float> m_data;          // flat points
    std::vector<uint32_t> m_ids;        // permutation used while building (tree order afterwards)
    size_t m_n{}, m_dim{};
    std::mt19937 m_gen;

    // build-only scratch
    std::vector<float> m_hist;    // [id * kHistory + level] = d(vp at level, point id)
    std::vector<float> m_scratch; // [id] = distance to current vp

    const float* point(uint32_t id) const { return m_data.data() + static_cast<size_t>(id) * m_dim; }

    float dist(const float* a, const float* b) const
    {
        float s = 0.0f;
        for (size_t i = 0; i < m_dim; ++i)
        {
            const float d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }

    // squared distance, but gives up (returns +inf) as soon as it is clear it is >= limit2
    float dist2_bounded(const float* a, const float* b, float limit2) const
    {
        float s = 0.0f;
        size_t i = 0;
        for (; i + 8 <= m_dim; i += 8)
        {
            for (size_t j = 0; j < 8; ++j)
            {
                const float d = a[i + j] - b[i + j];
                s += d * d;
            }
            if (s >= limit2) return kInf;
        }
        for (; i < m_dim; ++i)
        {
            const float d = a[i] - b[i];
            s += d * d;
        }
        return s >= limit2 ? kInf : s;
    }

    // Pick the candidate with the largest spread of distances (as in the paper), move it to m_ids[lo].
    void select_vp(size_t lo, size_t hi)
    {
        const size_t n = hi - lo;
        const size_t n_cand = std::min<size_t>(10, n);
        const size_t n_eval = std::min<size_t>(10, n);
        std::uniform_int_distribution<size_t> dis(lo, hi - 1);

        std::array<size_t, 10> eval_pos{};
        for (size_t e = 0; e < n_eval; ++e)
            eval_pos[e] = dis(m_gen);

        size_t best_pos = lo;
        float best_spread = -1.0f;
        std::array<float, 10> d{}, tmp{};
        for (size_t c = 0; c < n_cand; ++c)
        {
            const size_t pos = dis(m_gen);
            const float* cp = point(m_ids[pos]);
            for (size_t e = 0; e < n_eval; ++e)
                d[e] = dist(cp, point(m_ids[eval_pos[e]]));

            tmp = d;
            std::nth_element(tmp.begin(),
                             tmp.begin() + static_cast<std::ptrdiff_t>(n_eval / 2),
                             tmp.begin() + static_cast<std::ptrdiff_t>(n_eval));
            const float median = tmp[n_eval / 2];
            float spread = 0.0f;
            for (size_t e = 0; e < n_eval; ++e)
                spread += (d[e] - median) * (d[e] - median);

            if (spread > best_spread)
            {
                best_spread = spread;
                best_pos = pos;
            }
        }
        std::swap(m_ids[lo], m_ids[best_pos]);
    }

    // Builds the subtree for m_ids[lo, hi) and returns its node index.
    uint32_t build(size_t lo, size_t hi, size_t depth)
    {
        const size_t n = hi - lo;
        const size_t nlev = std::min(depth, kHistory);

        const auto idx = static_cast<uint32_t>(m_nodes.size());
        m_nodes.emplace_back();
        m_nodes[idx].bnd_off = static_cast<uint32_t>(m_bounds.size());

        // range of distances to each remembered ancestor, over everything in this subtree
        for (size_t l = 0; l < nlev; ++l)
        {
            float mn = kInf, mx = -kInf;
            for (size_t i = lo; i < hi; ++i)
            {
                const float h = m_hist[static_cast<size_t>(m_ids[i]) * kHistory + l];
                mn = std::min(mn, h);
                mx = std::max(mx, h);
            }
            m_bounds.push_back(mn);
            m_bounds.push_back(mx);
        }

        if (n <= kBucketSize)
        {
            m_nodes[idx].is_leaf = true;
            m_nodes[idx].left = static_cast<uint32_t>(m_bucket_ids.size());
            for (size_t i = lo; i < hi; ++i)
            {
                const uint32_t id = m_ids[i];
                m_bucket_ids.push_back(id);
                const float* h = m_hist.data() + static_cast<size_t>(id) * kHistory;
                m_bucket_hist.insert(m_bucket_hist.end(), h, h + kHistory);
            }
            m_nodes[idx].right = static_cast<uint32_t>(m_bucket_ids.size());
            return idx;
        }

        select_vp(lo, hi);
        const uint32_t p = m_ids[lo];
        const float* pv = point(p);

        // ONE distance per point per level; reused for the split, the bounds and the history
        for (size_t i = lo + 1; i < hi; ++i)
        {
            const uint32_t id = m_ids[i];
            const float d = dist(pv, point(id));
            m_scratch[id] = d;
            if (depth < kHistory) m_hist[static_cast<size_t>(id) * kHistory + depth] = d;
        }

        // exact median split: always balanced, even with many duplicate points
        const size_t mid = lo + 1 + (n - 1) / 2;
        std::nth_element(m_ids.begin() + static_cast<std::ptrdiff_t>(lo + 1),
                         m_ids.begin() + static_cast<std::ptrdiff_t>(mid),
                         m_ids.begin() + static_cast<std::ptrdiff_t>(hi),
                         [&](uint32_t a, uint32_t b) { return m_scratch[a] < m_scratch[b]; });

        float ll = kInf, lh = -kInf, rl = kInf, rh = -kInf;
        for (size_t i = lo + 1; i < mid; ++i)
        {
            ll = std::min(ll, m_scratch[m_ids[i]]);
            lh = std::max(lh, m_scratch[m_ids[i]]);
        }
        for (size_t i = mid; i < hi; ++i)
        {
            rl = std::min(rl, m_scratch[m_ids[i]]);
            rh = std::max(rh, m_scratch[m_ids[i]]);
        }

        m_nodes[idx].vp = p;
        m_nodes[idx].ll = ll;
        m_nodes[idx].lh = lh;
        m_nodes[idx].rl = rl;
        m_nodes[idx].rh = rh;

        // (no references into m_nodes are held across these calls: it may reallocate)
        const uint32_t l = build(lo + 1, mid, depth + 1);
        m_nodes[idx].left = l;
        const uint32_t r = build(mid, hi, depth + 1);
        m_nodes[idx].right = r;
        return idx;
    }

    struct Search
    {
        const VPTree& t;
        const float* q;
        uint32_t qid;
        size_t k;
        std::priority_queue<std::pair<float, uint32_t>> heap;
        std::array<float, kHistory> qd{}; // qd[l] = d(q, vantage point at depth l) on the current path

        Search(const VPTree& tree, const float* query, uint32_t id, size_t kk)
            : t(tree),
              q(query),
              qid(id),
              k(kk)
        {
        }

        float tau() const { return heap.size() < k ? kInf : heap.top().first; }

        void offer(float d, uint32_t id)
        {
            if (heap.size() < k)
                heap.emplace(d, id);
            else if (d < heap.top().first)
            {
                heap.pop();
                heap.emplace(d, id);
            }
        }

        void visit(uint32_t ni, size_t depth)
        {
            const Node& nd = t.m_nodes[ni];
            const size_t nlev = std::min(depth, kHistory);

            // (vps) Reject the whole subtree using only distances we already know.
            // No new distance computation needed.
            const float* b = t.m_bounds.data() + nd.bnd_off;
            {
                const float tau_now = tau();
                for (size_t l = 0; l < nlev; ++l)
                    if (qd[l] - b[2 * l + 1] >= tau_now || b[2 * l] - qd[l] >= tau_now) return;
            }

            if (nd.is_leaf)
            {
                for (uint32_t i = nd.left; i < nd.right; ++i)
                {
                    const uint32_t id = t.m_bucket_ids[i];
                    if (id == qid) continue;

                    // (vps^b) cheap lower bound |d(q,a) - d(x,a)| <= d(q,x) for every ancestor a
                    const float* h = t.m_bucket_hist.data() + static_cast<size_t>(i) * kHistory;
                    const float tau_now = tau();
                    bool skip = false;
                    for (size_t l = 0; l < nlev; ++l)
                        if (std::fabs(qd[l] - h[l]) >= tau_now)
                        {
                            skip = true;
                            break;
                        }
                    if (skip) continue;

                    const float d2 = t.dist2_bounded(q, t.point(id), tau_now * tau_now);
                    if (d2 < kInf) offer(std::sqrt(d2), id);
                }
                return;
            }

            const float x = t.dist(q, t.point(nd.vp));
            if (depth < kHistory) qd[depth] = x;
            if (nd.vp != qid) offer(x, nd.vp);

            // left reachable iff some d in [ll,lh] has |x - d| < tau (same for right)
            auto go_left = [&]
            {
                if (x - nd.lh < tau() && nd.ll - x < tau()) visit(nd.left, depth + 1);
            };
            auto go_right = [&]
            {
                if (x - nd.rh < tau() && nd.rl - x < tau()) visit(nd.right, depth + 1);
            };

            // heuristic from the paper: descend first into the side q is closer to
            if (x < 0.5f * (nd.lh + nd.rl))
            {
                go_left();
                go_right();
            }
            else
            {
                go_right();
                go_left();
            }
        }
    };
};

// #pragma once
//
// #include <FMM_tSNE/tsne/solvers/P_matrix/VP_tree/VP_node.h>
// #include <FMM_tSNE/tsne/solvers/P_matrix/VP_tree/VP_point.h>
// #include <cstddef>
// #include <cstdlib>
// #include <limits>
// #include <queue>
// #include <random>
// #include <vector>
//
// class VPTree
// {
//   public:
//     std::vector<VPNode> m_nodes;
//     std::mt19937 gen{std::random_device{}()};
//
//     [[nodiscard]] std::vector<size_t> find_kNN(const std::vector<std::vector<float>>& points, size_t q, size_t k)
//     const
//     {
//         std::vector<size_t> NNs;
//         if (m_nodes.empty() || k == 0) return NNs;
//
//         std::priority_queue<std::pair<float, size_t>> heap;
//         search_k(points, 0, q, k, heap);
//
//         NNs.resize(heap.size());
//         for (size_t i{heap.size()}; i-- > 0;)
//         {
//             NNs[i] = heap.top().second;
//             heap.pop();
//         }
//         return NNs;
//     }
//
//     void create_VP_tree(const std::vector<std::vector<float>>& points)
//     {
//         m_nodes.clear();
//         if (points.size() == 0) return;
//
//         VPNode root{};
//         std::vector<size_t> contained_points(points.size());
//         std::ranges::iota(contained_points, 0);
//         root.set_vp(points, gen, contained_points);
//         root.set_median_distance(points, contained_points);
//         m_nodes.push_back(root);
//
//         split_tree(points, 0, contained_points);
//     }
//
//   private:
//     void split_tree(const std::vector<std::vector<float>>& points, size_t n, std::vector<size_t>& contained_points)
//     {
//         if (contained_points.size() == 0) return;
//
//         std::vector<size_t> contained_points_l;
//         std::vector<size_t> contained_points_r;
//
//         for (size_t p : contained_points)
//         {
//             if (VPPoint::get_dist(points[p], points[m_nodes[n].p]) < m_nodes[n].mu)
//             {
//                 contained_points_l.push_back(p);
//             }
//             else
//             {
//                 contained_points_r.push_back(p);
//             }
//         }
//
//         if (!contained_points_l.empty())
//         {
//             VPNode node_l{};
//             node_l.set_vp(points, gen, contained_points_l);
//             node_l.set_median_distance(points, contained_points_l);
//
//             size_t node_l_index = m_nodes.size();
//             m_nodes[n].node_l = node_l_index;
//             m_nodes.push_back(node_l);
//
//             split_tree(points, node_l_index, contained_points_l);
//         }
//
//         if (!contained_points_r.empty())
//         {
//             VPNode node_r{};
//             node_r.set_vp(points, gen, contained_points_r);
//             node_r.set_median_distance(points, contained_points_r);
//
//             size_t node_r_index = m_nodes.size();
//             m_nodes[n].node_r = node_r_index;
//             m_nodes.push_back(node_r);
//
//             split_tree(points, node_r_index, contained_points_r);
//         }
//     }
//
//     void search_k(const std::vector<std::vector<float>>& points,
//                   size_t n,
//                   size_t q,
//                   size_t k,
//                   std::priority_queue<std::pair<float, size_t>>& heap) const
//     {
//         const VPNode& node = m_nodes[n];
//
//         const float dist = VPPoint::get_dist(points[q], points[node.p]);
//
//         if (node.p != q)
//         {
//             if (heap.size() < k)
//             {
//                 heap.emplace(dist, node.p);
//             }
//             else if (dist < heap.top().first)
//             {
//                 heap.pop();
//                 heap.emplace(dist, node.p);
//             }
//         }
//
//         auto smallest_dist = [&]()
//         { return heap.size() < k ? std::numeric_limits<float>::infinity() : heap.top().first; };
//
//         // Search the side q falls on first, then the other side only if
//         // the ball of radius tau around q can reach across mu.
//         // left:  d(vp, point) <  mu  -> reachable if x - tau <  mu
//         // right: d(vp, point) >= mu  -> reachable if x + tau >= mu
//         if (dist < node.mu)
//         {
//             if (node.node_l != 0 && dist - smallest_dist() < node.mu) search_k(points, node.node_l, q, k, heap);
//             if (node.node_r != 0 && dist + smallest_dist() >= node.mu) search_k(points, node.node_r, q, k, heap);
//         }
//         else
//         {
//             if (node.node_r != 0 && dist + smallest_dist() >= node.mu) search_k(points, node.node_r, q, k, heap);
//             if (node.node_l != 0 && dist - smallest_dist() < node.mu) search_k(points, node.node_l, q, k, heap);
//         }
//     }
// };
