#pragma once

#include <FMM_tSNE/tsne/solvers/P_matrix/VP_tree/VP_node.h>
#include <FMM_tSNE/tsne/solvers/P_matrix/VP_tree/VP_point.h>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <queue>
#include <random>
#include <vector>

class VPTree
{
  public:
    std::vector<VPNode> m_nodes;
    std::mt19937 gen{std::random_device{}()};

    [[nodiscard]] std::vector<size_t> find_kNN(const std::vector<std::vector<float>>& points, size_t q, size_t k) const
    {
        std::vector<size_t> NNs;
        if (m_nodes.empty() || k == 0) return NNs;

        std::priority_queue<std::pair<float, size_t>> heap;
        search_k(points, 0, q, k, heap);

        NNs.resize(heap.size());
        for (size_t i{heap.size()}; i-- > 0;)
        {
            NNs[i] = heap.top().second;
            heap.pop();
        }
        return NNs;
    }

    void create_VP_tree(const std::vector<std::vector<float>>& points)
    {
        m_nodes.clear();
        if (points.size() == 0) return;

        VPNode root{};
        std::vector<size_t> contained_points(points.size());
        std::ranges::iota(contained_points, 0);
        root.set_vp(points, gen, contained_points);
        root.set_median_distance(points, contained_points);
        m_nodes.push_back(root);

        split_tree(points, 0, contained_points);
    }

  private:
    void split_tree(const std::vector<std::vector<float>>& points, size_t n, std::vector<size_t>& contained_points)
    {
        if (contained_points.size() == 0) return;

        std::vector<size_t> contained_points_l;
        std::vector<size_t> contained_points_r;

        for (size_t p : contained_points)
        {
            if (VPPoint::get_dist(points[p], points[m_nodes[n].p]) < m_nodes[n].mu)
            {
                contained_points_l.push_back(p);
            }
            else
            {
                contained_points_r.push_back(p);
            }
        }

        if (!contained_points_l.empty())
        {
            VPNode node_l{};
            node_l.set_vp(points, gen, contained_points_l);
            node_l.set_median_distance(points, contained_points_l);

            size_t node_l_index = m_nodes.size();
            m_nodes[n].node_l = node_l_index;
            m_nodes.push_back(node_l);

            split_tree(points, node_l_index, contained_points_l);
        }

        if (!contained_points_r.empty())
        {
            VPNode node_r{};
            node_r.set_vp(points, gen, contained_points_r);
            node_r.set_median_distance(points, contained_points_r);

            size_t node_r_index = m_nodes.size();
            m_nodes[n].node_r = node_r_index;
            m_nodes.push_back(node_r);

            split_tree(points, node_r_index, contained_points_r);
        }
    }

    void search_k(const std::vector<std::vector<float>>& points,
                  size_t n,
                  size_t q,
                  size_t k,
                  std::priority_queue<std::pair<float, size_t>>& heap) const
    {
        const VPNode& node = m_nodes[n];

        const float dist = VPPoint::get_dist(points[q], points[node.p]);

        if (node.p != q)
        {
            if (heap.size() < k)
            {
                heap.emplace(dist, node.p);
            }
            else if (dist < heap.top().first)
            {
                heap.pop();
                heap.emplace(dist, node.p);
            }
        }

        auto smallest_dist = [&]()
        { return heap.size() < k ? std::numeric_limits<float>::infinity() : heap.top().first; };

        // Search the side q falls on first, then the other side only if
        // the ball of radius tau around q can reach across mu.
        // left:  d(vp, point) <  mu  -> reachable if x - tau <  mu
        // right: d(vp, point) >= mu  -> reachable if x + tau >= mu
        if (dist < node.mu)
        {
            if (node.node_l != 0 && dist - smallest_dist() < node.mu) search_k(points, node.node_l, q, k, heap);
            if (node.node_r != 0 && dist + smallest_dist() >= node.mu) search_k(points, node.node_r, q, k, heap);
        }
        else
        {
            if (node.node_r != 0 && dist + smallest_dist() >= node.mu) search_k(points, node.node_r, q, k, heap);
            if (node.node_l != 0 && dist - smallest_dist() < node.mu) search_k(points, node.node_l, q, k, heap);
        }
    }
};
