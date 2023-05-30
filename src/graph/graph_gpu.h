#pragma once
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include "graph.h"

struct GraphGPU
{
    thrust::device_vector<edge_t> d_degrees;
    thrust::device_vector<weight_t> d_weights;
    thrust::device_vector<vertex_t> d_neighbors;
    vertex_t vertex_num;
    edge_t edge_num;
    GraphGPU(Graph g) : d_degrees(g.degrees, g.degrees + g.vertex_num),
                        d_neighbors(g.neighbors, g.neighbors + g.edge_num * 2),
                        d_weights(g.weights, g.weights + g.edge_num * 2)
    {
        vertex_num = g.vertex_num;
        edge_num = g.edge_num;
    }
};
