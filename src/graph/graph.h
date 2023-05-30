#pragma once

#include<vector>
#include<string>
#include "graph_config.h"

struct Graph{
    edge_t* degrees;
    vertex_t* neighbors;
    weight_t* weights;
    vertex_t vertex_num;
    edge_t edge_num;
    bool is_weighted;
    
    Graph();
    Graph(std::string file, bool is_wt);
    void store_bin_graph(std::string bin_file);
    void load_bin_graph(std::string bin_file, bool is_wt);
};