#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include "graph.h"
using namespace std;

Graph::Graph()
{
    degrees = nullptr;
    neighbors = nullptr;
    weights = nullptr;
    vertex_num = 0;
    edge_num = 0;
    is_weighted = false;
}

Graph::Graph(string file, bool isWt)
{
    // read raw graph
    ifstream fin;
    fin.open(file, fstream::in);
    vector<vector<pair<vertex_t, weight_t>>> edges;
    this->is_weighted = isWt;
    edge_t count = 0;
    string line;
    while (getline(fin,line))
    {
        if (count % 10000000 == 0)
        {
            cerr << ".";
            fflush(stderr);
        }
        if (line.empty() || line[0] == '#') {
            continue; // 
        }
        istringstream iss(line);
        vertex_t src, dest;
        weight_t weight = 1;
        if (is_weighted)
            iss >> src >> dest >> weight;
        else
            iss >> src >> dest;
        
        if (edges.size() <= max(src, dest) + 1)
            edges.resize(max(src, dest) + 1);
        if(!is_weighted && src == dest){
            continue;
        }
        edges[src].push_back(make_pair(dest, weight));
        edges[dest].push_back(make_pair(src, weight));
        count++;
        
    }
    fin.close();

    // deduplicate
    edge_num = 0;
    for (unsigned int i = 0; i < edges.size(); i++)
    {
        map<int, int> m;
        map<int, int>::iterator it;
        for (unsigned int j = 0; j < edges[i].size(); j++)
        {
            it = m.find(edges[i][j].first);
            if (it == m.end())
                m.insert(make_pair(edges[i][j].first, edges[i][j].second));
            else if (is_weighted)
                it->second += edges[i][j].second;
        }
        vector<pair<int, int>> v;
        for (it = m.begin(); it != m.end(); it++)
            v.push_back(*it);
        edges[i].clear();
        edges[i] = v;
        edge_num += v.size();
    }
    edge_num /= 2;
    // renumber,file may skip some vertex number.
    vector<vertex_t> linked(edges.size(), -1);
    vector<vertex_t> renum(edges.size(), -1);
    vertex_num = 0;

    for (edge_t i = 0; i < edges.size(); i++)
    {
        for (edge_t j = 0; j < edges[i].size(); j++)
        {
            linked[i] = 1;
            linked[edges[i][j].first] = 1;
        }
    }

    for (edge_t i = 0; i < edges.size(); i++)
    {
        if (linked[i] == 1)
            renum[i] = vertex_num++;
    }

    for (edge_t i = 0; i < edges.size(); i++)
    {
        if (linked[i] == 1)
        {
            for (edge_t j = 0; j < edges[i].size(); j++)
            {
                edges[i][j].first = renum[edges[i][j].first];
            }
            edges[renum[i]] = edges[i];
        }
    }
    edges.resize(vertex_num);

    cout << "vertex number:" << vertex_num << " edge number:" << edge_num << endl;

    degrees = (edge_t *)malloc(vertex_num * sizeof(edge_t));
    neighbors = (vertex_t *)malloc(edge_num * 2 * sizeof(vertex_t));
    weights = (vertex_t *)malloc(edge_num * 2 * sizeof(vertex_t));
    // convert to binary

    edge_t degree = 0;
    for (vertex_t i = 0; i < edges.size(); i++)
    { // degree
        degree += edges[i].size();
        degrees[i] = degree;
    }

    count = 0;
    for (vertex_t i = 0; i < edges.size(); i++) // neighbors
        for (edge_t j = 0; j < edges[i].size(); j++)
        {
            vertex_t dest = edges[i][j].first;
            neighbors[count] = dest;
            count++;
        }

    if (is_weighted)
    {
        count = 0;
        for (vertex_t i = 0; i < edges.size(); i++)
        {
            for (edge_t j = 0; j < edges[i].size(); j++)
            {
                weight_t weight = edges[i][j].second;
                weights[count] = weight;
            }
        }
    }
    else
    {
        for (edge_t i = 0; i < edge_num * 2; i++)
        {
            weights[i] = 1;
        }
    }
}

void Graph::store_bin_graph(string bin_file)
{
    ofstream fout;
    fout.open(bin_file, fstream::out | fstream::binary);

    fout.write((char *)(&vertex_num), sizeof(vertex_t)); // number of vertex

    for (vertex_t i = 0; i < vertex_num; i++)
    { // degree

        fout.write((char *)(&degrees[i]), sizeof(edge_t));
    }

    for (edge_t i = 0; i < 2 * edge_num; i++) // neighbors
        fout.write((char *)(&neighbors[i]), sizeof(vertex_t));

    if (is_weighted)
    {
        for (edge_t i = 0; i < 2 * edge_num; i++) // neighbors
            fout.write((char *)(&weights[i]), sizeof(vertex_t));
    }
    fout.close();
}

void Graph::load_bin_graph(string bin_file, bool isWt)
{
    ifstream fin;
    // cout<<bin_file<<endl;
    fin.open(bin_file, fstream::in | fstream::binary);
    // csr format
    edge_t *tmp_degrees;
    vertex_t *tmp_neighbors;
    weight_t *tmp_weights;
    // read
    vertex_t tmp_vertex_num;
    fin.read((char *)&tmp_vertex_num, sizeof(vertex_t));

    tmp_degrees = (edge_t *)malloc(tmp_vertex_num * sizeof(edge_t));
    fin.read((char *)tmp_degrees, tmp_vertex_num * sizeof(edge_t));

    // read edges: 4 bytes for each edge (each edge is counted twice)

    edge_num = tmp_degrees[tmp_vertex_num - 1] / 2;
    tmp_neighbors = (vertex_t *)malloc(edge_num * 2 * sizeof(vertex_t));
    fin.read((char *)tmp_neighbors, edge_num * 2 * sizeof(vertex_t));

    // IF WEIGHTED : read weights: 4 bytes for each link (each link is counted twice)
    if (is_weighted)
    {
        tmp_weights = (weight_t *)malloc(edge_num * 2 * sizeof(weight_t));
        fin.read((char *)tmp_weights, edge_num * 2 * sizeof(weight_t));
    }
    else
    {
        tmp_weights = (weight_t *)malloc(edge_num * 2 * sizeof(weight_t));
        for (edge_t i = 0; i < edge_num * 2; i++)
        {
            tmp_weights[i] = 1;
        }
    }
    cout << "vertex number:" << tmp_vertex_num << " edge number:" << edge_num << endl;

    weights = tmp_weights;
    neighbors = tmp_neighbors;
    degrees = tmp_degrees;
    vertex_num = tmp_vertex_num;
}