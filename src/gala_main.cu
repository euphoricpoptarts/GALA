#include <unistd.h>
#include <sys/time.h>
#include <fstream>
#include "graph/graph.h"
#include "louvain_gpu/louvain.cuh"
using namespace std;

void save_community(string file, vertex_t *community, int vertex_num){
    std::ofstream outfile(file);
    if (!outfile) {
        std::cerr << "Error opening file for writing." << std::endl;
        return ;
    }
    for (size_t i = 0; i < vertex_num; ++i) {
        outfile << i+1 << "\t" << community[i]+1 << std::endl;
    }
    outfile.close();
    cout<<"write finished"<<endl;
}

double verify_modularity2(const Graph& g, vertex_t* community, int vertex_num) {
    vector<int> comm_size(vertex_num, 0);
    int uncutsize = 0, cutsize = 0;
    for(int i = 0; i < vertex_num; i++){
        int start = 0;
        if(i > 0) start = g.degrees[i-1];
        int end = g.degrees[i];
        for(int j = start; j < end; j++){
            int v = g.neighbors[j];
            if(community[i] == community[v]) uncutsize += g.weights[j];
            else cutsize += g.weights[j];
        }
        int deg = end - start;
        comm_size[community[i]] += deg;
    }
    int g_deg = cutsize + uncutsize;
    double mod = 0;
    int tdeg = 0;
    for(int i = 0; i < vertex_num; i++){
        double cs = comm_size[i];
        tdeg += cs;
        mod += cs*cs;
    }
    mod /= g_deg;
    mod = (uncutsize - mod) / g_deg;
    return mod;
}

int main(int argc, char **argv)
{
    string file_name;
    int is_weighted = 0;
    string output_file;
    int pruning = 0;//0:MG 1:RM 2:Vite 3:MG+RM
    double threshold = 0.000001;
    static const char *opt_string = "f:wo:p:t:";
    int opt = getopt(argc, argv, opt_string);
    while (opt != -1)
    {
        switch (opt)
        {
            case 'f':
                file_name = optarg;
                break;
            case 'w':
                is_weighted = true;
                break; 
            case 'o':
                output_file = optarg;
                break;
            case 'p':
                pruning = stoi(optarg);
                break;
            case 't':
                threshold = stod(optarg);
                break;
        }
        opt = getopt(argc, argv, opt_string);
    }

    double start = get_time();

    Graph g;
    g.load_bin_graph(file_name, is_weighted);

    cout << "load success" << endl;

    vertex_t *community = new vertex_t[g.vertex_num];

    for(int i = 0; i < 21; i++){
        double curMod = louvain_gpu(g, community, threshold, pruning);
    }

    double end = get_time();

    printf("elapsed time = %fms\n", end - start);
    // double v_mod = verify_modularity2(g, community, g.vertex_num);
    // cout << "Verify modularity: " << v_mod << endl;

    if(!output_file.empty())
        save_community(output_file,community, g.vertex_num);
    delete []community;
}
