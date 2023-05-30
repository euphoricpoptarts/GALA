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

    double curMod = louvain_gpu(g, community, threshold, pruning);

    double end = get_time();

    printf("elapsed time = %fms\n", end - start);

    if(!output_file.empty())
        save_community(output_file,community, g.vertex_num);
    delete []community;
}
