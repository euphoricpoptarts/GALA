#include <unistd.h>
#include <iostream>
#include <filesystem>
#include "graph/graph.h"


int main(int argc, char* argv[]){
    int opt;
    std::string input_file;
    bool is_weighted = false;
    std::string out_dir;
    int permutation=0;//0:not change 1:sort by degree 2:random 
    while ((opt = getopt(argc, argv, "f:wo:")) != -1) {
        switch (opt) {
            case 'f':
                input_file = optarg;
                break;
            case 'w':
                is_weighted = true;
                break;
            case 'o':
                out_dir = optarg;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -f <filename> [-w -o <output_dirname>]" << std::endl;
                return 1;
        }
    }
    if (input_file.empty()) {
        std::cout << "Input file not specified. Usage: " << argv[0] << " -f <filename> [-w -o <output_dirname>]" << std::endl;
        return 1;
    }
    std::string output_file;
    
    if(!out_dir.empty()){
        std::filesystem::path filePath = input_file;
        output_file = out_dir + "/" + filePath.filename().string() + ".bin";
    }
    else{
        output_file = input_file + ".bin";
    }
    std::cout<<output_file<<std::endl;
    Graph graph(input_file,is_weighted);
    graph.store_bin_graph(output_file);
    std::cout<<"store successfully. "<<std::endl;
    return 0;
}
