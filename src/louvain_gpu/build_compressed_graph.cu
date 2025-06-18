#include "louvain.cuh"
#include "kernel_functions.cuh"
#include <algorithm>
using namespace std;

int build_compressed_graph(thrust::device_vector<weight_t> &d_weights, thrust::device_vector<vertex_t> &d_neighbors, thrust::device_vector<edge_t> &degrees,
                           thrust::device_vector<vertex_t> &community_result, thrust::device_vector<int> &primes, int vertex_num)
{
    // renumber

    // build
    thrust::device_vector<int> size_of_each_community(vertex_num, 0); // vertex number of community
    int block_num = (vertex_num + 1024 - 1) / 1024;
    get_size_of_each_community<<<block_num, 1024>>>(thrust::raw_pointer_cast(community_result.data()), thrust::raw_pointer_cast(size_of_each_community.data()), vertex_num);
    cudaDeviceSynchronize();
    degree_filter_in_range filter0(1, vertex_num);
    int community_num = thrust::count_if(thrust::device, size_of_each_community.begin(), size_of_each_community.end(), filter0);

    thrust::device_vector<int> community(vertex_num, 0);
    thrust::transform(thrust::device, size_of_each_community.begin(), size_of_each_community.end(), community.begin(), degree_filter_greater_than(1));
    thrust::exclusive_scan(thrust::device, community.begin(), community.end(), community.begin()); // renumber from 0

    block_num = (vertex_num + 1024 - 1) / 1024;
    renumber<<<block_num, 1024>>>(thrust::raw_pointer_cast(community.data()), thrust::raw_pointer_cast(community_result.data()), vertex_num);
    community.clear();

    thrust::device_vector<int> pos_offset(community_num + 1, 0);

    thrust::copy_if(size_of_each_community.begin(), size_of_each_community.end(), pos_offset.begin() + 1, degree_filter_greater_than(1));
    thrust::inclusive_scan(thrust::device, pos_offset.begin(), pos_offset.end(), pos_offset.begin()); // pos_offset[i+1]-pos_offset[i]=size of comminity i
    size_of_each_community.clear();

    thrust::device_vector<int> gathered_vertex_ptr(pos_offset);          // save a copy
    thrust::device_vector<int> gathered_vertex_by_community(vertex_num); //{veretexs in com0},{veretexs in com1},...
    gather_vertex_by_community<<<block_num, 1024>>>(thrust::raw_pointer_cast(gathered_vertex_by_community.data()), thrust::raw_pointer_cast(community_result.data()),
                                                    thrust::raw_pointer_cast(pos_offset.data()), vertex_num); //{vertexs{com-0},vertexs{com-1},...}
    cudaDeviceSynchronize();
    pos_offset.clear();

    thrust::device_vector<edge_t> edge_num_of_community(community_num); // count edge number in community
    block_num = (community_num * WARP_SIZE + MAX_THREAD_PER_BLOCK - 1) / MAX_THREAD_PER_BLOCK;
    thrust::device_vector<edge_t> d_degrees(degrees.size() + 1, 0);
    thrust::copy(degrees.begin(), degrees.end(), d_degrees.begin() + 1);
    compute_edge_num_of_each_community<<<block_num, MAX_THREAD_PER_BLOCK>>>(thrust::raw_pointer_cast(d_degrees.data()), thrust::raw_pointer_cast(gathered_vertex_by_community.data()),
                                                                            thrust::raw_pointer_cast(gathered_vertex_ptr.data()), thrust::raw_pointer_cast(edge_num_of_community.data()), community_num, WARP_SIZE);
    cudaDeviceSynchronize();


    int global_limit = SHARE_MEM_SIZE * 2 / 3 - 1;

    degree_filter_in_range filter_for_warp(0, 127);
    degree_filter_in_range filter_for_block_shr(128, global_limit);
    degree_filter_in_range filter_for_block_glb(global_limit + 1, max(65536, community_num)-1);
    degree_filter_greater_than filter_for_grid_glb(max(65536, community_num));
    
    int deg_num_warp = thrust::count_if(thrust::device, edge_num_of_community.begin(), edge_num_of_community.end(), filter_for_warp);
    int deg_num_shr = thrust::count_if(thrust::device, edge_num_of_community.begin(), edge_num_of_community.end(), filter_for_block_shr);
    int deg_num_glb = thrust::count_if(thrust::device, edge_num_of_community.begin(), edge_num_of_community.end(), filter_for_block_glb);
    int deg_num_grid = thrust::count_if(thrust::device, edge_num_of_community.begin(), edge_num_of_community.end(), filter_for_grid_glb);

    thrust::device_vector<edge_t> new_degrees(community_num, 0);
    thrust::sequence(new_degrees.begin(), new_degrees.end(), 0);

    thrust::device_vector<int> sorted_community_id(community_num); //{Comms with fewer edges than warp},{Comms with edges in shr},{Comms with edges in glb}
    thrust::copy_if(thrust::device, new_degrees.begin(), new_degrees.end(), edge_num_of_community.begin(),
                    sorted_community_id.begin(), filter_for_warp);
    thrust::copy_if(thrust::device, new_degrees.begin(), new_degrees.end(), edge_num_of_community.begin(),
                    sorted_community_id.begin() + deg_num_warp, filter_for_block_shr);
    thrust::copy_if(thrust::device, new_degrees.begin(), new_degrees.end(), edge_num_of_community.begin(),
                    sorted_community_id.begin() + deg_num_warp + deg_num_shr, filter_for_block_glb);
    thrust::copy_if(thrust::device, new_degrees.begin(), new_degrees.end(), edge_num_of_community.begin(),
                    sorted_community_id.begin() + deg_num_warp + deg_num_shr + deg_num_glb, filter_for_grid_glb);
    
    thrust::device_vector<edge_t> sorted_edge_num(community_num, 0);
    thrust::gather(sorted_community_id.begin(), sorted_community_id.end(), edge_num_of_community.begin(), sorted_edge_num.begin());

    // cout << deg_num_warp << " " << deg_num_shr << " " << deg_num_glb << " "<<deg_num_grid << endl;
    thrust::sort_by_key(sorted_edge_num.begin() + deg_num_warp + deg_num_shr, sorted_edge_num.end(),
                        sorted_community_id.begin() + deg_num_warp + deg_num_shr, thrust::greater<edge_t>());//deg_num_glb and deg_num_grid by descend order
    //sorted_community_id: deg_num_warp deg_num_shr deg_num_grid deg_num_glb
    thrust::device_vector<edge_t> neighbor_com_ptr_max(community_num + 1, 0);
    thrust::replace_copy_if(edge_num_of_community.begin(), edge_num_of_community.end(), neighbor_com_ptr_max.begin() + 1, degree_filter_greater_than(community_num), community_num);
    thrust::inclusive_scan(thrust::device, neighbor_com_ptr_max.begin(),
                           neighbor_com_ptr_max.end(), neighbor_com_ptr_max.begin()); // record new_neighbors location
    // thrust::inclusive_scan(thrust::device, edge_num_of_community.begin(),
    //                        edge_num_of_community.end(), neighbor_com_ptr_max.begin()+1); //{com0},{com0+com1},...
    edge_t max_new_edge_num = neighbor_com_ptr_max.back();
    int prime_num = primes.size();
    int warp_size = WARP_SIZE;

    thrust::device_vector<int> global_table_offset(deg_num_glb + deg_num_grid + 1, 0);

    if (deg_num_glb + deg_num_grid > 0)
    {
        block_num = ((deg_num_glb + deg_num_grid) * WARP_SIZE + 1024 - 1) / 1024;
        compute_global_table_size<<<block_num, 1024>>>(thrust::raw_pointer_cast(primes.data()), prime_num, thrust::raw_pointer_cast(sorted_edge_num.data()) + deg_num_warp + deg_num_shr,
                                                       deg_num_glb + deg_num_grid, thrust::raw_pointer_cast(global_table_offset.data()) + 1, warp_size, community_num);
    }
    sorted_edge_num.clear();
    // thrust::inclusive_scan(sorted_edge_num.begin() + deg_num_warp + degGreaterThan1024, sorted_edge_num.end(),
    //                        global_table_offset.begin() + 1, thrust::plus<int>()); // global_table_offset:{0,deg1,deg1+deg2}
    thrust::inclusive_scan(global_table_offset.begin(), global_table_offset.end(),
                           global_table_offset.begin(), thrust::plus<int>());
    //  cout << "max_new_edge_num:" << max_new_edge_num << endl;

    int global_table_size = global_table_offset.back();
    thrust::device_vector<int> global_table(2 * global_table_size, 0); // ?????????
    // cout<<"build_glb:"<<global_table.size()<<endl;

    thrust::device_vector<vertex_t> new_neighbors(max_new_edge_num, -1);
    thrust::device_vector<weight_t> new_weights(max_new_edge_num, 0);

    if (deg_num_warp)
    {
        block_num = (deg_num_warp * warp_size + THREAD_NUM_PER_BLOCK - 1) / THREAD_NUM_PER_BLOCK;
        int table_size = 257;
        int shared_mem_size = (THREAD_NUM_PER_BLOCK / warp_size) * table_size;
        // ccommunity_result << shared_mem_size << endl;
        build_by_warp<<<block_num, THREAD_NUM_PER_BLOCK, shared_mem_size * 2 * sizeof(int)>>>(thrust::raw_pointer_cast(sorted_community_id.data()),
                                                                                              thrust::raw_pointer_cast(d_weights.data()), 
                                                                                              thrust::raw_pointer_cast(d_neighbors.data()),
                                                                                            // pinned_weights, pinned_neighbors,
                                                                                              thrust::raw_pointer_cast(d_degrees.data()),
                                                                                              thrust::raw_pointer_cast(new_weights.data()), thrust::raw_pointer_cast(new_neighbors.data()), thrust::raw_pointer_cast(new_degrees.data()),
                                                                                              thrust::raw_pointer_cast(neighbor_com_ptr_max.data()), thrust::raw_pointer_cast(edge_num_of_community.data()), thrust::raw_pointer_cast(gathered_vertex_ptr.data()),
                                                                                              thrust::raw_pointer_cast(gathered_vertex_by_community.data()), thrust::raw_pointer_cast(community_result.data()),
                                                                                              deg_num_warp, table_size, warp_size, community_num, 1);
        cudaDeviceSynchronize();
    }
   
    if (deg_num_shr)
    {
        block_num = (deg_num_shr * THREAD_NUM_PER_BLOCK + THREAD_NUM_PER_BLOCK - 1) / THREAD_NUM_PER_BLOCK;
        build_by_block<<<block_num, THREAD_NUM_PER_BLOCK>>>(thrust::raw_pointer_cast(sorted_community_id.data()) + deg_num_warp,
                                                            thrust::raw_pointer_cast(d_weights.data()), thrust::raw_pointer_cast(d_neighbors.data()), 
                                                            // pinned_weights, pinned_neighbors,
                                                            thrust::raw_pointer_cast(d_degrees.data()),
                                                            thrust::raw_pointer_cast(new_weights.data()), thrust::raw_pointer_cast(new_neighbors.data()), thrust::raw_pointer_cast(new_degrees.data()),
                                                            thrust::raw_pointer_cast(neighbor_com_ptr_max.data()), thrust::raw_pointer_cast(edge_num_of_community.data()), thrust::raw_pointer_cast(gathered_vertex_ptr.data()),
                                                            thrust::raw_pointer_cast(gathered_vertex_by_community.data()), thrust::raw_pointer_cast(community_result.data()),
                                                            thrust::raw_pointer_cast(global_table_offset.data()), 
                                                            thrust::raw_pointer_cast(global_table.data()), global_table_size,
                                                            thrust::raw_pointer_cast(primes.data()), prime_num,
                                                            deg_num_shr, warp_size, global_limit, community_num, 1);
        cudaDeviceSynchronize();
    }

    if (deg_num_glb)
    {
        block_num = ((deg_num_glb) * MAX_THREAD_PER_BLOCK + MAX_THREAD_PER_BLOCK - 1) / MAX_THREAD_PER_BLOCK;
        // ccommunity_result<<"block_num:"<<block_num<<endl;
        build_by_block<<<block_num, MAX_THREAD_PER_BLOCK>>>(thrust::raw_pointer_cast(sorted_community_id.data()) + deg_num_warp + deg_num_shr + deg_num_grid,
                                                            thrust::raw_pointer_cast(d_weights.data()), thrust::raw_pointer_cast(d_neighbors.data()), 
                                                            // pinned_weights, pinned_neighbors,
                                                            thrust::raw_pointer_cast(d_degrees.data()),
                                                            thrust::raw_pointer_cast(new_weights.data()), thrust::raw_pointer_cast(new_neighbors.data()), thrust::raw_pointer_cast(new_degrees.data()),
                                                            thrust::raw_pointer_cast(neighbor_com_ptr_max.data()), thrust::raw_pointer_cast(edge_num_of_community.data()), thrust::raw_pointer_cast(gathered_vertex_ptr.data()),
                                                            thrust::raw_pointer_cast(gathered_vertex_by_community.data()), thrust::raw_pointer_cast(community_result.data()),
                                                            thrust::raw_pointer_cast(global_table_offset.data()) + deg_num_grid, 
                                                            thrust::raw_pointer_cast(global_table.data()), global_table_size,
                                                            thrust::raw_pointer_cast(primes.data()), prime_num,
                                                            deg_num_glb, warp_size, global_limit, community_num, 1);
        cudaDeviceSynchronize();
    }
    if(deg_num_grid)
    {
            int *sorted_community_id_=thrust::raw_pointer_cast(sorted_community_id.data()) + deg_num_warp + deg_num_shr;
            weight_t *weights_=thrust::raw_pointer_cast(d_weights.data());
            vertex_t *neighbors_=thrust::raw_pointer_cast(d_neighbors.data());
            edge_t *degrees_=thrust::raw_pointer_cast(d_degrees.data());

            weight_t *new_weights_=thrust::raw_pointer_cast(new_weights.data());
            vertex_t *new_neighbors_=thrust::raw_pointer_cast(new_neighbors.data());
            edge_t *new_degrees_=thrust::raw_pointer_cast(new_degrees.data());
            edge_t *neighbor_com_ptr_max_=thrust::raw_pointer_cast(neighbor_com_ptr_max.data());
            edge_t *edge_num_of_community_=thrust::raw_pointer_cast(edge_num_of_community.data());
            int *gathered_vertex_ptr_=thrust::raw_pointer_cast(gathered_vertex_ptr.data());
            int *gathered_vertex_by_community_=thrust::raw_pointer_cast(gathered_vertex_by_community.data());
            int *community_=thrust::raw_pointer_cast(community_result.data());
            int *global_table_ptr_=thrust::raw_pointer_cast(global_table_offset.data()); 
            int *glb_table_=thrust::raw_pointer_cast(global_table.data());
            int *primes_=thrust::raw_pointer_cast(primes.data());
            int is_store=1;

            int min_grid_size = 0;
            int block_size = 0;

            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, build_by_grid, 0, 0);

            thrust::device_vector<int> block_sum(min_grid_size, 0);
            int *block_sum_=thrust::raw_pointer_cast(block_sum.data());
            void *kernelArgs[] = {(void *)&(sorted_community_id_),
                (void *)&weights_, (void *)&neighbors_, (void *)&degrees_,
                (void *)&new_weights_, (void *)&new_neighbors_, (void *)&new_degrees_,
                (void *)&neighbor_com_ptr_max_, (void *)&edge_num_of_community_, (void *)&gathered_vertex_ptr_,
                (void *)&gathered_vertex_by_community_, (void *)&community_,
                (void *)&global_table_ptr_, (void *)&glb_table_, (void *)&global_table_size, (void *)&primes_, (void *)&prime_num,
                (void *)&deg_num_grid, (void *)&warp_size, (void *)&global_limit, (void *)&community_num, 
                (void *)&block_sum_, (void *)&is_store};
            cudaError_t err =cudaLaunchCooperativeKernel((void*)build_by_grid,min_grid_size, block_size, kernelArgs);
            cudaDeviceSynchronize();
    }

    thrust::inclusive_scan(thrust::device, new_degrees.begin(), new_degrees.end(), new_degrees.begin(), thrust::plus<edge_t>());

    edge_t new_edge_num = new_degrees.back();
    degrees.resize(new_degrees.size());
    thrust::copy(thrust::device, new_degrees.begin(), new_degrees.end(), degrees.begin());
    
    // std::cout<<"new_edge_num:"<<new_edge_num<<endl;

    d_weights.resize(new_edge_num);
    d_neighbors.resize(new_edge_num);

    thrust::copy_if(thrust::device, new_weights.begin(),
                    new_weights.end(), d_weights.begin(),
                    degree_filter_greater_than_(0));
    thrust::copy_if(thrust::device, new_neighbors.begin(),
                    new_neighbors.end(), d_neighbors.begin(),
                    degree_filter_greater_than(0));

    return community_num;
}
