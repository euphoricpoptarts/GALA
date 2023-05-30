#include "louvain.cuh"
#include "kernel_functions.cuh"
using namespace std;

int build_compressed_graph(thrust::device_vector<weight_t> &d_weights, thrust::device_vector<vertex_t> &d_neighbors, thrust::device_vector<edge_t> &degrees,
                           thrust::device_vector<vertex_t> &coummunity_result, thrust::device_vector<int> &primes, int vertex_num)
{
    // renumber

    // build
    thrust::device_vector<int> size_of_each_community(vertex_num, 0); // vertex number of community
    int block_num = (vertex_num + 1024 - 1) / 1024;
    get_size_of_each_community<<<block_num, 1024>>>(thrust::raw_pointer_cast(coummunity_result.data()), thrust::raw_pointer_cast(size_of_each_community.data()), vertex_num);
    cudaDeviceSynchronize();
    degree_filter_in_range filter0(1, vertex_num);
    int community_num = thrust::count_if(thrust::device, size_of_each_community.begin(), size_of_each_community.end(), filter0);

    thrust::device_vector<int> community(vertex_num, 0);
    thrust::transform(thrust::device, size_of_each_community.begin(), size_of_each_community.end(), community.begin(), degree_filter_greater_than(1));
    thrust::exclusive_scan(thrust::device, community.begin(), community.end(), community.begin()); // renumber from 0

    block_num = (vertex_num + 1024 - 1) / 1024;
    renumber<<<block_num, 1024>>>(thrust::raw_pointer_cast(community.data()), thrust::raw_pointer_cast(coummunity_result.data()), vertex_num);
    community.clear();

    thrust::device_vector<int> pos_offset(community_num + 1, 0);

    thrust::copy_if(size_of_each_community.begin(), size_of_each_community.end(), pos_offset.begin() + 1, degree_filter_greater_than(1));
    thrust::inclusive_scan(thrust::device, pos_offset.begin(), pos_offset.end(), pos_offset.begin()); // pos_offset[i+1]-pos_offset[i]=size of comminity i
    size_of_each_community.clear();

    thrust::device_vector<int> gathered_vertex_ptr(pos_offset);          // save a copy
    thrust::device_vector<int> gathered_vertex_by_community(vertex_num); //{veretexs in com0},{veretexs in com1},...
    gather_vertex_by_community<<<block_num, 1024>>>(thrust::raw_pointer_cast(gathered_vertex_by_community.data()), thrust::raw_pointer_cast(coummunity_result.data()),
                                                    thrust::raw_pointer_cast(pos_offset.data()), vertex_num); //{vertexs{com-0},vertexs{com-1},...}
    cudaDeviceSynchronize();
    pos_offset.clear();

    thrust::device_vector<edge_t> edge_num_of_community(community_num); // count edge number in community
    block_num = (community_num * WARP_SIZE + MAX_THREAD_PER_BLOCK - 1) / MAX_THREAD_PER_BLOCK;
    // ccoummunity_result<<block_num<<" "<<gathered_vertex_ptr.size()<<endl;
    thrust::device_vector<edge_t> d_degrees(degrees.size() + 1, 0);
    thrust::copy(degrees.begin(), degrees.end(), d_degrees.begin() + 1);
    compute_edge_num_of_each_community<<<block_num, MAX_THREAD_PER_BLOCK>>>(thrust::raw_pointer_cast(d_degrees.data()), thrust::raw_pointer_cast(gathered_vertex_by_community.data()),
                                                                            thrust::raw_pointer_cast(gathered_vertex_ptr.data()), thrust::raw_pointer_cast(edge_num_of_community.data()), community_num, WARP_SIZE);
    cudaDeviceSynchronize();

    // thrust::device_ptr<int> maxEdgeNum = thrust::max_element(edge_num_of_community.begin(), edge_num_of_community.end());
    int global_limit = SHARE_MEM_SIZE * 2 / 3 - 1;

    degree_filter_in_range filter_for_warp(0, 127);
    degree_filter_in_range filter_for_block_shr(128, global_limit);
    degree_filter_greater_than filter_for_block_glb(global_limit + 1);

    int deg_num_warp = thrust::count_if(thrust::device, edge_num_of_community.begin(), edge_num_of_community.end(), filter_for_warp);
    int deg_num_shr = thrust::count_if(thrust::device, edge_num_of_community.begin(), edge_num_of_community.end(), filter_for_block_shr);
    int deg_num_glb = thrust::count_if(thrust::device, edge_num_of_community.begin(), edge_num_of_community.end(), filter_for_block_glb);

    thrust::device_vector<edge_t> new_degrees(community_num, 0);
    thrust::sequence(new_degrees.begin(), new_degrees.end(), 0);

    thrust::device_vector<int> sorted_community_id(community_num); //{Comms with fewer edges than warp},{Comms with edges in shr},{Comms with edges in glb}
    thrust::copy_if(thrust::device, new_degrees.begin(), new_degrees.end(), edge_num_of_community.begin(),
                    sorted_community_id.begin(), filter_for_warp);
    thrust::copy_if(thrust::device, new_degrees.begin(), new_degrees.end(), edge_num_of_community.begin(),
                    sorted_community_id.begin() + deg_num_warp, filter_for_block_shr);
    thrust::copy_if(thrust::device, new_degrees.begin(), new_degrees.end(), edge_num_of_community.begin(),
                    sorted_community_id.begin() + deg_num_warp + deg_num_shr, filter_for_block_glb);

    thrust::device_vector<edge_t> sorted_edge_num(community_num, 0);
    thrust::gather(sorted_community_id.begin(), sorted_community_id.end(), edge_num_of_community.begin(), sorted_edge_num.begin());

    // ccoummunity_result << deg_num_warp << " " << degGreaterThan1024 << " " << deg_num_glb << endl;
    thrust::sort_by_key(sorted_edge_num.begin() + deg_num_warp + deg_num_shr, sorted_edge_num.end(),
                        sorted_community_id.begin() + deg_num_warp + deg_num_shr, thrust::greater<edge_t>());

    thrust::device_vector<edge_t> neighbor_com_ptr_max(community_num + 1, 0);
    thrust::replace_copy_if(edge_num_of_community.begin(), edge_num_of_community.end(), neighbor_com_ptr_max.begin() + 1, degree_filter_greater_than(community_num), community_num);
    thrust::inclusive_scan(thrust::device, neighbor_com_ptr_max.begin(),
                           neighbor_com_ptr_max.end(), neighbor_com_ptr_max.begin()); // record new_neighbors location
    // thrust::inclusive_scan(thrust::device, edge_num_of_community.begin(),
    //                        edge_num_of_community.end(), neighbor_com_ptr_max.begin()+1); //{com0},{com0+com1},...
    edge_t max_new_edge_num = neighbor_com_ptr_max.back();

    int prime_num = primes.size();
    int warp_size = WARP_SIZE;

    thrust::device_vector<int> global_table_offset(deg_num_glb + 1, 0);

    if (deg_num_glb > 0)
    {
        block_num = (deg_num_glb * WARP_SIZE + 1024 - 1) / 1024;
        compute_global_table_size<<<block_num, 1024>>>(thrust::raw_pointer_cast(primes.data()), prime_num, thrust::raw_pointer_cast(sorted_edge_num.data()) + deg_num_warp + deg_num_shr,
                                                       deg_num_glb, thrust::raw_pointer_cast(global_table_offset.data()) + 1, warp_size, community_num);
    }
    sorted_edge_num.clear();
    // thrust::inclusive_scan(sorted_edge_num.begin() + deg_num_warp + degGreaterThan1024, sorted_edge_num.end(),
    //                        global_table_offset.begin() + 1, thrust::plus<int>()); // global_table_offset:{0,deg1,deg1+deg2}
    thrust::inclusive_scan(global_table_offset.begin(), global_table_offset.end(),
                           global_table_offset.begin(), thrust::plus<int>());
    //  ccoummunity_result << "max_new_edge_num:" << max_new_edge_num << endl;
    //  ccoummunity_result<<"build_glb:"<<global_table_offset.back()<<endl;
    thrust::device_vector<int> global_table(2 * global_table_offset.back(), 0); // ?????????
    // ccoummunity_result<<"build_glb:"<<global_table_offset.back()<<endl;

    thrust::device_vector<vertex_t> new_neighbors(max_new_edge_num, -1);
    thrust::device_vector<weight_t> new_weights(max_new_edge_num, -1);

    if (deg_num_warp)
    {
        block_num = (deg_num_warp * warp_size + THREAD_NUM_PER_BLOCK - 1) / THREAD_NUM_PER_BLOCK;
        int table_size = 257;
        int shared_mem_size = (THREAD_NUM_PER_BLOCK / warp_size) * table_size;
        // ccoummunity_result << shared_mem_size << endl;
        build_by_warp<<<block_num, THREAD_NUM_PER_BLOCK, shared_mem_size * 2 * sizeof(int)>>>(thrust::raw_pointer_cast(sorted_community_id.data()),
                                                                                              thrust::raw_pointer_cast(d_weights.data()), thrust::raw_pointer_cast(d_neighbors.data()), thrust::raw_pointer_cast(d_degrees.data()),
                                                                                              thrust::raw_pointer_cast(new_weights.data()), thrust::raw_pointer_cast(new_neighbors.data()), thrust::raw_pointer_cast(new_degrees.data()),
                                                                                              thrust::raw_pointer_cast(neighbor_com_ptr_max.data()), thrust::raw_pointer_cast(edge_num_of_community.data()), thrust::raw_pointer_cast(gathered_vertex_ptr.data()),
                                                                                              thrust::raw_pointer_cast(gathered_vertex_by_community.data()), thrust::raw_pointer_cast(coummunity_result.data()),
                                                                                              deg_num_warp, table_size, warp_size, community_num);
        cudaDeviceSynchronize();
    }

    if (deg_num_shr)
    {
        block_num = (deg_num_shr * THREAD_NUM_PER_BLOCK + THREAD_NUM_PER_BLOCK - 1) / THREAD_NUM_PER_BLOCK;
        build_by_block<<<block_num, THREAD_NUM_PER_BLOCK>>>(thrust::raw_pointer_cast(sorted_community_id.data()) + deg_num_warp,
                                                            thrust::raw_pointer_cast(d_weights.data()), thrust::raw_pointer_cast(d_neighbors.data()), thrust::raw_pointer_cast(d_degrees.data()),
                                                            thrust::raw_pointer_cast(new_weights.data()), thrust::raw_pointer_cast(new_neighbors.data()), thrust::raw_pointer_cast(new_degrees.data()),
                                                            thrust::raw_pointer_cast(neighbor_com_ptr_max.data()), thrust::raw_pointer_cast(edge_num_of_community.data()), thrust::raw_pointer_cast(gathered_vertex_ptr.data()),
                                                            thrust::raw_pointer_cast(gathered_vertex_by_community.data()), thrust::raw_pointer_cast(coummunity_result.data()),
                                                            thrust::raw_pointer_cast(global_table_offset.data()), thrust::raw_pointer_cast(global_table.data()), thrust::raw_pointer_cast(primes.data()), prime_num,
                                                            deg_num_shr, warp_size, global_limit, community_num);
        cudaDeviceSynchronize();
    }

    if (deg_num_glb)
    {
        block_num = (deg_num_glb * MAX_THREAD_PER_BLOCK + MAX_THREAD_PER_BLOCK - 1) / MAX_THREAD_PER_BLOCK;
        // ccoummunity_result<<"block_num:"<<block_num<<endl;
        build_by_block<<<block_num, MAX_THREAD_PER_BLOCK>>>(thrust::raw_pointer_cast(sorted_community_id.data()) + deg_num_warp + deg_num_shr,
                                                            thrust::raw_pointer_cast(d_weights.data()), thrust::raw_pointer_cast(d_neighbors.data()), thrust::raw_pointer_cast(d_degrees.data()),
                                                            thrust::raw_pointer_cast(new_weights.data()), thrust::raw_pointer_cast(new_neighbors.data()), thrust::raw_pointer_cast(new_degrees.data()),
                                                            thrust::raw_pointer_cast(neighbor_com_ptr_max.data()), thrust::raw_pointer_cast(edge_num_of_community.data()), thrust::raw_pointer_cast(gathered_vertex_ptr.data()),
                                                            thrust::raw_pointer_cast(gathered_vertex_by_community.data()), thrust::raw_pointer_cast(coummunity_result.data()),
                                                            thrust::raw_pointer_cast(global_table_offset.data()), thrust::raw_pointer_cast(global_table.data()), thrust::raw_pointer_cast(primes.data()), prime_num,
                                                            deg_num_glb, warp_size, global_limit, community_num);
        cudaDeviceSynchronize();
    }

    // global_table.clear();
    // global_table_offset.clear();
    // edge_num_of_community.clear();

    thrust::inclusive_scan(thrust::device, new_degrees.begin(), new_degrees.end(), new_degrees.begin(), thrust::plus<edge_t>());

    edge_t new_edge_num = new_degrees.back();
    degrees.resize(new_degrees.size());
    thrust::copy(thrust::device, new_degrees.begin(), new_degrees.end(), degrees.begin());
    d_weights.resize(new_edge_num);
    d_neighbors.resize(new_edge_num);
    thrust::copy_if(thrust::device, new_weights.begin(),
                    new_weights.end(), d_weights.begin(),
                    degree_filter_greater_than(0));
    thrust::copy_if(thrust::device, new_neighbors.begin(),
                    new_neighbors.end(), d_neighbors.begin(),
                    degree_filter_greater_than(0));
    // new_degrees.clear();
    // new_weights.clear();
    // new_neighbors.clear();

    return community_num;
}
