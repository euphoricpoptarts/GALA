#pragma once
#include "gpu_config.h"
#include "graph/graph_config.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <time.h>
#include <unistd.h>
#include <vector>
#define VITE_ALPHA 0.25
using namespace std;

struct degree_filter_in_range
{
    int lower;
    int upper;
    __host__ __device__ degree_filter_in_range(int l, int u)
        : lower(l), upper(u) {}
    __host__ __device__ bool operator()(edge_t x)
    {
        return (lower <= x && x <= upper);
    }
};

struct degree_filter_greater_than
{
    int lower;
    __host__ __device__ degree_filter_greater_than(int l) : lower(l) {}
    __host__ __device__ bool operator()(edge_t x) { return (lower <= x); }
};

struct modularity_op
{
    double constant;
    __host__ __device__ modularity_op(double c) : constant(c) {}

    __host__ __device__ double operator()(const int &x, const int &y)
    {
        return (double)((double)x * constant -
                        (double)((long)y * (long)y) * constant * constant);
    }
};

struct vertex_filter
{
    thrust::device_ptr<int> is_retained;
    int flag;
    __host__ __device__ vertex_filter(thrust::device_ptr<int> q, bool f)
    {
        is_retained = q;
        flag = f;
    }

    __host__ __device__ bool operator()(int x)
    {
        return is_retained[x] == flag;
    }
};

struct Tot_op
{
    edge_t constant;
    __host__ __device__ Tot_op(edge_t c) : constant(c) {}

    __host__ __device__ double operator()(const int &x)
    {
        return x == 0 ? constant : x;
    }
};

__device__ int find_prime_by_warp(int *primes, int prime_num, int threshold, int warp_size);

__global__ void init_communities(weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                                 int *K, int *Tot, int *Self, int *prev_community,
                                 int vertex_num, int warp_size);

__global__ void compute_global_table_size_louvain(int *primes, int prime_num, int *degree_of_vertex, int deg_num_glb, int *global_table_ptr, int warp_size);

__device__ int insert_neighbor_com_info(int *shared_neighbor_com_ids, int *shared_neighbor_com_weights, int *shared_Tot,
                                        int *global_neighbor_com_ids, int *global_neighbor_com_weights,
                                        int shared_size, int global_size, int neighbor_com, int neighbor_weight,
                                        const int *__restrict__ Tot, int Ki, int cur_com, int tot_cur_com, double *maxdq, int *best_com,
                                        double constant, int vertex, int is_global);

__device__ int find_neighbor_com_info(int *shared_neighbor_com_ids, int *shared_neighbor_com_weights, int *shared_Tot,
                                      int *global_neighbor_com_ids, int *global_neighbor_com_weights,
                                      int shared_size, int global_size, int is_global, int vertex_com);

__device__ void build_and_select_in_warp(int vertex, const weight_t *__restrict__ weights, const vertex_t *__restrict__ neighbors, const edge_t *__restrict__ degrees,
                                         int *In, int *next_In, const int *__restrict__ Tot, int *Self, int Ki, double constant,
                                         int *prev_community, int *cur_community, int *com_size,
                                         int *com_size_update, int *Tot_update,
                                         int cur_com, edge_t start_neighbor, edge_t end_neighbor, int neighbor_num,
                                         int laneId, int warp_size,
                                         int *active_set, int *is_moved, int *target_com_weights,
                                         int iteration);

__global__ void decide_and_move_shuffle(int *sorted_vertex_id, const weight_t *__restrict__ weights, const vertex_t *__restrict__ neighbors, const edge_t *__restrict__ degrees,
                                        int *prev_community, int *cur_community,
                                        int *K, const int *__restrict__ Tot, int *In, int *next_In, int *Self,
                                        int *com_size, int *Tot_update, int *com_size_update,
                                        int vertex_num_to_proc, int warp_size, double constant,
                                        int *active_set, int *is_moved, int *target_com_weights,
                                        int iteration);

__device__ void build_and_select(int vertex, const weight_t *__restrict__ weights, const vertex_t *__restrict__ neighbors, const edge_t *__restrict__ degrees,
                                 int *In, int *next_In, const int *__restrict__ Tot, int *Self, int Ki, double constant,
                                 int *prev_community, int *cur_community, int *com_size,
                                 int *com_size_update, int *Tot_update,
                                 int *neighbor_com_ids, int *neighbor_com_weights, int *shared_Tot,
                                 int cur_com, edge_t start_neighbor, edge_t end_neighbor, int neighbor_num,
                                 int laneId, int warp_size, int table_size, int is_global,
                                 int *active_set, int *is_moved, int *target_com_weights);

__global__ void decide_and_move_hash_shared(int *sorted_vertex_id, const weight_t *__restrict__ weights, const vertex_t *__restrict__ neighbors, const edge_t *__restrict__ degrees,
                                            int *prev_community, int *cur_community,
                                            int *K, const int *__restrict__ Tot, int *In, int *next_In, int *Self,
                                            int *com_size, int *Tot_update, int *com_size_update,
                                            int vertex_num_to_proc, int table_size, int warp_size, double constant,
                                            int *active_set, int *is_moved, int *target_com_weights, int iteration);

__device__ void build_and_select_blk(int vertex, const weight_t *__restrict__ weights, const vertex_t *__restrict__ neighbors, const edge_t *__restrict__ degrees,
                                     int *In, int *next_In, const int *__restrict__ Tot, int *Self, int Ki, double constant,
                                     int *prev_community, int *cur_community, int *com_size,
                                     int *com_size_update, int *Tot_update,
                                     int *shared_neighbor_com_ids, int *shared_neighbor_com_weights, int *shared_Tot,
                                     int *global_neighbor_com_ids, int *global_neighbor_com_weights,
                                     int cur_com, edge_t start_neighbor, int neighbor_num,
                                     int threadIdInBlk, int threadNumInBlk, int warp_size, int shared_size, int global_size, int is_global,
                                     int *active_set, int *is_moved, int *target_com_weights, int iteration);

__global__ void decide_and_move_hash_hierarchical(int *sorted_vertex_id, const weight_t *__restrict__ weights, const vertex_t *__restrict__ neighbors, const edge_t *__restrict__ degrees,
                                                  int *prev_community, int *cur_community,
                                                  int *K, const int *__restrict__ Tot, int *In, int *next_In, int *Self,
                                                  int *com_size, int *Tot_update, int *com_size_update,
                                                  int *global_table_ptr, int *glbTable, int *primes, int prime_num,
                                                  int vertex_num_to_proc, int warp_size, int global_limit, double constant,
                                                  int *active_set, int *is_moved, int *target_com_weights, int iteration);

__global__ void compute_In(int *sorted_vertex_id, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                           int *cur_community,
                           int *K, int *Tot, int *In, int *Self,
                           int vertex_num_to_proc, int warp_size, double constant, int min_Tot,
                           int *active_set, int pruning_method);

__global__ void compute_In_blk(int *sorted_vertex_id, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                               int *cur_community,
                               int *K, int *Tot, int *In, int *Self,
                               int vertex_num_to_proc, int warp_size, double constant, int min_Tot,
                               int *active_set, int pruning_method);

//
__global__ void renumber(int *renumber, vertex_t *out, int vertex_num);

__global__ void get_size_of_each_community(vertex_t *community, int *size_of_community, int vertex_num);

__global__ void gather_vertex_by_community(int *gathered_vertex_by_community, int *community, int *pos_ptr, int vertex_num);

__global__ void compute_edge_num_of_each_community(edge_t *degrees, int *gathered_vertex_by_community, int *gathered_vertex_ptr,
                                                   edge_t *edge_num_of_community, int community_num, int warp_size);

__global__ void compute_global_table_size(int *primes, int prime_num, edge_t *edge_num_of, int deg_num_glb, int *global_table_ptr, int warp_size, int community_num);

__device__ int insert_neighbor_com_info_for_com(int *neighbor_com_ids, int *neighbor_com_weights,
                                                int table_size, int neighbor_com, int neighbor_weight);

__device__ int find_neighbor_com_info_for_com(int *neighbor_com_ids, int *neighbor_com_weights,
                                              int table_size, int vertex_com);

__device__ void compute_neighbors_weights(int com, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                                          weight_t *new_weights, vertex_t *new_neighbors, edge_t *new_degrees,
                                          int *community, int *gathered_vertex_ptr, int *gathered_vertex_by_community,
                                          int *neighbor_com_ids, int *neighbor_com_weights,
                                          int laneId, int table_size, int warp_size);

__device__ void compute_neighbors_weights_blk(int com, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                                              weight_t *new_weights, vertex_t *new_neighbors, edge_t *new_degrees,
                                              int *community, int *gathered_vertex_ptr, int *gathered_vertex_by_community,
                                              int *neighbor_com_ids, int *neighbor_com_weights,
                                              int threadIdInBlk, int table_size, int warp_size);

__device__ void compute_neighbors_weights_list(int com, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                                               weight_t *new_weights, vertex_t *new_neighbors, edge_t *new_degrees,
                                               int *community, int *gathered_vertex_ptr, int *gathered_vertex_by_community,
                                               int *neighbor_com_ids, int *neighbor_com_weights,
                                               int laneId, int community_num, int warp_size);

__device__ void compute_neighbors_weights_blk_list(int com, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                                                   weight_t *new_weights, vertex_t *new_neighbors, edge_t *new_degrees,
                                                   int *community, int *gathered_vertex_ptr, int *gathered_vertex_by_community,
                                                   int *neighbor_com_ids, int *neighbor_com_weights,
                                                   int threadIdInBlk, int community_num, int warp_size);

__global__ void build_by_warp(int *sorted_community_id, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                              weight_t *new_weights, vertex_t *new_neighbors, edge_t *new_degrees,
                              edge_t *neighbor_com_ptr_max, edge_t *edge_num_of_community, int *gathered_vertex_ptr, int *gathered_vertex_by_community, int *community,
                              int com_num_to_proc, int table_size, int warp_size, int community_num);

__global__ void build_by_block(int *sorted_community_id, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                               weight_t *new_weights, vertex_t *new_neighbors, edge_t *new_degrees,
                               edge_t *neighbor_com_ptr_max, edge_t *edge_num_of_community, int *gathered_vertex_ptr, int *gathered_vertex_by_community, int *community,
                               int *global_table_ptr, int *glbTable, int *primes, int prime_num,
                               int com_num_to_proc, int warp_size, int global_limit, int community_num);

__global__ void save_next_In(int *In, int *next_In, int* prev_community, int *cur_community,
                             int *active_set, int *is_moved, int *target_com_weights,
                             int *Tot, int *Self, int *K, double *prob, edge_t min_Tot,
                            double constant, int vertex_num, int iteration, int pruning_method);