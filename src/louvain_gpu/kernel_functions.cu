#include "gpu_config.h"
#include "graph/graph_config.h"
#include "kernel_functions.cuh"
using namespace std;

__device__ int find_prime_by_warp(int *primes, int prime_num, int threshold, int warp_size)
{
    unsigned int lane_id = threadIdx.x & (warp_size - 1);
    int my_prime = primes[prime_num - 1]; // Largest one in the file as MAXPRIME

    if (threshold > my_prime)
        return -1;

    for (unsigned int i = lane_id; i < prime_num; i = i + warp_size)
    {

        int current_prime = __ldg(&primes[i]);

        if (current_prime > threshold)
        {
            my_prime = current_prime;
            break;
        }
    }
    for (unsigned int i = warp_size / 2; i >= 1; i = i / 2)
    {
        int received = __shfl_xor_sync(0xffffffff, my_prime, i, warp_size);
        if (received < my_prime)
            my_prime = received;
    }
    return my_prime;
}

__global__ void init_communities(weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                                 int *K, int *Tot, int *Self, int *prev_community,
                                 int vertex_num, int warp_size)
{ // 1 warp per vertex

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int thread_num = blockDim.x * gridDim.x;
    int warp_id = thread_id / warp_size;
    int lane_id = threadIdx.x % warp_size;

    while (warp_id < vertex_num)
    {
        vertex_t vertex_id = warp_id;
        edge_t start = degrees[warp_id];
        edge_t end = degrees[warp_id + 1];
        int neighbor_num = end - start;
        int degree = 0;
        for (int i = lane_id; i < neighbor_num; i = i + warp_size)
        {
            weight_t w = weights[start + i];
            degree = degree + w;
            if (vertex_id == neighbors[start + i])
            {
                Self[vertex_id] = w;
            }
        }
        for (int i = warp_size / 2; i >= 1; i = i / 2)
        {
            degree += __shfl_xor_sync(0xffffffff, degree, i, WARP_SIZE);
        }
        if (lane_id == 0)
        {
            prev_community[warp_id] = warp_id;
            K[warp_id] = degree;
            Tot[warp_id] = degree;
        }
        warp_id += thread_num / warp_size;
    }
}

__global__ void compute_global_table_size_louvain(int *primes, int prime_num, int *degree_of_vertex, int deg_num_glb, int *global_table_ptr, int warp_size)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;

    while (warp_id < deg_num_glb)
    {
        int edge_num = degree_of_vertex[warp_id];
        int nearest_prime = find_prime_by_warp(primes, prime_num, (edge_num * 3) / 2, warp_size); // neighbor_num must<37858284

        if (nearest_prime < 0)
        {
            nearest_prime = find_prime_by_warp(primes, prime_num, edge_num, warp_size);
        }
        if (nearest_prime == -1)
        {
            global_table_ptr[warp_id] = edge_num;
        }
        else
        {
            global_table_ptr[warp_id] = nearest_prime;
        }

        warp_id += (blockDim.x * gridDim.x) / WARP_SIZE;
    }
}

__device__ int insert_neighbor_com_info(int *shared_neighbor_com_ids, int *shared_neighbor_com_weights, int *shared_Tot,
                                        int *global_neighbor_com_ids, int *global_neighbor_com_weights,
                                        int shared_size, int global_size, int neighbor_com, int neighbor_weight,
                                        const int *__restrict__ Tot, int Ki, int cur_com, int tot_cur_com, double *maxdq, int *best_com,
                                        double constant, int vertex, int is_global)
{ // insert into hashmap
    int hash_value1 = neighbor_com % shared_size;
    int hash_value2 = 1 + (neighbor_com % (shared_size - 1));
    int pos;
    if (is_global == 0)
    { // shared

        int prev_weight, cur_weight;
        int i = 0;

        while (i < shared_size)
        {
            pos = (hash_value1 + i * hash_value2) % shared_size;
            int com_id = atomicCAS((int *)&shared_neighbor_com_ids[pos], 0, (1 + neighbor_com)); // 0 means no element in this position and swap
            if (com_id == 0 || com_id == (neighbor_com + 1))
            { // this position is vacant
                prev_weight = atomicAdd((int *)&shared_neighbor_com_weights[pos], neighbor_weight);
                int tot_neighbor_com = shared_Tot[pos];
                if (tot_neighbor_com == 0)
                {
                    tot_neighbor_com = Tot[neighbor_com];
                    shared_Tot[pos] = tot_neighbor_com;
                }
                cur_weight = prev_weight + neighbor_weight;
                double dq = 0;
                if (neighbor_com != cur_com)
                {
                    dq = (double)(2.0 * (double)cur_weight - 2.0 * (double)Ki * ((double)tot_neighbor_com - (double)tot_cur_com + (double)Ki) * constant);
                }
                if ((dq > *maxdq) || ((dq == *maxdq) && (dq != 0) && (neighbor_com < *best_com)))
                { // dq may equal 0
                    *maxdq = dq;
                    *best_com = neighbor_com;
                }

                return pos;
            }
            else
            {
                i += 1;
            }
        }
    }
    else
    { // global+share
        pos = hash_value1 % shared_size;
        int com_id = atomicCAS((int *)&shared_neighbor_com_ids[pos], 0, (1 + neighbor_com)); // 0 means no element in this position and swap
        int prev_weight, cur_weight;
        if (com_id == 0 || com_id == (neighbor_com + 1)) // share
        {                                                // this position is vacant
            prev_weight = atomicAdd((int *)&shared_neighbor_com_weights[pos], neighbor_weight);
            int tot_neighbor_com = shared_Tot[pos];
            if (tot_neighbor_com == 0)
            {
                tot_neighbor_com = Tot[neighbor_com];
                shared_Tot[pos] = tot_neighbor_com;
            }
            cur_weight = prev_weight + neighbor_weight;
            double dq = 0;
            if (neighbor_com != cur_com)
            {
                dq = (double)(2.0 * (double)cur_weight - 2.0 * (double)Ki * ((double)tot_neighbor_com - (double)tot_cur_com + (double)Ki) * constant);
            }
            if ((dq > *maxdq) || ((dq == *maxdq) && (dq != 0) && (neighbor_com < *best_com)))
            { // dq may equal 0
                *maxdq = dq;
                *best_com = neighbor_com;
            }

            return pos;
        }
        else
        { // global
            hash_value1 = neighbor_com % global_size;
            hash_value2 = 1 + (neighbor_com % (global_size - 1));
            int i = 0;
            while (i < global_size)
            {
                pos = (hash_value1 + i * hash_value2) % global_size;
                int com_id = atomicCAS((int *)&global_neighbor_com_ids[pos], 0, (1 + neighbor_com)); // 0 means no element in this position and swap
                if (com_id == 0 || com_id == (neighbor_com + 1))
                { // this position is vacant
                    prev_weight = atomicAdd((int *)&global_neighbor_com_weights[pos], neighbor_weight);

                    cur_weight = prev_weight + neighbor_weight;
                    double dq = 0;
                    if (neighbor_com != cur_com)
                    {
                        dq = (double)(2.0 * (double)cur_weight - 2.0 * (double)Ki * ((double)Tot[neighbor_com] - (double)tot_cur_com + (double)Ki) * constant);
                    }
                    if ((dq > *maxdq) || ((dq == *maxdq) && (dq != 0) && (neighbor_com < *best_com)))
                    { // dq may equal 0
                        *maxdq = dq;
                        *best_com = neighbor_com;
                    }
                    return pos;
                }
                else
                {
                    i += 1;
                }
            }
            return -1;
        }
    }
    return -1;
}

__device__ int find_neighbor_com_info(int *shared_neighbor_com_ids, int *shared_neighbor_com_weights, int *shared_Tot,
                                      int *global_neighbor_com_ids, int *global_neighbor_com_weights,
                                      int shared_size, int global_size, int is_global, int vertex_com)
{ // find in hash table
    int hash_value1 = vertex_com % shared_size;
    int hash_value2 = 1 + (vertex_com % (shared_size - 1));
    int pos;
    if (is_global == 0)
    { // shared
        int i = 0;
        do
        {
            pos = (hash_value1 + i * hash_value2) % shared_size;
            int com_id = shared_neighbor_com_ids[pos];

            if (com_id == (vertex_com + 1))
            {

                return shared_neighbor_com_weights[pos];
            }
            else
            {
                i += 1;
            }
        } while (i < shared_size);
    }
    else
    { // global+shared
        pos = hash_value1 % shared_size;
        int com_id = shared_neighbor_com_ids[pos];

        if (com_id == (vertex_com + 1))
        {

            return shared_neighbor_com_weights[pos];
        }
        int i = 0;
        hash_value1 = vertex_com % global_size;
        hash_value2 = 1 + (vertex_com % (global_size - 1));
        do
        {
            // while(i<table_size){
            pos = (hash_value1 + i * hash_value2) % global_size;
            // int com_id=neighbor_com_ids[pos];
            int com_id = global_neighbor_com_ids[pos];

            if (com_id == (vertex_com + 1))
            {

                return global_neighbor_com_weights[pos];
            }
            else
            {
                i += 1;
            }

        } while (i < global_size);
    }
    return -1;
}

__device__ void build_and_select_in_warp(int vertex, const weight_t *__restrict__ weights, const vertex_t *__restrict__ neighbors, const edge_t *__restrict__ degrees,
                                         int *In, int *next_In, const int *__restrict__ Tot, int *Self, int Ki, double constant,
                                         int *prev_community, int *cur_community, int *com_size,
                                         int *com_size_update, int *Tot_update,
                                         int cur_com, edge_t start_neighbor, edge_t end_neighbor, int neighbor_num,
                                         int lane_id, int warp_size,
                                         int *active_set, int *is_moved, int *target_com_weights,
                                         int iteration)
{ // for each neighbor,compute gain
    double maxdq = 0;
    int best_com = cur_com;

    // init hash table
    int com_v = cur_com, wt = 0;

    if (lane_id < neighbor_num)
    {

        int v = neighbors[start_neighbor + lane_id];

        com_v = prev_community[v];

        wt = weights[start_neighbor + lane_id];
    }

    int wt_final = wt;

    // exchange info between warp
    if (iteration > 0)
    {
        int com_v_other = com_v;
        int wt_other = wt;
        for (int i = 0; i < warp_size - 1; i += 1)
        {
            com_v_other = __shfl_sync(0xffffffff, com_v_other, (lane_id + 1) % warp_size, warp_size);
            wt_other = __shfl_sync(0xffffffff, wt_other, (lane_id + 1) % warp_size, warp_size);

            if (com_v_other == com_v)
            {
                wt_final += wt_other;
            }
        }
    }

    // __syncthreads();
    int cur_com_weight = wt_final;
    cur_com_weight = __shfl_sync(0xffffffff, wt_final, neighbor_num, warp_size);

    // __syncthreads();

    if (iteration > 0)
    {
        if (lane_id == neighbor_num)
        {
            In[vertex] = wt_final;
        }
    }
    else
    {
        if (com_v == cur_com)
        { // when iteration==0,only one thread's com_v==cur_com
            In[vertex] = wt_final;
        }
    }

    int best_com_weight = wt_final;
    // select best community
    if (com_v != cur_com)
    {
        maxdq = (double)(2.0 * (double)wt_final - 2.0 * (double)Ki * ((double)Tot[com_v] - (double)Tot[cur_com] + (double)Ki) * constant);
        best_com = com_v;
    }

    for (int i = warp_size / 2; i >= 1; i = i / 2)
    {
        double recv_dq = __shfl_xor_sync(0xffffffff, maxdq, i, warp_size); // only for warp
        int recv_com = __shfl_xor_sync(0xffffffff, best_com, i, warp_size);
        int recv_weight = __shfl_xor_sync(0xffffffff, best_com_weight, i, warp_size);
        if ((recv_dq > maxdq) || (recv_dq == maxdq && recv_com < best_com))
        {
            maxdq = recv_dq;
            best_com = recv_com;
            best_com_weight = recv_weight;
        }
    }

    maxdq = maxdq - 2 * cur_com_weight + 2 * Self[vertex];

    if (best_com >= 0 && best_com != cur_com && maxdq > 0)
    {
        if (com_size[best_com] == 1 && com_size[cur_com] == 1 && best_com > cur_com)
        { 

            best_com = cur_com;
        }
        else
        {

            if (lane_id == neighbor_num % warp_size)
            {
                atomicAdd(&Tot_update[best_com], Ki);
                atomicAdd(&com_size_update[best_com], 1);
                atomicSub(&Tot_update[cur_com], Ki);
                atomicSub(&com_size_update[cur_com], 1);
            }
        }
    }
    else
    {
        best_com = cur_com;
    }

    if (lane_id == 0)
    {
        cur_community[vertex] = best_com;
    }

    if (best_com != cur_com)
    { 
        if (lane_id == 0)
        {
            is_moved[vertex] = 1;
            next_In[vertex] = best_com_weight; // weights of new community and v
        }

        for (int thd = lane_id; thd < neighbor_num; thd += warp_size)
        {
            int v = neighbors[start_neighbor + thd];
            int wt = weights[start_neighbor + thd];
            atomicOr(&active_set[v], 1); //
            if (prev_community[v] == cur_com)
            {
                atomicSub(&target_com_weights[v], wt);
            }
            else if (prev_community[v] == best_com)
            {
                atomicAdd(&target_com_weights[v], wt);
            }
        }
    }

    // update community
}

// don't use share memory, degree must < warp_size
__global__ void decide_and_move_shuffle(int *sorted_vertex_id, const weight_t *__restrict__ weights, const vertex_t *__restrict__ neighbors, const edge_t *__restrict__ degrees,
                                        int *prev_community, int *cur_community,
                                        int *K, const int *__restrict__ Tot, int *In, int *next_In, int *Self,
                                        int *com_size, int *Tot_update, int *com_size_update,
                                        int vertex_num_to_proc, int warp_size, double constant,
                                        int *active_set, int *is_moved, int *target_com_weights,
                                        int iteration)
{ // warp_size=threadAlloc
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int warp_id = thread_id / warp_size; // vertex_id,warp_size may be 4,8,16...
    int warp_num = blockDim.x * gridDim.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    for (int wp = warp_id; wp < vertex_num_to_proc; wp += warp_num)
    {
        int vertex = sorted_vertex_id[wp];
        edge_t start_neighbor = degrees[vertex];
        edge_t end_neighbor = degrees[vertex + 1];
        int cur_com = prev_community[vertex];
        int neighbor_num = end_neighbor - start_neighbor;
        build_and_select_in_warp(vertex, weights, neighbors, degrees, In, next_In, Tot, Self,
                                 K[vertex], constant, prev_community, cur_community,
                                 com_size, com_size_update, Tot_update,
                                 cur_com, start_neighbor, end_neighbor, neighbor_num, lane_id,
                                 warp_size, active_set, is_moved, target_com_weights, iteration);
    }
}

__device__ void build_and_select(int vertex, const weight_t *__restrict__ weights, const vertex_t *__restrict__ neighbors, const edge_t *__restrict__ degrees,
                                 int *In, int *next_In, const int *__restrict__ Tot, int *Self, int Ki, double constant,
                                 int *prev_community, int *cur_community, int *com_size,
                                 int *com_size_update, int *Tot_update,
                                 int *neighbor_com_ids, int *neighbor_com_weights, int *shared_Tot,
                                 int cur_com, edge_t start_neighbor, edge_t end_neighbor, int neighbor_num,
                                 int lane_id, int warp_size, int table_size, int is_global,
                                 int *active_set, int *is_moved, int *target_com_weights)
{ // for each neighbor,compute gain

    double maxdq = 0;
    int best_com = cur_com;
    // build map
    int com_v, wt;

    // int v=vertex;

    int tot_cur_com = Tot[cur_com];
    for (int thd = lane_id; thd <= neighbor_num; thd += warp_size)
    {
        if (thd == 0)
        {
            com_v = cur_com;
            wt = 0;
        }
        else
        { 
            int v = neighbors[start_neighbor + thd - 1];
            com_v = prev_community[v];
            wt = weights[start_neighbor + thd - 1];
        }

        insert_neighbor_com_info(neighbor_com_ids, neighbor_com_weights, shared_Tot,
                                 nullptr, nullptr, table_size, 0, com_v, wt, Tot, Ki, cur_com, tot_cur_com,
                                 &maxdq, &best_com, constant, vertex, is_global);
    }

    __syncwarp();

    // select best community
    int cur_com_weight = find_neighbor_com_info(neighbor_com_ids, neighbor_com_weights, shared_Tot,
                                                nullptr, nullptr, table_size, 0, is_global, cur_com);
    if (lane_id == 0)
    {
        In[vertex] = cur_com_weight; // atomic??????
    }

    for (int i = warp_size / 2; i >= 1; i = i / 2)
    {
        double recv_dq = __shfl_xor_sync(0xffffffff, maxdq, i, warp_size); // only for warp!!!!!!!!!!!!!!
        int recv_com = __shfl_xor_sync(0xffffffff, best_com, i, warp_size);
        if ((recv_dq > maxdq) || (recv_dq == maxdq && recv_com < best_com))
        {

            maxdq = recv_dq;
            best_com = recv_com;
        }
    }

    maxdq = maxdq - 2 * cur_com_weight + 2 * Self[vertex];

    if (best_com >= 0 && best_com != cur_com && maxdq > 0)
    {
        if (com_size[best_com] == 1 && com_size[cur_com] == 1 && best_com > cur_com)
        { 
            best_com = cur_com;
        }
        else
        {

            if (lane_id == neighbor_num % warp_size)
            {
                atomicAdd(&Tot_update[best_com], Ki);
                atomicAdd(&com_size_update[best_com], 1);
                atomicSub(&Tot_update[cur_com], Ki);
                atomicSub(&com_size_update[cur_com], 1);
            }
        }
    }
    else
    {
        best_com = cur_com;
    }
    // __syncwarp();//!!!!!!!
    int best_com_weight = find_neighbor_com_info(neighbor_com_ids, neighbor_com_weights, shared_Tot,
                                                 nullptr, nullptr, table_size, 0, is_global, best_com);

    if (lane_id == 0)
    {
        cur_community[vertex] = best_com;
        next_In[vertex] = best_com_weight; // may improve
    }

    if (best_com != cur_com)
    { // prune
        if (lane_id == 0)
        {
            is_moved[vertex] = 1;
        }

        for (int thd = lane_id; thd < neighbor_num; thd += warp_size)
        {
            int v = neighbors[start_neighbor + thd];
            int wt = weights[start_neighbor + thd];
            atomicOr(&active_set[v], 1); //
            if (prev_community[v] == cur_com)
            {
                atomicSub(&target_com_weights[v], wt);
            }
            else if (prev_community[v] == best_com)
            {
                atomicAdd(&target_com_weights[v], wt);
            }
        }
    }

    // update community
}

// use share memory
__global__ void decide_and_move_hash_shared(int *sorted_vertex_id, const weight_t *__restrict__ weights, const vertex_t *__restrict__ neighbors, const edge_t *__restrict__ degrees,
                                            int *prev_community, int *cur_community,
                                            int *K, const int *__restrict__ Tot, int *In, int *next_In, int *Self,
                                            int *com_size, int *Tot_update, int *com_size_update,
                                            int vertex_num_to_proc, int table_size, int warp_size, double constant,
                                            int *active_set, int *is_moved, int *target_com_weights, int iteration)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ int neighbor_com_info[];
    int shared_size = table_size * (blockDim.x / warp_size); // total share memory size

    int warp_id = thread_id / warp_size;
    int warp_num = blockDim.x * gridDim.x / warp_size;
    int lane_id = threadIdx.x % warp_size;
    int *neighbor_com_ids = neighbor_com_info + (threadIdx.x / warp_size) * table_size;
    int *neighbor_com_weights = neighbor_com_info + shared_size + (threadIdx.x / warp_size) * table_size;
    int *shared_Tot = neighbor_com_info + 2 * shared_size + (threadIdx.x / warp_size) * table_size;
    int is_global = 0;

    for (int wp = warp_id; wp < vertex_num_to_proc; wp += warp_num)
    {
        int vertex = sorted_vertex_id[wp];
        edge_t start_neighbor = degrees[vertex];
        edge_t end_neighbor = degrees[vertex + 1];
        int cur_com = prev_community[vertex];
        int neighbor_num = end_neighbor - start_neighbor;
        for (int i = lane_id; i < table_size; i += warp_size)
        {
            neighbor_com_ids[i] = 0;
            neighbor_com_weights[i] = 0;
            shared_Tot[i] = 0;
        }
        __syncthreads();
        build_and_select(vertex, weights, neighbors, degrees, In, next_In, Tot, Self,
                         K[vertex], constant, prev_community, cur_community,
                         com_size, com_size_update, Tot_update,
                         neighbor_com_ids, neighbor_com_weights, shared_Tot, cur_com,
                         start_neighbor, end_neighbor, neighbor_num, lane_id, warp_size, table_size, is_global,
                         active_set, is_moved, target_com_weights);
        __syncthreads();
    }
}

__device__ void build_and_select_blk(int vertex, const weight_t *__restrict__ weights, const vertex_t *__restrict__ neighbors, const edge_t *__restrict__ degrees,
                                     int *In, int *next_In, const int *__restrict__ Tot, int *Self, int Ki, double constant,
                                     int *prev_community, int *cur_community, int *com_size,
                                     int *com_size_update, int *Tot_update,
                                     int *shared_neighbor_com_ids, int *shared_neighbor_com_weights, int *shared_Tot,
                                     int *global_neighbor_com_ids, int *global_neighbor_com_weights,
                                     int cur_com, edge_t start_neighbor, int neighbor_num,
                                     int thread_id_in_blk, int thread_numInBlk, int warp_size, int shared_size, int global_size, int is_global,
                                     int *active_set, int *is_moved, int *target_com_weights, int iteration)
{ // for each neighbor,compute gain
    double maxdq = 0;
    int best_com = cur_com;
    // init hash table
    int com_v, wt;
    int lane_id = threadIdx.x % warp_size;
    int tot_cur_com = Tot[cur_com];
    for (int thd = thread_id_in_blk; thd <= neighbor_num; thd += thread_numInBlk)
    {
        if (thd == 0)
        {
            com_v = cur_com;
            wt = 0;
        }
        else
        { 
            int v = neighbors[start_neighbor + thd - 1];
            com_v = prev_community[v];
            wt = weights[start_neighbor + thd - 1];
        }

        insert_neighbor_com_info(shared_neighbor_com_ids, shared_neighbor_com_weights, shared_Tot,
                                 global_neighbor_com_ids, global_neighbor_com_weights,
                                 shared_size, global_size, com_v, wt, Tot, Ki, cur_com, tot_cur_com,
                                 &maxdq, &best_com, constant, vertex, is_global);
    }


    __syncthreads(); 

    int cur_com_weight = find_neighbor_com_info(shared_neighbor_com_ids, shared_neighbor_com_weights, shared_Tot,
                                                global_neighbor_com_ids, global_neighbor_com_weights,
                                                shared_size, global_size, is_global, cur_com);
    if (thread_id_in_blk == 0)
    {
        In[vertex] = cur_com_weight; 
    }

    // select best community

    int warp_numPerBlk = (blockDim.x + warp_size - 1) / warp_size;
    int warp_id = threadIdx.x / warp_size;

    if (warp_numPerBlk <= WARP_SIZE)
    { 
        // select best in warp
        for (int i = warp_size / 2; i >= 1; i = i / 2)
        {
            double recv_dq = __shfl_xor_sync(0xffffffff, maxdq, i, warp_size); 
            int recv_com = __shfl_xor_sync(0xffffffff, best_com, i, warp_size);
            if ((recv_dq > maxdq) || (recv_dq == maxdq && recv_com < best_com))
            {
                maxdq = recv_dq;
                best_com = recv_com;
            }
        }

        __shared__ double shared_to_compare[64]; // may exceed share mem limit?

        volatile double *best_each_warp = shared_to_compare;
        if (lane_id == 0)
        {
            best_each_warp[warp_id] = maxdq;
            best_each_warp[32 + warp_id] = (double)best_com;
        }
        __syncthreads();
        best_com = -1;
        maxdq = 0;

        if (lane_id < warp_numPerBlk)
        {
            maxdq = best_each_warp[lane_id]; 
            best_com = (int)best_each_warp[lane_id + 32];
        }
        for (int i = warp_size / 2; i >= 1; i = i / 2)
        {
            double recv_dq = __shfl_xor_sync(0xffffffff, maxdq, i, warp_size); // only for warp
            int recv_com = __shfl_xor_sync(0xffffffff, best_com, i, warp_size);
            if ((recv_dq > maxdq) || (recv_dq == maxdq && recv_com < best_com))
            {
                maxdq = recv_dq;
                best_com = recv_com;
            }
        }
    }
    else
    {
        if (thread_id_in_blk == 0)
        {
            printf("warp_numPerBlk:%d\n", warp_numPerBlk);
        }
    }

    maxdq = maxdq - 2 * cur_com_weight + 2 * Self[vertex];

    if (best_com >= 0 && best_com != cur_com && maxdq > 0)
    {
        if (com_size[best_com] == 1 && com_size[cur_com] == 1 && best_com > cur_com)
        { //!!!!!!!!!!!!
            best_com = cur_com;
        }
        else
        {

            if (thread_id_in_blk == neighbor_num % thread_numInBlk)
            {
                atomicAdd(&Tot_update[best_com], Ki);
                atomicAdd(&com_size_update[best_com], 1);
                atomicSub(&Tot_update[cur_com], Ki);
                atomicSub(&com_size_update[cur_com], 1);
            }
        }
    }
    else
    {
        best_com = cur_com;
    }

    // __syncthreads();
    int best_com_weight = find_neighbor_com_info(shared_neighbor_com_ids, shared_neighbor_com_weights, shared_Tot,
                                                 global_neighbor_com_ids, global_neighbor_com_weights,
                                                 shared_size, global_size, is_global, best_com);

    if (thread_id_in_blk == 0)
    {
        cur_community[vertex] = best_com;
        next_In[vertex] = best_com_weight;
    }

    if (best_com != cur_com)
    { // prune
        if (thread_id_in_blk == 0)
        {
            is_moved[vertex] = 1;
        }

        for (int thd = thread_id_in_blk; thd < neighbor_num; thd += thread_numInBlk)
        {
            int v = neighbors[start_neighbor + thd];
            int wt = weights[start_neighbor + thd];
            atomicOr(&active_set[v], 1); //
            if (prev_community[v] == cur_com)
            {
                atomicSub(&target_com_weights[v], wt);
            }
            else if (prev_community[v] == best_com)
            {
                atomicAdd(&target_com_weights[v], wt);
            }
        }
    }

    // update community
}

__global__ void decide_and_move_hash_hierarchical(int *sorted_vertex_id, const weight_t *__restrict__ weights, const vertex_t *__restrict__ neighbors, const edge_t *__restrict__ degrees,
                                                  int *prev_community, int *cur_community,
                                                  int *K, const int *__restrict__ Tot, int *In, int *next_In, int *Self,
                                                  int *com_size, int *Tot_update, int *com_size_update,
                                                  int *global_table_ptr, int *glbTable, int *primes, int prime_num,
                                                  int vertex_num_to_proc, int warp_size, int global_limit, double constant,
                                                  int *active_set, int *is_moved, int *target_com_weights, int iteration)
{ // warp_size=threadAlloc

    __shared__ int neighbor_com_info[SHARE_MEM_SIZE * 3];
    int shared_size = SHARE_MEM_SIZE; // total share memory size
    // int table_size;
    int block_id = blockIdx.x; // vertex_id
    int block_num = gridDim.x;
    // int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int *shared_neighbor_com_ids = neighbor_com_info; // share memory:k-v-tot
    int *shared_neighbor_com_weights = neighbor_com_info + shared_size;
    int *shared_Tot = neighbor_com_info + 2 * shared_size;
    int *global_neighbor_com_ids; // global memory:k-v
    int *global_neighbor_com_weights;
    int global_size = 0;
    int is_global = 0;
    for (int wp = block_id; wp < vertex_num_to_proc; wp += block_num)
    {

        int vertex = sorted_vertex_id[wp];
        edge_t start_neighbor = degrees[vertex];
        edge_t end_neighbor = degrees[vertex + 1];
        int cur_com = prev_community[vertex];
        int neighbor_num = end_neighbor - start_neighbor;
        if (neighbor_num > global_limit)
        { // global memory
            int start_glb_table = global_table_ptr[wp];
            int global_half_size = global_table_ptr[vertex_num_to_proc]; // vertex num whose deg>4047
            global_neighbor_com_ids = &glbTable[start_glb_table];
            global_neighbor_com_weights = &glbTable[start_glb_table + global_half_size];
            is_global = 1;
            global_size = global_table_ptr[wp + 1] - start_glb_table;
            for (unsigned int i = threadIdx.x; i < global_size; i = i + blockDim.x)
            {
                global_neighbor_com_ids[i] = 0;
                global_neighbor_com_weights[i] = 0;
            }
        }

        for (unsigned int i = threadIdx.x; i < shared_size * 3; i = i + blockDim.x)
        {
            neighbor_com_info[i] = 0;
        }

        __syncthreads();
        int Ki = K[vertex];
        build_and_select_blk(vertex, weights, neighbors, degrees, In, next_In, Tot, Self,
                             Ki, constant, prev_community, cur_community,
                             com_size, com_size_update, Tot_update,
                             shared_neighbor_com_ids, shared_neighbor_com_weights, shared_Tot,
                             global_neighbor_com_ids, global_neighbor_com_weights,
                             cur_com, start_neighbor, neighbor_num, threadIdx.x, blockDim.x, warp_size, shared_size, global_size, is_global,
                             active_set, is_moved, target_com_weights, iteration);
        __syncthreads(); //
    }
}

// compute In-list
__global__ void compute_In(int *sorted_vertex_id, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                           int *cur_community,
                           int *K, int *Tot, int *In, int *Self,
                           int vertex_num_to_proc, int warp_size, double constant, int min_Tot,
                           int *active_set, int pruning_method)
{ // warp_size=threadAlloc
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int warp_id = thread_id / warp_size; // vertex_id,warp_size may be 4,8,16...
    int warp_num = blockDim.x * gridDim.x / warp_size;
    int lane_id = threadIdx.x % warp_size;

    for (int wp = warp_id; wp < vertex_num_to_proc; wp += warp_num)
    {
        int vertex = sorted_vertex_id[wp];
        edge_t start_neighbor = degrees[vertex];
        edge_t end_neighbor = degrees[vertex + 1];
        int cur_com = cur_community[vertex];
        int neighbor_num = end_neighbor - start_neighbor;
        int Ki = K[vertex];
        // build map
        int com_v, wt;
        int cur_com_weight = 0;
        for (int thd = lane_id; thd < neighbor_num; thd += warp_size)
        {
            int v = neighbors[start_neighbor + thd];
            com_v = cur_community[v];
            wt = weights[start_neighbor + thd];
            if (com_v == cur_com)
            {
                cur_com_weight += wt;
            }
        }
        // __syncthreads(); 
        // reduce
        // cur_com_weight=__reduce_add_sync(0xffffffff,cur_com_weight);
        for (int i = 1; i <= warp_size / 2; i *= 2)
        {
            int value = __shfl_up_sync(0xffffffff, cur_com_weight, i, warp_size);
            if (lane_id >= i)
                cur_com_weight += value;
        }


        if (lane_id == warp_size - 1)
        {
            In[vertex] = cur_com_weight;
            if ((pruning_method == 0 || pruning_method == 3) && 
                2 * (cur_com_weight - Self[vertex]) - (Ki - Self[vertex]) + (double)(min_Tot - (Tot[cur_com] - Ki)) * constant * Ki > 0)
            {
                active_set[vertex] = 0;
                // return ;
            }
        }
    }
}

// compute In-list
__global__ void compute_In_blk(int *sorted_vertex_id, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                               int *cur_community,
                               int *K, int *Tot, int *In, int *Self,
                               int vertex_num_to_proc, int warp_size, double constant, int min_Tot,
                               int *active_set, int pruning_method)
{ // warp_size=threadAlloc

    int block_id = blockIdx.x; // vertex_id
    int block_num = gridDim.x;

    for (int wp = block_id; wp < vertex_num_to_proc; wp += block_num)
    {
        int vertex = sorted_vertex_id[wp];
        edge_t start_neighbor = degrees[vertex];
        edge_t end_neighbor = degrees[vertex + 1];
        int cur_com = cur_community[vertex];
        int neighbor_num = end_neighbor - start_neighbor;
        int Ki = K[vertex];
        int com_v, wt;
        int cur_com_weight = 0;
        for (int thd = threadIdx.x; thd < neighbor_num; thd += blockDim.x)
        {
            int v = neighbors[start_neighbor + thd];
            com_v = cur_community[v];
            wt = weights[start_neighbor + thd];
            if (com_v == cur_com)
            {
                cur_com_weight += wt;
            }
        }
        // __syncthreads(); //??????????????????
        // reduce
        int lane_id = threadIdx.x % warp_size;
        int warp_num_in_blk = blockDim.x / warp_size;

        for (int i = 1; i <= warp_size / 2; i *= 2)
        { // total neighbor num of a warp(merge vertex)

            int value = __shfl_up_sync(0xffffffff, cur_com_weight, i, warp_size);

            if (lane_id >= i)
                cur_com_weight += value;
        }
        // __syncthreads();

        __shared__ int shared_to_sum[32]; // blockdim:1024 warp:32 must!!!!
        volatile int *sum_each_warp = shared_to_sum;
        if (lane_id == warp_size - 1)
        {
            sum_each_warp[threadIdx.x / warp_size] = cur_com_weight;
        }

        __syncthreads();
        int sum = 0;
        if (lane_id < warp_num_in_blk)
        {
            sum = sum_each_warp[lane_id];
        }

        for (int i = 1; i <= warp_size / 2; i *= 2)
        { // total cur_com_weight of a block
            int value = __shfl_up_sync(0xffffffff, sum, i, warp_size);
            if (lane_id >= i)
                sum += value;
        }

        if (threadIdx.x == blockDim.x - 1)
        {
            In[vertex] = sum;
            if ((pruning_method == 0 || pruning_method == 3) && 
                2 * (sum - Self[vertex]) - (Ki - Self[vertex]) + (double)(min_Tot - (Tot[cur_com] - Ki)) * constant * Ki > 0)
            {
                active_set[vertex] = 0;
            }
        }
    }
}

// get modularity
// __host__ double modularity(int *weights, int *neighbors, int *degrees, int *community, int vertex_num, double constant /*,int* tmp_In,int* tmp_Q*/)
// {
//     int *Tot = new int[vertex_num]; // comunity-tot 
//     int *In = new int[vertex_num];  // comunity-in 
//     // int* Self=new int[vertex_num];
//     for (int i = 0; i < vertex_num; i++)
//     {
//         In[i] = 0;
//         Tot[i] = 0;
//     }

//     int m = 0; // number of edges

//     for (int i = 0; i < vertex_num; i++)
//     {
//         int degree = 0;
//         for (int j = (i == 0 ? 0 : degrees[i - 1]); j < degrees[i]; j++)
//         {
//             degree += weights[j];
//             if (community[neighbors[j]] == community[i])
//             {
//                 In[i] += weights[j];
//             }
//         }

//         Tot[community[i]] += degree;
//     }

//     double q = 0;
//     for (int i = 0; i < vertex_num; i++)
//     {

//         q += (double)In[i] * constant - (double)((long)Tot[i]) * ((long)Tot[i]) * constant * constant;
//     }

//     // cout<<"In:";
//     // for(int i=0;i<vertex_num;i++){
//     //     if(In[i]!=tmp_In[i])
//     //         cout<<i<<":"<<In[i]<<" "<<tmp_In[i]<<" "<<tmp_Q[i]<<endl;
//     // }cout<<endl;

//     return q;
// }

// build graph functions start

__global__ void renumber(int *renumber, vertex_t *out, int vertex_num)
{
    int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
    if (thread_id < vertex_num)
        out[thread_id] = renumber[out[thread_id]];
}

__global__ void get_size_of_each_community(vertex_t *community, int *size_of_community, int vertex_num)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_num = blockDim.x * gridDim.x;
    for (int i = thread_id; i < vertex_num; i += thread_num)
    {
        int c = community[i];
        atomicAdd(&size_of_community[c], 1);
    }
}

__global__ void gather_vertex_by_community(int *gathered_vertex_by_community, int *community, int *pos_ptr, int vertex_num)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int thread_num = blockDim.x * gridDim.x;
    for (int i = thread_id; i < vertex_num; i += thread_num)
    {
        int c = community[i];
        int curPos = atomicAdd(&pos_ptr[c], 1);
        gathered_vertex_by_community[curPos] = i;
    }
}

__global__ void compute_edge_num_of_each_community(edge_t *degrees, int *gathered_vertex_by_community, int *gathered_vertex_ptr,
                                                   edge_t *edge_num_of_community, int community_num, int warp_size)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;
    int lane_id = threadIdx.x % warp_size;

    edge_t num = 0;
    if (warp_id < community_num)
    { // one warp-one communuty;one thread-one vertex
        int start = gathered_vertex_ptr[warp_id];
        int end = gathered_vertex_ptr[warp_id + 1];

        for (int thd = lane_id; thd < (end - start); thd += warp_size)
        {

            int v = gathered_vertex_by_community[start + thd];
            // edge_numForWarp[lane_id]+=(degrees[v+1]-degrees[v]);
            num += (degrees[v + 1] - degrees[v]);
        }
    }

    for (int i = 1; i <= warp_size / 2; i *= 2)
    { // total neighbor num of a comminuty(merge vertex)
        edge_t value = __shfl_up_sync(0xffffffff, num, i, warp_size);
        if (lane_id >= i)
            num += value;
    }

    // __syncthreads();
    if (warp_id < community_num && lane_id == warp_size - 1) //!!!!!!!!!!!!!
    {
        edge_num_of_community[warp_id] = num;
    }
}

__global__ void compute_global_table_size(int *primes, int prime_num, edge_t *edge_num_of, int deg_num_glb, int *global_table_ptr, int warp_size,
                                          int community_num)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;

    while (warp_id < deg_num_glb)
    {
        edge_t edge_num = edge_num_of[warp_id];
        int nearest_prime = find_prime_by_warp(primes, prime_num, edge_num, warp_size);
        if (edge_num >= community_num)
        {
            global_table_ptr[warp_id] = community_num;
        }
        else
        {
            global_table_ptr[warp_id] = nearest_prime;
        }

        warp_id += (blockDim.x * gridDim.x) / WARP_SIZE;
    }
}

__device__ int insert_neighbor_com_info_for_com(int *neighbor_com_ids, int *neighbor_com_weights,
                                                int table_size, int neighbor_com, int neighbor_weight)
{ // insert into hashmap
    int hash_value1 = neighbor_com % table_size;
    int hash_value2 = 1 + (neighbor_com % (table_size - 1));

    int i = 0;
    int pos;
    while (i < table_size)
    { 
        pos = (hash_value1 + i * hash_value2) % table_size;

        int com_id = atomicCAS(&neighbor_com_ids[pos], 0, (1 + neighbor_com)); // 0 means no element in this position and swap

        if (com_id == 0)
        { // this position is vacant
            atomicAdd((int *)&neighbor_com_weights[pos], neighbor_weight);

            return table_size;
        }
        else if (com_id == (neighbor_com + 1))
        { // code may be redundant?????????
            atomicAdd((int *)&neighbor_com_weights[pos], neighbor_weight);

            return pos;
        }
        else
        {
            i += 1;
        }
    }

    return -1;
}

__device__ int find_neighbor_com_info_for_com(int *neighbor_com_ids, int *neighbor_com_weights,
                                              int table_size, int vertex_com)
{ // find in hashmap
    int hash_value1 = vertex_com % table_size;
    int hash_value2 = 1 + (vertex_com % (table_size - 1));
    int i = 0;
    int pos;
    do
    {
        pos = (hash_value1 + i * hash_value2) % table_size;
        int com_id = neighbor_com_ids[pos];
        if (com_id == (vertex_com + 1))
        {
            return pos;
        }
        else
        {
            i += 1;
        }
    } while (i < table_size);

    return -1;
}

__device__ void compute_neighbors_weights(int com, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                                          weight_t *new_weights, vertex_t *new_neighbors, edge_t *new_degrees,
                                          int *community, int *gathered_vertex_ptr, int *gathered_vertex_by_community,
                                          int *neighbor_com_ids, int *neighbor_com_weights,
                                          int lane_id, int table_size, int warp_size)
{

    int start_vertex = gathered_vertex_ptr[com];
    int end_vertex = gathered_vertex_ptr[com + 1]; // get the position of vertexs in com
    int neighbor_com_num = 0;                      //
    for (int i = start_vertex; i < end_vertex; i++)
    { //
        int v = gathered_vertex_by_community[i];
        edge_t start_neighbor = degrees[v];
        edge_t end_neighbor = degrees[v + 1];

        for (int j = lane_id; j < (end_neighbor - start_neighbor); j += warp_size)
        {
            int neighbor_v = neighbors[start_neighbor + j];
            int neighbor_com = community[neighbor_v];
            int neighbor_weight = weights[start_neighbor + j]; //!!!!!!!!!!!!!!!!!!!!!!
            int res = insert_neighbor_com_info_for_com(neighbor_com_ids, neighbor_com_weights, table_size, neighbor_com, neighbor_weight);
            if (res == table_size)
                neighbor_com_num += 1;
            res = (res == table_size) ? -1 : 1;
            neighbors[start_neighbor + j] = (neighbors[start_neighbor + j] + 1) * res; // 首次插入hashmap某位置的neighbors[start_neighbor+i]为负
        }
    }
    // __syncthreads();

    // inclusive scan
    int pos = neighbor_com_num;
    for (int i = 1; i <= warp_size / 2; i *= 2)
    { // total neighbor num of a comminuty(merge vertex)
        int value = __shfl_up_sync(0xffffffff, pos, i, warp_size);
        if (lane_id >= i)
            pos += value;
    }
    // __syncthreads();

    if (lane_id == warp_size - 1)
    { // save edge num in the community

        new_degrees[com] = pos;
    }

    pos = pos - neighbor_com_num; // the positon of new neighborcom

    for (int i = start_vertex; i < end_vertex; i++)
    { //
        int v = gathered_vertex_by_community[i];
        edge_t start_neighbor = degrees[v];
        edge_t end_neighbor = degrees[v + 1];

        for (int j = lane_id; j < (end_neighbor - start_neighbor); j += warp_size)
        {
            int neighbor_v = neighbors[start_neighbor + j];
            int flag = 1;
            if (neighbor_v < 0)
            {
                flag = -1;
            }
            neighbors[start_neighbor + j] = neighbor_v = neighbor_v * flag - 1;
            int neighbor_com = community[neighbor_v];
            int map_pos = find_neighbor_com_info_for_com(neighbor_com_ids, neighbor_com_weights, table_size, neighbor_com);
            if (map_pos >= 0 && flag == -1)
            { // each neighborcom is only visited once
                new_neighbors[pos] = neighbor_com_ids[map_pos] - 1;
                new_weights[pos] = neighbor_com_weights[map_pos];
                pos = pos + 1;
            }
        }
    }
}

__device__ void compute_neighbors_weights_blk(int com, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                                              weight_t *new_weights, vertex_t *new_neighbors, edge_t *new_degrees,
                                              int *community, int *gathered_vertex_ptr, int *gathered_vertex_by_community,
                                              int *neighbor_com_ids, int *neighbor_com_weights,
                                              int thread_id_in_blk, int table_size, int warp_size)
{
    int start_vertex = gathered_vertex_ptr[com];
    int end_vertex = gathered_vertex_ptr[com + 1]; // get the position of vertexs in com
    int neighbor_com_num = 0;                      //

    for (int i = start_vertex; i < end_vertex; i++)
    { //
        int v = gathered_vertex_by_community[i];
        edge_t start_neighbor = degrees[v];
        edge_t end_neighbor = degrees[v + 1];

        for (int j = threadIdx.x; j < (end_neighbor - start_neighbor); j += blockDim.x)
        {
            int neighbor_v = neighbors[start_neighbor + j];
            int neighbor_com = community[neighbor_v];
            int neighbor_weight = weights[start_neighbor + j]; //!!!!!!!!!!!!!
            int res = insert_neighbor_com_info_for_com(neighbor_com_ids, neighbor_com_weights, table_size, neighbor_com, neighbor_weight);

            // __syncthreads();//????????????

            if (res == table_size)
                neighbor_com_num += 1;
            res = (res == table_size) ? -1 : 1;

            neighbors[start_neighbor + j] = (neighbors[start_neighbor + j] + 1) * res; //
            // __syncthreads();
        }
    }
    // __syncthreads(); //????????????

    // inclusive scan block
    int pos = neighbor_com_num;

    int lane_id = threadIdx.x % warp_size;
    int warp_num_in_blk = blockDim.x / warp_size;

    for (int i = 1; i <= warp_size / 2; i *= 2)
    { // total neighbor num of a warp(merge vertex)

        int value = __shfl_up_sync(0xffffffff, pos, i, warp_size);

        if (lane_id >= i)
            pos += value;
    }
    // __syncthreads();

    __shared__ int shared_to_sum[32]; // blockdim:1024 warp:32 must!!!!
    int *sum_each_warp = shared_to_sum;
    if (lane_id == warp_size - 1)
    {
        sum_each_warp[threadIdx.x / warp_size] = pos;
    }

    __syncthreads();
    int sum = 0;
    if (lane_id < warp_num_in_blk)
    {
        sum = sum_each_warp[lane_id];
    }

    for (int i = 1; i <= warp_size / 2; i *= 2)
    { // total neighbor num of a block(merge vertex)
        int value = __shfl_up_sync(0xffffffff, sum, i, warp_size);
        if (lane_id >= i)
            sum += value;
    }

    if (threadIdx.x == blockDim.x - 1)
    {
        // printf("%d %d %d\n",blockIdx.x,com,sum);
        new_degrees[com] = sum;
    }

    int warp_id = threadIdx.x / warp_size;

    int recv_sum = __shfl_sync(0xffffffff, sum, warp_id - 1, warp_size);
    if (warp_id >= 1)
    {
        pos += recv_sum;
    }

    pos = pos - neighbor_com_num; // the positon of new neighborcom

    // __syncthreads();

    for (int i = start_vertex; i < end_vertex; i++)
    { //

        int v = gathered_vertex_by_community[i];

        edge_t start_neighbor = degrees[v];
        edge_t end_neighbor = degrees[v + 1];

        for (int j = thread_id_in_blk; j < (end_neighbor - start_neighbor); j += blockDim.x)
        {
            int neighbor_v = neighbors[start_neighbor + j];
            int flag = 1;
            if (neighbor_v < 0)
            {
                flag = -1;
            }
            neighbors[start_neighbor + j] = neighbor_v = neighbor_v * flag - 1;
            int neighbor_com = community[neighbor_v];
            int map_pos = find_neighbor_com_info_for_com(neighbor_com_ids, neighbor_com_weights, table_size, neighbor_com);

            if (map_pos >= 0 && flag == -1)
            { // each neighborcom is only visited once

                new_neighbors[pos] = neighbor_com_ids[map_pos] - 1;
                new_weights[pos] = neighbor_com_weights[map_pos];
                pos = pos + 1;
            }
        }
    }
}

__device__ void compute_neighbors_weights_list(int com, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                                               weight_t *new_weights, vertex_t *new_neighbors, edge_t *new_degrees,
                                               int *community, int *gathered_vertex_ptr, int *gathered_vertex_by_community,
                                               int *neighbor_com_ids, int *neighbor_com_weights,
                                               int lane_id, int community_num, int warp_size)
{

    int start_vertex = gathered_vertex_ptr[com];
    int end_vertex = gathered_vertex_ptr[com + 1]; // get the position of vertexs in com
                                                   //
    for (int i = start_vertex; i < end_vertex; i++)
    { //
        int v = gathered_vertex_by_community[i];
        edge_t start_neighbor = degrees[v];
        edge_t end_neighbor = degrees[v + 1];

        for (int j = lane_id; j < (end_neighbor - start_neighbor); j += warp_size)
        {
            int neighbor_v = neighbors[start_neighbor + j];
            int neighbor_com = community[neighbor_v];
            int neighbor_weight = weights[start_neighbor + j]; //!!!!!!!!!!!!!!!!!!!!!!
            atomicAdd(&neighbor_com_weights[neighbor_com], neighbor_weight);
        }
    }
    __syncthreads();
    int neighbor_num = 0;
    for (int i = lane_id; i < community_num; i += warp_size)
    {
        if (neighbor_com_weights[i] > 0)
        {
            new_neighbors[i] = i;
            new_weights[i] = neighbor_com_weights[i];
            neighbor_num += 1;
        }
    }

    int sum = __reduce_add_sync(0xffffffff, neighbor_num);
    if (lane_id == 0)
    {
        new_degrees[com] = sum;
        // printf("2lim:%d %d \n",com,sum);
    }
}

__device__ void compute_neighbors_weights_blk_list(int com, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                                                   weight_t *new_weights, vertex_t *new_neighbors, edge_t *new_degrees,
                                                   int *community, int *gathered_vertex_ptr, int *gathered_vertex_by_community,
                                                   int *neighbor_com_ids, int *neighbor_com_weights,
                                                   int thread_id_in_blk, int community_num, int warp_size)
{
    int start_vertex = gathered_vertex_ptr[com];
    int end_vertex = gathered_vertex_ptr[com + 1]; // get the position of vertexs in com
    int neighbor_com_num = 0;                      //

    for (int i = start_vertex; i < end_vertex; i++)
    { //
        int v = gathered_vertex_by_community[i];
        edge_t start_neighbor = degrees[v];
        edge_t end_neighbor = degrees[v + 1];

        for (int j = threadIdx.x; j < (end_neighbor - start_neighbor); j += blockDim.x)
        {
            int neighbor_v = neighbors[start_neighbor + j];
            // printf("%d %d\n",blockIdx.x,neighbor_v);
            int neighbor_com = community[neighbor_v];
            int neighbor_weight = weights[start_neighbor + j]; //!!!!!!!!!!!!!

            atomicAdd(&neighbor_com_weights[neighbor_com], neighbor_weight);

            // __syncthreads();
        }
    }
    __syncthreads();

    for (int i = thread_id_in_blk; i < community_num; i += blockDim.x)
    {
        if (neighbor_com_weights[i] > 0)
        {
            new_neighbors[i] = i;
            new_weights[i] = neighbor_com_weights[i];
            neighbor_com_num += 1;
        }
    }
    int lane_id = threadIdx.x % warp_size;
    int warp_num_in_blk = blockDim.x / warp_size;

    for (int i = 1; i <= warp_size / 2; i *= 2)
    { // total neighbor num of a warp(merge vertex)

        int value = __shfl_up_sync(0xffffffff, neighbor_com_num, i, warp_size);

        if (lane_id >= i)
            neighbor_com_num += value;
    }
    // __syncthreads();

    __shared__ int shared_to_sum[32]; // blockdim:1024 warp:32 must!!!!
    int *sum_each_warp = shared_to_sum;
    if (lane_id == warp_size - 1)
    {
        sum_each_warp[threadIdx.x / warp_size] = neighbor_com_num;
    }

    __syncthreads();
    int sum = 0;
    if (lane_id < warp_num_in_blk)
    {
        sum = sum_each_warp[lane_id];
    }

    for (int i = 1; i <= warp_size / 2; i *= 2)
    { // total neighbor num of a block(merge vertex)
        int value = __shfl_up_sync(0xffffffff, sum, i, warp_size);
        if (lane_id >= i)
            sum += value;
    }

    if (threadIdx.x == blockDim.x - 1)
    {
        new_degrees[com] = sum;
    }
}

__global__ void build_by_warp(int *sorted_community_id, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                              weight_t *new_weights, vertex_t *new_neighbors, edge_t *new_degrees,
                              edge_t *neighbor_com_ptr_max, edge_t *edge_num_of_community, int *gathered_vertex_ptr, int *gathered_vertex_by_community, int *community,
                              int com_num_to_proc, int table_size, int warp_size, int community_num)
{ //
    int warp_id = threadIdx.x / warp_size;
    int lane_id = threadIdx.x % warp_size;
    extern __shared__ int new_neighbor_com_info[];
    int shared_half_size = table_size * (blockDim.x / warp_size);

    int *neighbor_com_ids = new_neighbor_com_info + warp_id * table_size; // all neighborcoms of one community
    int *neighbor_com_weights = new_neighbor_com_info + shared_half_size + warp_id * table_size;

    warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size;

    int warp_num = blockDim.x * gridDim.x / warp_size;
    for (int wp = warp_id; wp < com_num_to_proc; wp += warp_num)
    {

        int com = sorted_community_id[wp];
        edge_t start_neighbor_com = neighbor_com_ptr_max[com];

        int edge_num_in_community = edge_num_of_community[com];

        for (int i = lane_id; i < table_size; i += warp_size)
        { 
            neighbor_com_ids[i] = 0;
            neighbor_com_weights[i] = 0;
        }
        __syncthreads();
        if (edge_num_in_community < community_num)
        {

            compute_neighbors_weights(com, weights, neighbors, degrees,
                                      &new_weights[start_neighbor_com], &new_neighbors[start_neighbor_com], new_degrees,
                                      community, gathered_vertex_ptr, gathered_vertex_by_community,
                                      neighbor_com_ids, neighbor_com_weights,
                                      lane_id, table_size, warp_size);
        }
        else
        {

            compute_neighbors_weights_list(com, weights, neighbors, degrees,
                                           &new_weights[start_neighbor_com], &new_neighbors[start_neighbor_com], new_degrees,
                                           community, gathered_vertex_ptr, gathered_vertex_by_community,
                                           neighbor_com_ids, neighbor_com_weights,
                                           lane_id, community_num, warp_size);
        }

        __syncthreads(); //
    }
}

__global__ void build_by_block(int *sorted_community_id, weight_t *weights, vertex_t *neighbors, edge_t *degrees,
                               weight_t *new_weights, vertex_t *new_neighbors, edge_t *new_degrees,
                               edge_t *neighbor_com_ptr_max, edge_t *edge_num_of_community, int *gathered_vertex_ptr, int *gathered_vertex_by_community, int *community,
                               int *global_table_ptr, int *glbTable, int *primes, int prime_num,
                               int com_num_to_proc, int warp_size, int global_limit, int community_num)
{ // a community allocate a block
    __shared__ int neighbor_com_info[SHARE_MEM_SIZE * 2];
    int shared_half_size = SHARE_MEM_SIZE; // total share memory size
    int table_size;
    int block_id = blockIdx.x; // com_id
    int block_num = gridDim.x;

    int *neighbor_com_ids;
    int *neighbor_com_weights;

    for (int wp = block_id; wp < com_num_to_proc; wp += block_num)
    {

        int com = sorted_community_id[wp];
        edge_t start_neighbor_com = neighbor_com_ptr_max[com];

        int edge_num_in_community = edge_num_of_community[com];

        if (edge_num_in_community <= global_limit) // degree small
        {                                          // share memory
            neighbor_com_ids = neighbor_com_info;
            neighbor_com_weights = neighbor_com_info + shared_half_size;
            table_size = SHARE_MEM_SIZE;

            // may
        }
        else // degree large
        {    //
            // hash or list
            int start_glb_table = global_table_ptr[blockIdx.x];
            int global_half_size = global_table_ptr[com_num_to_proc]; // the edge num in community >4047

            neighbor_com_ids = &glbTable[start_glb_table];
            neighbor_com_weights = &glbTable[start_glb_table + global_half_size];
            table_size = global_table_ptr[blockIdx.x + 1] - global_table_ptr[blockIdx.x];

            if (edge_num_in_community >= community_num && community_num <= SHARE_MEM_SIZE)
            { // list
                neighbor_com_ids = neighbor_com_info;
                neighbor_com_weights = neighbor_com_info + shared_half_size;
                table_size = SHARE_MEM_SIZE;
            }
        }
        // __syncthreads();
        for (int i = threadIdx.x; i < table_size; i = i + blockDim.x)
        {
            neighbor_com_ids[i] = 0;
            neighbor_com_weights[i] = 0;
        }
        __syncthreads();
        if (edge_num_in_community < community_num)
        {

            compute_neighbors_weights_blk(com, weights, neighbors, degrees,
                                          &new_weights[start_neighbor_com], &new_neighbors[start_neighbor_com], new_degrees,
                                          community, gathered_vertex_ptr, gathered_vertex_by_community,
                                          neighbor_com_ids, neighbor_com_weights,
                                          threadIdx.x, table_size, warp_size);
        }
        else
        {

            compute_neighbors_weights_blk_list(com, weights, neighbors, degrees,
                                               &new_weights[start_neighbor_com], &new_neighbors[start_neighbor_com], new_degrees,
                                               community, gathered_vertex_ptr, gathered_vertex_by_community,
                                               neighbor_com_ids, neighbor_com_weights,
                                               threadIdx.x, community_num, warp_size);
        }
        __syncthreads(); 
    }
}

// build graph functions end


//pruning
__global__ void save_next_In(int *In, int *next_In, int* prev_community, int *cur_community,
                             int *active_set, int *is_moved, int *target_com_weights,
                             int *Tot, int *Self, int *K, double *prob, edge_t min_Tot,
                            double constant, int vertex_num, int iteration, int pruning_method)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < vertex_num)
    {

        int vertex = thread_id;
        int Ki = K[thread_id];
        int is_neighbors_moved = active_set[vertex];
        int is_self_moved = is_moved[vertex];
        int target_com_weight = 0;
        int tot_target_com = 0;
        if (is_neighbors_moved == 1 && is_self_moved == 0)
        { // prune further
            target_com_weight = target_com_weights[vertex] + In[vertex];
            tot_target_com = Tot[cur_community[vertex]];

            // if (2 * (target_com_weight - Self[vertex]) - (Ki - Self[vertex]) + (double)(min_Tot - (tot_target_com - Ki)) * constant * Ki > 0)
            // {
            //     active_set[vertex] = 0;
            // }
            In[vertex] = target_com_weight;
            is_moved[vertex] = 0;
        }
        else if (is_neighbors_moved == 0 && is_self_moved == 1)
        {
            target_com_weight = next_In[vertex];
            tot_target_com = Tot[cur_community[vertex]];
            // if (2 * (target_com_weight - Self[vertex]) - (Ki - Self[vertex]) + (double)(min_Tot - (tot_target_com - Ki)) * constant * Ki > 0)
            // {
            //     //
            // }
            // else
            // {
            //     active_set[vertex] = 1;
            // }
            In[vertex] = target_com_weight;
            is_moved[vertex] = 0;
        }
        else if (is_neighbors_moved == 0 && is_self_moved == 0)
        {
            target_com_weight = In[vertex];
            tot_target_com = Tot[cur_community[vertex]];
            // if (2 * (target_com_weight - Self[vertex]) - (Ki - Self[vertex]) + (double)(min_Tot - (tot_target_com - Ki)) * constant * Ki > 0)
            // {
            //     // In[vertex]=target_com_weight;
            // }
            // else
            // {
            //     active_set[vertex] = 1;
            // }
            is_moved[vertex] = 0;
        }
        else if (is_neighbors_moved == 1 && is_self_moved == 1)
        {
            is_moved[vertex] = 1;
        }

        if(pruning_method == 0){
            if (2 * (target_com_weight - Self[vertex]) - (Ki - Self[vertex]) + (double)(min_Tot - (tot_target_com - Ki)) * constant * Ki > 0)
            {
                active_set[vertex] = 0;
            }else{
                active_set[vertex] = 1;
            }

        }
        else if(pruning_method == 1){
            //do nothing
        }
        else if(pruning_method == 2){
            int is_active = 1;
            if (iteration >= 2 && (cur_community[vertex] == prev_community[vertex])) {
                prob[vertex] = prob[vertex] * (1.0 - VITE_ALPHA);
                if (prob[vertex] <= 0.02)
                    is_active = 0;
            }else{
                prob[vertex] = 1;
            }
            active_set[vertex] = is_active;
        }else if(pruning_method == 3){
            if(is_neighbors_moved == 1 && is_self_moved == 0 &&
                2 * (target_com_weight - Self[vertex]) - (Ki - Self[vertex]) + 
                (double)(min_Tot - (tot_target_com - Ki)) * constant * Ki > 0){
                active_set[vertex] = 0;
            }
        }
    }
    
}
