#include "kernel_functions.cuh"
#include "louvain.cuh"
using namespace std;

void efficient_weight_updating(thrust::device_vector<weight_t> &d_weights,
							   thrust::device_vector<vertex_t> &d_neighbors,
							   thrust::device_vector<edge_t> &d_degrees,
							   thrust::device_vector<int> &cur_community,
							   thrust::device_vector<int> &sorted_vertex_id,
							   thrust::device_vector<int> &active_set,
							   thrust::device_vector<int> &In,
							   thrust::device_vector<int> &Tot,
							   thrust::device_vector<int> &K,
							   thrust::device_vector<int> &Self, int vertex_num,
							   int *deg_num_tbl, int min_Tot, double constant, int pruning_method)
{
	int deg_num_4 = deg_num_tbl[0];
	int deg_num_8 = deg_num_tbl[1];
	int deg_num_16 = deg_num_tbl[2];
	int deg_num_32 = deg_num_tbl[3];
	int deg_num_128 = deg_num_tbl[4];
	int deg_num_1024 = deg_num_tbl[5];
	int deg_num_limit = deg_num_tbl[6];
	int deg_num_greater_than_limit = deg_num_tbl[7];
	int thread_num_to_alloc, block_num;
	if (deg_num_4)
	{
		thread_num_to_alloc = 4;
		block_num = (deg_num_4 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
					THREAD_NUM_PER_BLOCK;

		compute_In<<<block_num, THREAD_NUM_PER_BLOCK>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()),
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_4, thread_num_to_alloc,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()), pruning_method);
		cudaDeviceSynchronize();
	}

	if (deg_num_8)
	{
		thread_num_to_alloc = 8;

		block_num = (deg_num_8 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
					THREAD_NUM_PER_BLOCK;

		compute_In<<<block_num, THREAD_NUM_PER_BLOCK>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_8, thread_num_to_alloc,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()), pruning_method);
		cudaDeviceSynchronize();
	}
	if (deg_num_16)
	{
		thread_num_to_alloc = 16;
		block_num = (deg_num_16 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
					THREAD_NUM_PER_BLOCK;

		compute_In<<<block_num, THREAD_NUM_PER_BLOCK>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 + deg_num_8,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_16, thread_num_to_alloc,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()), pruning_method);
		cudaDeviceSynchronize();
	}

	if (deg_num_32)
	{
		thread_num_to_alloc = 32;
		block_num = (deg_num_32 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
					THREAD_NUM_PER_BLOCK;

		compute_In<<<block_num, THREAD_NUM_PER_BLOCK>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
				deg_num_8 + deg_num_16,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_32, thread_num_to_alloc,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()), pruning_method);
		cudaDeviceSynchronize();
	}

	if (deg_num_128)
	{ // 33-127
		thread_num_to_alloc = 32;
		block_num = (deg_num_128 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
					THREAD_NUM_PER_BLOCK;
		compute_In<<<block_num, THREAD_NUM_PER_BLOCK>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
				deg_num_8 + deg_num_16 + deg_num_32,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_128, thread_num_to_alloc,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()), pruning_method);
		cudaDeviceSynchronize();
	}

	if (deg_num_1024)
	{ // 128-1024
		int warp_size = 32;
		block_num = (deg_num_1024 * 256 + 256 - 1) / 256;
		compute_In_blk<<<block_num, 256>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
				deg_num_8 + deg_num_16 + deg_num_32 + deg_num_128,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_1024, warp_size,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()), pruning_method);
		cudaDeviceSynchronize();
	}
	if (deg_num_limit)
	{ // 1025-4047
		int warp_size = 32;
		block_num = (deg_num_limit * 1024 + 1024 - 1) / 1024;
		compute_In_blk<<<block_num, 1024>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
				deg_num_8 + deg_num_16 + deg_num_32 + deg_num_128 + deg_num_1024,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_limit, warp_size,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()), pruning_method);
		cudaDeviceSynchronize();
	}

	if (deg_num_greater_than_limit)
	{ // 4048-
		int warp_size = 32;
		thread_num_to_alloc = 1024;
		block_num = (deg_num_greater_than_limit * thread_num_to_alloc + 1024 - 1) / 1024;
		compute_In_blk<<<block_num, 1024>>>(
			thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
				deg_num_8 + deg_num_16 + deg_num_32 + deg_num_128 + deg_num_1024 +
				deg_num_limit,
			thrust::raw_pointer_cast(d_weights.data()),
			thrust::raw_pointer_cast(d_neighbors.data()),
			thrust::raw_pointer_cast(d_degrees.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(Self.data()), deg_num_greater_than_limit, warp_size,
			constant, min_Tot, thrust::raw_pointer_cast(active_set.data()), pruning_method);
		cudaDeviceSynchronize();
	}
}

void filter_vertex_into_new_vector(thrust::device_vector<int> &sorted_vertex_id_const, thrust::device_vector<int> &sorted_vertex_id, vertex_filter filter_, const int *deg_num_tbl_const, int *h_deg_num_tbl, int degree_type_size)
{
	thrust::device_vector<int>::iterator const_start = sorted_vertex_id_const.begin();
	thrust::device_vector<int>::iterator const_end = const_start;
	thrust::device_vector<int>::iterator filter_start = sorted_vertex_id.begin();
	thrust::device_vector<int>::iterator filter_end;
	for (int i = 0; i < degree_type_size; i++)
	{
		const_end = const_start + deg_num_tbl_const[i];
		filter_end = thrust::copy_if(const_start, const_end,
									 filter_start, filter_);
		h_deg_num_tbl[i] = thrust::distance(filter_start, filter_end);
		const_start = const_end;
		filter_start = filter_end;
	}
}

double louvain_main_process(thrust::device_vector<weight_t> &d_weights,
							thrust::device_vector<vertex_t> &d_neighbors,
							thrust::device_vector<edge_t> &degrees,
							thrust::device_vector<vertex_t> &coummunity_result,
							thrust::device_vector<int> &primes,
							vertex_t vertex_num, int &round,
							double min_modularity, edge_t m2, int pruning_method)
{
	// init gpu vector
	int iteration = 0;

	thrust::device_vector<edge_t> d_degrees(vertex_num + 1, 0);
	thrust::copy(thrust::device, degrees.begin(), degrees.end(),
				 d_degrees.begin() + 1); // d_degrees[0]=0

	thrust::device_vector<int> degree_of_vertex(vertex_num, 0);
	thrust::transform(d_degrees.begin() + 1, d_degrees.end(), d_degrees.begin(),
					  degree_of_vertex.begin(),
					  thrust::minus<edge_t>()); // d[i]-d[i-1]

	double constant = 1 / (double)m2;
	// init filter

	int global_limit = SHARE_MEM_SIZE * 2 / 3 - 1; // vertex degree under limit can use share mem

	degree_filter_in_range filter_for_4(1, 3);
	degree_filter_in_range filter_for_8(4, 7);
	degree_filter_in_range filter_for_16(8, 15);
	degree_filter_in_range filter_for_32(16, 31);
	degree_filter_in_range filter_for_128(32, 128);
	degree_filter_in_range filter_for_1024(129, 1024);
	degree_filter_in_range filter_for_limit(1025, global_limit);
	degree_filter_in_range filter_for_global(global_limit + 1, vertex_num);

	int h_degree_type_low[] = {1, 4, 8, 16, 32, 129, 1025, global_limit + 1};
	int h_degree_type_high[] = {3, 7, 15, 31, 128, 1024, global_limit, vertex_num};
	int degree_type_size = 8;
	thrust::device_vector<int> degree_type_low(degree_type_size);
	thrust::device_vector<int> degree_type_high(degree_type_size);
	thrust::copy(h_degree_type_low, h_degree_type_low + degree_type_size,
				 degree_type_low.begin());
	thrust::copy(h_degree_type_high, h_degree_type_high + degree_type_size,
				 degree_type_high.begin());

	// count degree
	int deg_num_4 = thrust::count_if(thrust::device, degree_of_vertex.begin(),
									 degree_of_vertex.end(), filter_for_4);
	int deg_num_8 = thrust::count_if(thrust::device, degree_of_vertex.begin(),
									 degree_of_vertex.end(), filter_for_8);
	int deg_num_16 = thrust::count_if(thrust::device, degree_of_vertex.begin(),
									  degree_of_vertex.end(), filter_for_16);
	int deg_num_32 = thrust::count_if(thrust::device, degree_of_vertex.begin(),
									  degree_of_vertex.end(), filter_for_32);
	int deg_num_128 = thrust::count_if(thrust::device, degree_of_vertex.begin(),
									   degree_of_vertex.end(), filter_for_128);
	int deg_num_1024 =
		thrust::count_if(thrust::device, degree_of_vertex.begin(),
						 degree_of_vertex.end(), filter_for_1024);
	int deg_num_limit =
		thrust::count_if(thrust::device, degree_of_vertex.begin(),
						 degree_of_vertex.end(), filter_for_limit);
	int deg_num_greater_than_limit =
		thrust::count_if(thrust::device, degree_of_vertex.begin(),
						 degree_of_vertex.end(), filter_for_global);

	int deg_num_tbl_const[] = {
		deg_num_4, deg_num_8, deg_num_16, deg_num_32,
		deg_num_128, deg_num_1024, deg_num_limit, deg_num_greater_than_limit};

	// init Tot, In...
	thrust::device_vector<int> Tot(vertex_num, 0); // community-tot 
	thrust::device_vector<int> In(vertex_num, 0);  // community-in 
	thrust::device_vector<int> next_In(vertex_num, 0);
	thrust::device_vector<int> K(vertex_num, 0); // degree of vertex 
	// merge-vertex i thrust::sequence(community.begin(),community.end());
	thrust::device_vector<int> Self(vertex_num, 0);
	// community labels
	thrust::device_vector<int> prev_community(vertex_num);
	thrust::device_vector<int> cur_community(vertex_num);

	thrust::device_vector<int> Tot_update(vertex_num,
										  0); // record update infomation
	thrust::device_vector<int> community_size(vertex_num, 1);
	thrust::device_vector<int> community_size_update(vertex_num, 0);
	int community_num = vertex_num; // number of community

	// fill Tot,K,Self
	int warp_size = WARP_SIZE;
	int block_num = (community_num + THREAD_NUM_PER_BLOCK - 1) /
					THREAD_NUM_PER_BLOCK; 
	init_communities<<<block_num, THREAD_NUM_PER_BLOCK>>>(
		thrust::raw_pointer_cast(d_weights.data()),
		thrust::raw_pointer_cast(d_neighbors.data()),
		thrust::raw_pointer_cast(d_degrees.data()),
		thrust::raw_pointer_cast(K.data()),
		thrust::raw_pointer_cast(Tot.data()),
		thrust::raw_pointer_cast(Self.data()),
		thrust::raw_pointer_cast(prev_community.data()), vertex_num, warp_size);
	cur_community = prev_community;

	// init sorted list for bucket
	thrust::device_vector<int> sorted_vertex_id(vertex_num);
	thrust::device_vector<int> original_vertex_id(vertex_num);
	thrust::sequence(original_vertex_id.begin(), original_vertex_id.end());
	thrust::copy_if(original_vertex_id.begin(), original_vertex_id.end(),
					degree_of_vertex.begin(), sorted_vertex_id.begin(),
					filter_for_4);
	thrust::copy_if(original_vertex_id.begin(), original_vertex_id.end(),
					degree_of_vertex.begin(), sorted_vertex_id.begin() + deg_num_4,
					filter_for_8);
	thrust::copy_if(original_vertex_id.begin(), original_vertex_id.end(),
					degree_of_vertex.begin(),
					sorted_vertex_id.begin() + deg_num_4 + deg_num_8, filter_for_16);
	thrust::copy_if(original_vertex_id.begin(), original_vertex_id.end(),
					degree_of_vertex.begin(),
					sorted_vertex_id.begin() + deg_num_4 + deg_num_8 + deg_num_16,
					filter_for_32);
	thrust::copy_if(original_vertex_id.begin(), original_vertex_id.end(),
					degree_of_vertex.begin(),
					sorted_vertex_id.begin() + deg_num_4 + deg_num_8 + deg_num_16 +
						deg_num_32,
					filter_for_128);
	thrust::copy_if(original_vertex_id.begin(), original_vertex_id.end(),
					degree_of_vertex.begin(),
					sorted_vertex_id.begin() + deg_num_4 + deg_num_8 + deg_num_16 +
						deg_num_32 + deg_num_128,
					filter_for_1024);
	thrust::copy_if(original_vertex_id.begin(), original_vertex_id.end(),
					degree_of_vertex.begin(),
					sorted_vertex_id.begin() + deg_num_4 + deg_num_8 + deg_num_16 +
						deg_num_32 + deg_num_128 + deg_num_1024,
					filter_for_limit);
	thrust::copy_if(original_vertex_id.begin(), original_vertex_id.end(),
					degree_of_vertex.begin(),
					sorted_vertex_id.begin() + deg_num_4 + deg_num_8 + deg_num_16 +
						deg_num_32 + deg_num_128 + deg_num_1024 +
						deg_num_limit,
					filter_for_global);

	// original_vertex_id.resize(sorted_vertex_id.size(), 0);
	thrust::gather(
		thrust::device, sorted_vertex_id.begin(), sorted_vertex_id.end(),
		degree_of_vertex.begin(),
		original_vertex_id
			.begin()); // original_vertex_id[i]=degree_of_vertex[sorted[i]]
	// sort large vertex by degree
	thrust::sort_by_key(
		original_vertex_id.begin() + deg_num_4 + deg_num_8 + deg_num_16 + deg_num_32 +
			deg_num_128 + deg_num_1024 + deg_num_limit,
		original_vertex_id.end(),
		sorted_vertex_id.begin() + deg_num_4 + deg_num_8 + deg_num_16 + deg_num_32 +
			deg_num_128 + deg_num_1024 + deg_num_limit,
		thrust::greater<unsigned int>());

	// prepare global table start location for large vertices
	thrust::device_vector<int> global_table_offset(deg_num_greater_than_limit + 1, 0);

	if (deg_num_greater_than_limit > 0)
	{
		block_num = (deg_num_greater_than_limit * WARP_SIZE + 1024 - 1) / 1024;
		compute_global_table_size_louvain<<<block_num, 1024>>>(
			thrust::raw_pointer_cast(primes.data()), primes.size(),
			thrust::raw_pointer_cast(original_vertex_id.data()) + deg_num_4 +
				deg_num_8 + deg_num_16 + deg_num_32 + deg_num_128 + deg_num_1024 +
				deg_num_limit,
			deg_num_greater_than_limit, thrust::raw_pointer_cast(global_table_offset.data()) + 1,
			warp_size);
	}

	thrust::inclusive_scan(global_table_offset.begin(), global_table_offset.end(),
						   global_table_offset.begin(), thrust::plus<int>());
	thrust::device_vector<int> global_table(2 * global_table_offset.back(), 0);


	// hash table return 1;

	thrust::device_vector<int> active_set(vertex_num, 1); // active set
	thrust::device_vector<int> is_moved(vertex_num, 0);	  // whether vertex has moved into new comm in the iteration

	thrust::device_vector<int> sorted_vertex_id_const(vertex_num); // save a sorted_vertex_id copy
	thrust::device_vector<double> vite_prob(vertex_num, 1);
	thrust::copy(thrust::device, sorted_vertex_id.begin(), sorted_vertex_id.end(),
				 sorted_vertex_id_const.begin());

	double prev_modularity = -1;
	double cur_modularity = -1;
	int thread_num_to_alloc;

	thrust::device_vector<int> target_com_weights(vertex_num, 0);

	double start, end;
	start = get_time();

	efficient_weight_updating(
		d_weights, d_neighbors, d_degrees, cur_community, sorted_vertex_id_const,
		active_set, In, Tot, K, Self, vertex_num, deg_num_tbl_const, 1, constant, pruning_method);
	thrust::device_vector<double> sum(vertex_num, 0);
	thrust::transform(thrust::device, In.begin(), In.end(), Tot.begin(),
					  sum.begin(), modularity_op(constant));
	cur_modularity = thrust::reduce(thrust::device, sum.begin(), sum.end(),
									(double)0.0, thrust::plus<double>());

	printf("Iteration:%d Q:%f\n", iteration, cur_modularity);

	iteration++;
	prev_modularity = cur_modularity;

	// save vertices whose neighbors do not moving in the iteration

	double decideandmovetime = 0;
	double updatetime = 0;
	double remainingtime = 0;

	while (true)
	{
		thrust::fill(active_set.begin(), active_set.end(), 0);
		thrust::fill(is_moved.begin(), is_moved.end(), 0);
		thrust::fill(Tot_update.begin(), Tot_update.end(), 0);
		// thrust::fill(In.begin(),In.end(),0);
		thrust::fill(community_size_update.begin(), community_size_update.end(), 0);
		thrust::fill(target_com_weights.begin(), target_com_weights.end(), 0);

		double start1, end1;

		start1 = get_time();

		if (deg_num_4)
		{
			thread_num_to_alloc = 4;

			block_num = (deg_num_4 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
						THREAD_NUM_PER_BLOCK;

			decide_and_move_shuffle<<<block_num, THREAD_NUM_PER_BLOCK>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()),
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()), deg_num_4,
				thread_num_to_alloc, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}

		if (deg_num_8)
		{
			thread_num_to_alloc = 8;

			block_num = (deg_num_8 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
						THREAD_NUM_PER_BLOCK;

			decide_and_move_shuffle<<<block_num, THREAD_NUM_PER_BLOCK>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()), deg_num_8,
				thread_num_to_alloc, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);

			cudaDeviceSynchronize();
		}

		if (deg_num_16)
		{
			thread_num_to_alloc = 16;

			block_num = (deg_num_16 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
						THREAD_NUM_PER_BLOCK;

			decide_and_move_shuffle<<<block_num, THREAD_NUM_PER_BLOCK>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
					deg_num_8,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()), deg_num_16,
				thread_num_to_alloc, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}

		if (deg_num_32)
		{
			thread_num_to_alloc = 32;

			block_num = (deg_num_32 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
						THREAD_NUM_PER_BLOCK;

			decide_and_move_shuffle<<<block_num, THREAD_NUM_PER_BLOCK>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
					deg_num_8 + deg_num_16,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()), deg_num_32,
				thread_num_to_alloc, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}

		if (deg_num_128)
		{ // 33-127
			thread_num_to_alloc = 32;
			int table_size = 257;
			block_num = (deg_num_128 * thread_num_to_alloc + THREAD_NUM_PER_BLOCK - 1) /
						THREAD_NUM_PER_BLOCK;
			int shared_size =
				table_size * (THREAD_NUM_PER_BLOCK / thread_num_to_alloc);
			decide_and_move_hash_shared<<<block_num, THREAD_NUM_PER_BLOCK,
										  3 * shared_size * sizeof(int)>>>(
				// find_best_com_blk_no_share<<<deg_num_128, 128,2 * table_size *
				// sizeof(int)>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
					deg_num_8 + deg_num_16 + deg_num_32,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()), deg_num_128,
				table_size, thread_num_to_alloc, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}
		if (deg_num_1024)
		{ // 129-1024
			warp_size = 32;
			block_num = (deg_num_1024 * 256 + 256 - 1) / 256;
			decide_and_move_hash_hierarchical<<<block_num, 256>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
					deg_num_8 + deg_num_16 + deg_num_32 + deg_num_128,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()),
				thrust::raw_pointer_cast(global_table_offset.data()),
				thrust::raw_pointer_cast(global_table.data()),
				thrust::raw_pointer_cast(primes.data()), primes.size(),
				deg_num_1024, warp_size, global_limit, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}
		if (deg_num_limit)
		{ // 1025-2703
			warp_size = 32;
			block_num = (deg_num_limit * MAX_THREAD_PER_BLOCK +
						 MAX_THREAD_PER_BLOCK - 1) /
						MAX_THREAD_PER_BLOCK;
			decide_and_move_hash_hierarchical<<<block_num, MAX_THREAD_PER_BLOCK>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
					deg_num_8 + deg_num_16 + deg_num_32 + deg_num_128 +
					deg_num_1024,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()),
				thrust::raw_pointer_cast(global_table_offset.data()),
				thrust::raw_pointer_cast(global_table.data()),
				thrust::raw_pointer_cast(primes.data()), primes.size(),
				deg_num_limit, warp_size, global_limit, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}

		if (deg_num_greater_than_limit)
		{ // 2704-
			warp_size = 32;
			block_num = (deg_num_greater_than_limit * MAX_THREAD_PER_BLOCK +
						 MAX_THREAD_PER_BLOCK - 1) /
						MAX_THREAD_PER_BLOCK;
			// read primes
			decide_and_move_hash_hierarchical<<<block_num, MAX_THREAD_PER_BLOCK>>>(
				thrust::raw_pointer_cast(sorted_vertex_id.data()) + deg_num_4 +
					deg_num_8 + deg_num_16 + deg_num_32 + deg_num_128 +
					deg_num_1024 + deg_num_limit,
				thrust::raw_pointer_cast(d_weights.data()),
				thrust::raw_pointer_cast(d_neighbors.data()),
				thrust::raw_pointer_cast(d_degrees.data()),
				thrust::raw_pointer_cast(prev_community.data()),
				thrust::raw_pointer_cast(cur_community.data()),
				thrust::raw_pointer_cast(K.data()),
				thrust::raw_pointer_cast(Tot.data()),
				thrust::raw_pointer_cast(In.data()),
				thrust::raw_pointer_cast(next_In.data()),
				thrust::raw_pointer_cast(Self.data()),
				thrust::raw_pointer_cast(community_size.data()),
				thrust::raw_pointer_cast(Tot_update.data()),
				thrust::raw_pointer_cast(community_size_update.data()),
				thrust::raw_pointer_cast(global_table_offset.data()),
				thrust::raw_pointer_cast(global_table.data()),
				thrust::raw_pointer_cast(primes.data()), primes.size(),
				deg_num_greater_than_limit, warp_size, global_limit, constant,
				thrust::raw_pointer_cast(active_set.data()),
				thrust::raw_pointer_cast(is_moved.data()),
				thrust::raw_pointer_cast(target_com_weights.data()), iteration);
			cudaDeviceSynchronize();
		}

		end1 = get_time();
		decideandmovetime += end1 - start1;

		double start2, end2;
		start2 = get_time();
		// compute In
		thrust::transform(thrust::device, Tot.begin(), Tot.end(),
						  Tot_update.begin(), Tot.begin(), thrust::plus<int>());
		thrust::transform(thrust::device, community_size.begin(), community_size.end(),
						  community_size_update.begin(), community_size.begin(),
						  thrust::plus<int>());

		edge_t min_Tot = thrust::transform_reduce(Tot.begin(), Tot.end(), Tot_op(m2), m2,
												  thrust::minimum<edge_t>());

		block_num = (vertex_num + 1024 - 1) / 1024;
		save_next_In<<<block_num, 1024>>>(
			thrust::raw_pointer_cast(In.data()),
			thrust::raw_pointer_cast(next_In.data()),
			thrust::raw_pointer_cast(prev_community.data()),
			thrust::raw_pointer_cast(cur_community.data()),
			thrust::raw_pointer_cast(active_set.data()),
			thrust::raw_pointer_cast(is_moved.data()),
			thrust::raw_pointer_cast(target_com_weights.data()),
			thrust::raw_pointer_cast(Tot.data()),
			thrust::raw_pointer_cast(Self.data()),
			thrust::raw_pointer_cast(K.data()),
			thrust::raw_pointer_cast(vite_prob.data()), 
			(int)min_Tot, constant,
			vertex_num, iteration, pruning_method);
		cudaDeviceSynchronize();

		int h_deg_num_tbl[degree_type_size];
		vertex_filter moved_filter = vertex_filter(is_moved.data(), 1); // vertices to compute In
		thrust::device_vector<int> compute_In_sorted_vertex_id(vertex_num);
		filter_vertex_into_new_vector(sorted_vertex_id_const, compute_In_sorted_vertex_id, moved_filter, deg_num_tbl_const, h_deg_num_tbl, degree_type_size);

		efficient_weight_updating(d_weights, d_neighbors, d_degrees,
								  cur_community, compute_In_sorted_vertex_id,
								  active_set, In, Tot, K, Self, vertex_num,
								  h_deg_num_tbl, (int)min_Tot, constant, pruning_method);
		end2 = get_time();
		updatetime += end2 - start2;

		thrust::device_vector<double> sum(vertex_num, 0);
		thrust::transform(thrust::device, In.begin(), In.end(), Tot.begin(),
						  sum.begin(), modularity_op(constant));
		cur_modularity = thrust::reduce(thrust::device, sum.begin(), sum.end(),
										(double)0.0, thrust::plus<double>());

		printf("Iteration:%d Q:%f\n", iteration, cur_modularity);
		if ((cur_modularity - prev_modularity) < min_modularity)
		{ // threshold
			break;
		}

		vertex_filter active_filter = vertex_filter(active_set.data(), 1);
		filter_vertex_into_new_vector(sorted_vertex_id_const, sorted_vertex_id, active_filter, deg_num_tbl_const, h_deg_num_tbl, degree_type_size);
		deg_num_4 = h_deg_num_tbl[0];
		deg_num_8 = h_deg_num_tbl[1];
		deg_num_16 = h_deg_num_tbl[2];
		deg_num_32 = h_deg_num_tbl[3];
		deg_num_128 = h_deg_num_tbl[4];
		deg_num_1024 = h_deg_num_tbl[5];
		deg_num_limit = h_deg_num_tbl[6];
		deg_num_greater_than_limit = h_deg_num_tbl[7];

		prev_modularity = cur_modularity;
		prev_community = cur_community; // Current holds the chosen assignment
		iteration++;

	}

	round++;

	end = get_time();

	printf("time without data init = %fms\n", end - start);

	remainingtime = end - start - decideandmovetime - updatetime;
	printf("decideandmove time = %fms weight updating time = %fms remaining time = %fms\n", decideandmovetime, updatetime, remainingtime);

	coummunity_result.resize(vertex_num);
	thrust::copy(prev_community.begin(), prev_community.end(), coummunity_result.begin());

	return prev_modularity;
}
