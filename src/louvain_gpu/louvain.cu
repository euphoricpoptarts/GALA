#include <fstream>
#include <iostream>
#include <string>
#include "kernel_functions.cuh"
#include "louvain.cuh"
using namespace std;

int *read_primes(string file_name, int *prime_num)
{
	ifstream file(file_name);
	int *primes;
	if (file.is_open())
	{
		file >> *prime_num;
		int p;
		// std::cout << "Reading " << *prime_num << " prime numbers." <<
		// std::endl; Read primes in host memory
		int index = 0;
		primes = new int[*prime_num];
		while (file >> p)
		{

			primes[index++] = p;
			if (index >= *prime_num)
				break;
			// std::cout << aPrimeNum << " ";
		}
	}
	else
	{
		cout << "Can't open file containing prime numbers." << endl;
	}
	return primes;
}

double louvain_gpu(Graph &g, vertex_t *h_comminity,
				   double min_modularity, int pruning_method)
{ // return modularity
	double prev_mod = -1, cur_mod = -1;
	thrust::device_vector<vertex_t> d_community(g.vertex_num);
	thrust::sequence(d_community.begin(), d_community.end());

	int round = 0;
	vertex_t community_num = g.vertex_num;

	// trans to gpu
	GraphGPU g_gpu(g);

	edge_t m2 = 0; // total weight
	m2 = thrust::reduce(g_gpu.d_weights.begin(), g_gpu.d_weights.end(),
						(edge_t)0);

	thrust::device_vector<vertex_t> community_round(g.vertex_num);
	int prime_num;
	int *h_primes = read_primes("primes", &prime_num);
	thrust::device_vector<int> primes(prime_num);
	thrust::copy(h_primes, h_primes + prime_num, primes.begin());
	delete []h_primes;

	// init end
	double start, end;
	start = get_time();

	printf("===============round:%d===============\n", round);
	double start1, end1;
	start1 = get_time();
	cur_mod = louvain_main_process(g_gpu.d_weights, g_gpu.d_neighbors,
								   g_gpu.d_degrees, community_round, primes,
								   community_num, round, min_modularity, m2, pruning_method);
	end1 = get_time();

	printf("louvain time in the first round = %fms\n", end1 - start1);

	start1 = get_time();
	community_num = build_compressed_graph(g_gpu.d_weights, g_gpu.d_neighbors,
										   g_gpu.d_degrees, community_round,
										   primes, community_num);
	
	// this needs to be after build_compressed_graph
	// because the communities are renumbered in that function
	thrust::gather(d_community.begin(), d_community.end(),
				   community_round.begin(), d_community.begin());

	end1 = get_time();
	printf("build time in the first round = %fms\n", end1 - start1);

	printf("number of communities:%d modularity:%f\n", community_num, cur_mod);
	// return 1;//!!!!!!!!!!!!!!!!!!
	while (cur_mod - prev_mod >= min_modularity)
	{

		prev_mod = cur_mod;
		printf("===============round:%d===============\n", round);
		cur_mod = louvain_main_process(
			g_gpu.d_weights, g_gpu.d_neighbors, g_gpu.d_degrees,
			community_round, primes, community_num, round, min_modularity, m2, pruning_method);

		community_num = build_compressed_graph(
			g_gpu.d_weights, g_gpu.d_neighbors, g_gpu.d_degrees,
			community_round, primes, community_num);
		// this needs to be after build_compressed_graph
		// because the communities are renumbered in that function
		thrust::gather(d_community.begin(), d_community.end(),
					community_round.begin(), d_community.begin());

		printf("number of communities:%d modularity:%f\n", community_num, cur_mod);
		// print_CSR(&weights,&neighbors,&degrees,&community_num);
	}
	printf("=====================================\n");
	end = get_time();
	printf("final number of communities:%d --> %d final modularity:%f\n", g.vertex_num, community_num, cur_mod);
	printf("execution time without data transfer = %fms\n", end - start);

	thrust::copy(d_community.begin(), d_community.end(), h_comminity);

	return cur_mod;
}