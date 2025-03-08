# GALA

[Swift Unfolding of Communities: GPU-Accelerated Louvain Algorithm](https://dl.acm.org/doi/10.1145/3710848.3710884)

## Dependencies

- GNU Make 4.3
- NVCC 11.6
- GCC/G++ 10.4

## Prepare graph

```shell
$ cd data
$ bash ./prepare_graph.sh
```

or

```shell
$ cd data
$ wget https://snap.stanford.edu/data/bigdata/communities/com-lj.ungraph.txt.gz
$ gunzip com-lj.ungraph.txt.gz
```

## Compile

```shell
$ cd ../src/
$ make -j
```

## Preprocess data

unweighted graph:

```shell
$ ./preprocess -f ../data/com-lj.ungraph.txt
```

weighted graph:

```shell
$ ./preprocess -f ../data/weighted-graph.txt -w
```

## Run

```shell
$ ./gala_main -f ../data/com-lj.ungraph.txt.bin -t 0.000001 -p 0
```
`-t` sets the termination threshold, and `-p` sets the pruning strategy **(0:MG 1:RM 2:Vite 3:MG+RM)** .

## Output

<!-- example:

```
$ ./gala_main -f ../data//karate.txt.bin
vertex number:34 edge number:78
load success
===============round:0===============
Iteration:0 Q:-0.049803
Iteration:1 Q:0.136834
Iteration:2 Q:0.191239
Iteration:3 Q:0.169214
time without data init = 16.410156ms
decideandmove time = 0.125000ms weight updating time = 10.340820ms remaining time = 5.944336ms
louvain time in the first round = 21.297119ms
build time in the first round = 3.638184ms
number of communities:11 modularity:0.191239
===============round:1===============
Iteration:0 Q:0.191239
Iteration:1 Q:0.313116
Iteration:2 Q:0.401134
Iteration:3 Q:0.408695
Iteration:4 Q:0.408695
time without data init = 1.347900ms
decideandmove time = 0.119141ms weight updating time = 0.564209ms remaining time = 0.664551ms
number of communities:4 modularity:0.408695
===============round:2===============
Iteration:0 Q:0.408695
Iteration:1 Q:0.408695
time without data init = 1.472168ms
decideandmove time = 0.020996ms weight updating time = 0.114990ms remaining time = 1.336182ms
number of communities:4 modularity:0.408695
=====================================
final number of communities:34 -> 4 final modularity:0.408695
execution time without data transfer = 50.241943ms
elapsed time = 313.437988ms
``` -->

"iteration" refers to the iterations of the first phase of the Louvain algorithm, "Q" refers to the temporary modularity;

"round" refers to the rounds of both the first and second phases of the algorithm, "number of communities" refers to the current number of community partitions;

"final number of communities" and "final modularity" indicate the resulting partition and modularity at the end, and "elapsed time" refers to the program's running time.
