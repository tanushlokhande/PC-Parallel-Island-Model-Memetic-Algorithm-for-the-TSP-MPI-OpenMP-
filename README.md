# PC-Parallel-Island-Model-Memetic-Algorithm-for-the-TSP-MPI-OpenMP-
# Parallel Island-Model Memetic Algorithm for the Travelling Salesman Problem
### Hybrid MPI + OpenMP implementation

**Student:** Tanush Rajaram Lokhande  |  **Roll No:** 241080035
**Course:** Parallel Computing Mini Project

---

## 1. Overview

The Travelling Salesman Problem (TSP) asks for the shortest closed tour that visits every city exactly once.
It is NP-hard, so large instances are solved with heuristics. This project solves TSP with a **memetic genetic
algorithm** (a GA whose offspring are improved by local search) and parallelises it on two levels:

| Level | Technology | What it does |
|---|---|---|
| Between islands | **MPI** | Several populations ("islands") evolve independently and exchange their best tours in a ring |
| Inside an island | **OpenMP** | Threads build and improve offspring in parallel |

Three versions are built from one source file and compared: **serial**, **OpenMP-only** and **hybrid MPI + OpenMP**.

## 2. Project structure

```
tsp_hybrid_ga/
├── README.md                    this file
├── TSP_Hybrid_GA_Colab.ipynb    run everything online on Google Colab
├── Makefile                     builds the three executables
├── src/tsp_ga.cpp               the solver (serial / OpenMP / hybrid from one file)
├── scripts/
│   ├── gen_instance.py          create a random TSPLIB-format instance
│   ├── run_benchmarks.py        experiment driver (fetch, scaling, migration, quality)
│   └── plot_results.py          makes graphs from results/*.csv
├── data/                        TSP instances (two random samples included)
└── results/                     output CSVs and PNG graphs
```

## 3. Quick start

### Option A: Google Colab (no installation)
1. Open https://colab.research.google.com and choose **File → Upload notebook**.
2. Upload `TSP_Hybrid_GA_Colab.ipynb`.
3. Choose **Runtime → Run all**.

The notebook installs OpenMPI, compiles all three versions, downloads TSPLIB instances, runs the experiments
and draws the graphs. Free Colab has about 2 CPU cores, so speedup there is limited to roughly 2×.

### Option B: Linux or WSL
```bash
sudo apt install build-essential libopenmpi-dev openmpi-bin python3-pip
pip install pandas matplotlib
make                                         # creates bin/tsp_serial, bin/tsp_omp, bin/tsp_hybrid
python3 scripts/run_benchmarks.py fetch      # download TSPLIB instances into data/
```

Run examples:
```bash
bin/tsp_serial --tsp data/pr1002.tsp --pop 64 --gens 100 --opt 259045
bin/tsp_omp    --tsp data/pr1002.tsp --pop 64 --gens 100 --threads 4
mpirun -np 4 bin/tsp_hybrid --tsp data/pr1002.tsp --pop 64 --strong --gens 100 --threads 4 --out best.tour
```

## 4. Algorithm

```
for each island (MPI rank), in parallel:
    initial population = nearest-neighbour tours (lightly kicked) + local search      [OpenMP]
    repeat for G generations:
        offspring = tournament select 2 parents -> Order Crossover
                    -> mutate with probability p_m (double-bridge)
                    -> 2-opt + Or-opt local search                                    [OpenMP parallel for]
        population = best P distinct tours of (population + offspring)
        if no improvement for S generations: replace worst half with kicked copies of the best half
        every I generations: Isend top-M tours to rank+1, Test/Irecv from rank-1,
                             insert immigrants over the worst individuals             [asynchronous]
final: MPI_Allreduce(MINLOC) finds the best island; its tour is broadcast and validated
```

### Features
- **Memetic search:** 2-opt and Or-opt with K-nearest-neighbour candidate lists and a work queue
  (don't-look-bits idea), so only cities near a change are re-examined.
- **Asynchronous ring migration:** `MPI_Isend / MPI_Irecv / MPI_Test`; islands never wait for each other.
- **Heterogeneous islands:** mutation probability rises from 0.05 to 0.50 across ranks (explore vs exploit).
- **Diversity control:** duplicate tours are removed by hashing their edge sets; stagnation triggers kicks.
- **MPI-3 shared memory:** one copy of the distance matrix per node, shared by all ranks on it.
- **Dynamic OpenMP scheduling:** local-search time varies per offspring, so threads are load-balanced.
- **Reproducibility:** each offspring has its own random stream, so a seed gives the same result for any thread count.
- **Validation:** the final tour is checked to be a valid permutation and its length is recomputed.

## 5. Command-line options

| Option | Meaning | Default |
|---|---|---|
| `--tsp FILE` / `--random N` | TSPLIB file (EUC_2D, CEIL_2D, ATT) or random instance of N cities | required |
| `--pop P` | Population per island | 48 |
| `--strong` | Treat `--pop` as the **total** population and split it across islands | off |
| `--gens G` | Number of generations | 200 |
| `--mig-int I` | Migrate every I generations (`0` = isolated islands) | 10 |
| `--mig-count M` | Tours sent per migration | 2 |
| `--threads T` | OpenMP threads per process | all |
| `--ls 0\|1` | Local search on offspring (`0` gives the textbook GA) | 1 |
| `--pm x` | Mutation probability (otherwise automatic per island) | auto |
| `--stag S` | Stagnation generations before kicks (`0` = off) | 40 |
| `--k K` | Neighbour-list size | 10 |
| `--seed s` | Random seed | 1 |
| `--opt L` | Best known length, used to print the gap | none |
| `--csv F`, `--trace F`, `--out F` | Append result row, write convergence trace, write best tour | none |

## 6. Experiments

```bash
# Strong scaling: speedup and efficiency
python3 scripts/run_benchmarks.py scaling   --inst pr1002 --threads 1 2 4 8 --ranks 2 4 --reps 5

# Migration interval and size (0 = isolated islands as control)
python3 scripts/run_benchmarks.py migration --inst pr1002 --nranks 8 --mig-int 2 5 10 25 --mig-count 1 2 4

# Plain GA vs memetic GA vs hybrid islands, gap to best known
python3 scripts/run_benchmarks.py quality   --insts eil51 kroA100 ch150 a280 pcb442 pr1002

python3 scripts/plot_results.py              # writes PNG graphs into results/
```

**Metrics**

| Metric | Formula |
|---|---|
| Speedup | S = T_serial / T_p |
| Efficiency | E = S / (ranks × threads) |
| Solution quality (gap) | 100 × (L − L_best_known) / L_best_known |

- **Strong scaling:** total population fixed (`--strong`); the same work is divided among islands.
- **Weak scaling:** omit `--strong`; each island keeps its population, so total work grows with the number of islands.
- On a cluster, use one rank per socket and one thread per core, for example
  `mpirun --map-by ppr:1:socket --bind-to socket -np 2 bin/tsp_hybrid --threads 8 ...`

## 7. Output

Every run prints, for example:
```
instance=pr1002.tsp n=1002 mode=hybrid ranks=4 threads=2 pop/island=16 gens=100 ls=1 mig=10/2
best_length=262310 valid=yes gap=1.264%
setup_time=0.120s evolve_time=1.870s evaluations=6464
```

## 8. Verified status

Tested in a single-core Linux sandbox with GCC 13 and OpenMPI 4.1:
- All three versions compile and return valid permutations.
- The hybrid version runs correctly with 4 oversubscribed ranks, including migration and clean shutdown.
- On a 200-city random instance (same population and generations), the plain GA gave tour length 37,286 and the
  memetic version gave 10,982, a 3.4× shorter tour.
- The Colab notebook runs end to end. Its TSPLIB download path could not be tested offline, so it uses the
  random-instance fallback there; on Colab with internet it should download the real instances.

**Not measured yet:** real multi-core speedup. Run section 6 on a multi-core machine or cluster and report your own numbers.

## 9. Limitations and future work

- Order Crossover is simple. **Edge Assembly Crossover (EAX)** or **partition crossover** would give much better tours on large instances.
- Local search is 2-opt + Or-opt; **Or-3opt or Lin-Kernighan** would be stronger.
- Runs stop after a fixed number of generations (needed to match every send with a receive), not at a time limit.
- The full distance matrix is stored only for n ≤ 4500; larger instances compute distances on the fly.
- A CUDA port of the local search would give a CPU + GPU hybrid.

## 10. References

1. E. Cantú-Paz, "A Survey of Parallel Genetic Algorithms," *Calculateurs Parallèles*, 10(2), 1998.
2. E. Alba, M. Tomassini, "Parallelism and Evolutionary Algorithms," *IEEE Trans. Evol. Comput.*, 6(5), 2002.
3. G. Reinelt, "TSPLIB - A Traveling Salesman Problem Library," *ORSA J. Computing*, 3(4), 1991.
4. Y. Nagata, S. Kobayashi, "A powerful genetic algorithm using edge assembly crossover for the TSP," *INFORMS J. Computing*, 25(2), 2013.
5. P. Merz, B. Freisleben, "Memetic algorithms for the traveling salesman problem," *Complex Systems*, 13(4), 2001.
6. K. Helsgaun, "An effective implementation of the Lin-Kernighan traveling salesman heuristic," *Eur. J. Oper. Res.*, 126(1), 2000.
7. J. L. Bentley, "Fast algorithms for geometric traveling salesman problems," *ORSA J. Computing*, 4(4), 1992.
8. D. Whitley, D. Hains, A. Howe, "Tunneling between optima: partition crossover for the TSP," *GECCO*, 2009.
9. B. Ruciński, D. Izzo, F. Biscani, "On the impact of the migration topology on the island model," *Parallel Computing*, 36(10-11), 2010.
10. T. Hoefler et al., "MPI+MPI: a new hybrid approach to parallel programming with MPI plus shared memory," *Computing*, 95(12), 2013.

Note: references 4 to 10 were added from memory. Verify volume and page details against the publishers before submitting.

## 11. Tools

C++17, GCC, OpenMPI (MPI-3), OpenMP, TSPLIB benchmark instances, Python (pandas, Matplotlib), Google Colab.
