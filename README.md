# Distances.jl

[![Build Status](https://travis-ci.org/JuliaStats/Distances.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/Distances.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaStats/Distances.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaStats/Distances.jl?branch=master)

[![Distances](http://pkg.julialang.org/badges/Distances_0.5.svg)](http://pkg.julialang.org/?pkg=Distances)

A Julia package for evaluating distances(metrics) between vectors.

This package also provides optimized functions to compute column-wise and pairwise distances, which are often substantially faster than a straightforward loop implementation. (See the benchmark section below for details).


## Supported distances

* Euclidean distance
* Squared Euclidean distance
* Cityblock distance
* Jaccard distance
* Rogers-Tanimoto distance
* Chebyshev distance
* Minkowski distance
* Hamming distance
* Cosine distance
* Correlation distance
* Chi-square distance
* Kullback-Leibler divergence
* Generalized Kullback-Leibler divergence
* Rényi divergence
* Jensen-Shannon divergence
* Mahalanobis distance
* Squared Mahalanobis distance
* Bhattacharyya distance
* Hellinger distance
* Mean absolute deviation
* Mean squared deviation
* Root mean squared deviation
* Normalized root mean squared deviation

For ``Euclidean distance``, ``Squared Euclidean distance``, ``Cityblock distance``, ``Minkowski distance``, and ``Hamming distance``, a weighted version is also provided.


## Basic Use

The library supports three ways of computation: *computing the distance between two vectors*, *column-wise computation*, and *pairwise computation*.

#### Computing the distance between two vectors

Each distance corresponds to a *distance type*. You can always compute a certain distance between two vectors using the following syntax

```julia
r = evaluate(dist, x, y)
```

Here, dist is an instance of a distance type. For example, the type for Euclidean distance is ``Euclidean`` (more distance types will be introduced in the next section), then you can compute the Euclidean distance between ``x`` and ``y`` as

```julia
r = evaluate(Euclidean(), x, y)
```

Common distances also come with convenient functions for distance evaluation. For example, you may also compute Euclidean distance between two vectors as below

```julia
r = euclidean(x, y)
```

#### Computing distances between corresponding columns

Suppose you have two ``m-by-n`` matrix ``X`` and ``Y``, then you can compute all distances between corresponding columns of X and Y in one batch, using the ``colwise`` function, as

```julia
r = colwise(dist, X, Y)
```

The output ``r`` is a vector of length ``n``. In particular, ``r[i]`` is the distance between ``X[:,i]`` and ``Y[:,i]``. The batch computation typically runs considerably faster than calling ``evaluate`` column-by-column.

Note that either of ``X`` and ``Y`` can be just a single vector -- then the ``colwise`` function will compute the distance between this vector and each column of the other parameter.

#### Computing pairwise distances

Let ``X`` and ``Y`` respectively have ``m`` and ``n`` columns. Then the ``pairwise`` function computes distances between each pair of columns in ``X`` and ``Y``:

```julia
R = pairwise(dist, X, Y)
```

In the output, ``R`` is a matrix of size ``(m, n)``, such that ``R[i,j]`` is the distance between ``X[:,i]`` and ``Y[:,j]``. Computing distances for all pairs using ``pairwise`` function is often remarkably faster than evaluting for each pair individually.

If you just want to just compute distances between columns of a matrix ``X``, you can write

```julia
R = pairwise(dist, X)
```

This statement will result in an ``m-by-m`` matrix, where ``R[i,j]`` is the distance between ``X[:,i]`` and ``X[:,j]``.
``pairwise(dist, X)`` is typically more efficient than ``pairwise(dist, X, X)``, as the former will take advantage of the symmetry when ``dist`` is a semi-metric (including metric).

#### Computing column-wise and pairwise distances inplace

If the vector/matrix to store the results are pre-allocated, you may use the storage (without creating a new array) using the following syntax:

```julia
colwise!(r, dist, X, Y)
pairwise!(R, dist, X, Y)
pairwise!(R, dist, X)
```

Please pay attention to the difference, the functions for inplace computation are ``colwise!`` and ``pairwise!`` (instead of ``colwise`` and ``pairwise``).


## Distance type hierarchy

The distances are organized into a type hierarchy.

At the top of this hierarchy is an abstract class **PreMetric**, which is defined to be a function ``d`` that satisfies

	d(x, x) == 0  for all x
	d(x, y) >= 0  for all x, y

**SemiMetric** is a abstract type that refines **PreMetric**. Formally, a *semi-metric* is a *pre-metric* that is also symmetric, as

	d(x, y) == d(y, x)  for all x, y

**Metric** is a abstract type that further refines **SemiMetric**. Formally, a *metric* is a *semi-metric* that also satisfies triangle inequality, as

	d(x, z) <= d(x, y) + d(y, z)  for all x, y, z

This type system has practical significance. For example, when computing pairwise distances between a set of vectors, you may only perform computation for half of the pairs, and derive the values immediately for the remaining halve by leveraging the symmetry of *semi-metrics*.

Each distance corresponds to a distance type. The type name and the corresponding mathematical definitions of the distances are listed in the following table.

| type name            |  convenient syntax         | math definition     |
| -------------------- | -------------------------- | --------------------|
|  Euclidean           |  `euclidean(x, y)`         | `sqrt(sum((x - y) .^ 2))` |
|  SqEuclidean         |  `sqeuclidean(x, y)`       | `sum((x - y).^2)` |
|  Cityblock           |  `cityblock(x, y)`         | `sum(abs(x - y))` |
|  Chebyshev           |  `chebyshev(x, y)`         | `max(abs(x - y))` |
|  Minkowski           |  `minkowski(x, y, p)`      | `sum(abs(x - y).^p) ^ (1/p)` |
|  Hamming             |  `hamming(k, l)`           | `sum(k .!= l)` |
|  Rogers-Tanimoto     |  `rogerstanimoto(a, b)`    | `2(sum(a&!b) + sum(!a&b)) / (2(sum(a&!b) + sum(!a&b)) + sum(a&b) + sum(!a&!b))` |
|  Jaccard             |  `jaccard(x, y)`           | `1 - sum(min(x, y)) / sum(max(x, y))` |
|  CosineDist          |  `cosine_dist(x, y)`       | `1 - dot(x, y) / (norm(x) * norm(y))` |
|  CorrDist            |  `corr_dist(x, y)`         | `cosine_dist(x - mean(x), y - mean(y))` |
|  ChiSqDist           |  `chisq_dist(x, y)`        | `sum((x - y).^2 / (x + y))` |
|  KLDivergence        |  `kl_divergence(p, q)`     | `sum(p .* log(p ./ q))` |
|  GenKLDivergence     |  `gkl_divergence(x, y)`    | `sum(p .* log(p ./ q) - p + q)` |
|  RenyiDivergence     | `renyi_divergence(p, q, k)`| `log(sum( p .* (p ./ q) .^ (k - 1))) / (k - 1)` |
|  JSDivergence        |  `js_divergence(p, q)`     | `KL(p, m) / 2 + KL(p, m) / 2 with m = (p + q) / 2` |
|  SpanNormDist        |  `spannorm_dist(x, y)`     | `max(x - y) - min(x - y )` |
|  BhattacharyyaDist   |  `bhattacharyya(x, y)`     | `-log(sum(sqrt(x .* y) / sqrt(sum(x) * sum(y)))` |
|  HellingerDist       |  `hellinger(x, y) `        | `sqrt(1 - sum(sqrt(x .* y) / sqrt(sum(x) * sum(y))))` |
|  Mahalanobis         |  `mahalanobis(x, y, Q)`    | `sqrt((x - y)' * Q * (x - y))` |
|  SqMahalanobis       |  `sqmahalanobis(x, y, Q)`  | ` (x - y)' * Q * (x - y)`  |
|  MeanAbsDeviation    |  `meanad(x, y)`            | `mean(abs.(x - y))` |
|  MeanSqDeviation     |  `msd(x, y)`               | `mean(abs2.(x - y))` |
|  RMSDeviation        |  `rmsd(x, y)`              | `sqrt(msd(x, y))` |
|  NormRMSDeviation    |  `nrmsd(x, y)`             | `rmsd(x, y) / (maximum(x) - minimum(x))` |
|  WeightedEuclidean   |  `weuclidean(x, y, w)`     | `sqrt(sum((x - y).^2 .* w))`  |
|  WeightedSqEuclidean |  `wsqeuclidean(x, y, w)`   | `sum((x - y).^2 .* w)`  |
|  WeightedCityblock   |  `wcityblock(x, y, w)`     | `sum(abs(x - y) .* w)`  |
|  WeightedMinkowski   |  `wminkowski(x, y, w, p)`  | `sum(abs(x - y).^p .* w) ^ (1/p)` |
|  WeightedHamming     |  `whamming(x, y, w)`       | `sum((x .!= y) .* w)`  |

**Note:** The formulas above are using *Julia*'s functions. These formulas are mainly for conveying the math concepts in a concise way. The actual implementation may use a faster way. The arguments `x` and `y` are arrays of real numbers; `k` and `l` are arrays of distinct elements of any kind; a and b are arrays of Bools; and finally, `p` and `q` are arrays forming a discrete probability distribution and are therefore both expected to sum to one.

### Precision for Euclidean and SqEuclidean

For efficiency (see the benchmarks below), `Euclidean` and
`SqEuclidean` make use of BLAS3 matrix-matrix multiplication to
calculate distances.  This corresponds to the following expansion:

```julia
(x-y)^2 == x^2 - 2xy + y^2
```

However, equality is not precise in the presence of roundoff error,
and particularly when `x` and `y` are nearby points this may not be
accurate.  Consequently, `Euclidean` and `SqEuclidean` allow you to
supply a relative tolerance to force recalculation:

```julia
julia> x = reshape([0.1, 0.3, -0.1], 3, 1);

julia> pairwise(Euclidean(), x, x)
1×1 Array{Float64,2}:
 7.45058e-9

julia> pairwise(Euclidean(1e-12), x, x)
1×1 Array{Float64,2}:
 0.0
```


## Benchmarks

The implementation has been carefully optimized based on benchmarks. The script in `benchmarks/benchmark.jl` defines a benchmark suite 
for a variety of distances, under column-wise and pairwise settings. 
 
Here are benchmarks obtained running Julia 0.6 on a computer with a quad-core Intel Core i5-2500K processor @ 3.3 GHz.
The tables below can be replicated using the script in `benchmarks/print_table.jl`.

#### Column-wise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance between corresponding columns in two ``200-by-10000`` matrices.

|  distance  |  loop  |  colwise  |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.007467s |  0.002171s |  3.4393 |
| Euclidean | 0.007421s |  0.002185s |  3.3961 |
| Cityblock | 0.007442s |  0.002168s |  3.4328 |
| Chebyshev | 0.011494s |  0.005846s |  1.9662 |
| Minkowski | 0.174122s |  0.143938s |  1.2097 |
| Hamming | 0.007586s |  0.002249s |  3.3739 |
| CosineDist | 0.008581s |  0.002853s |  3.0076 |
| CorrDist | 0.014991s |  0.011402s |  1.3148 |
| ChiSqDist | 0.012990s |  0.006910s |  1.8800 |
| KLDivergence | 0.051694s |  0.047433s |  1.0898 |
| RenyiDivergence | 0.021406s |  0.017845s |  1.1996 |
| RenyiDivergence | 0.031397s |  0.027801s |  1.1294 |
| JSDivergence | 0.115657s |  0.495861s |  0.2332 |
| BhattacharyyaDist | 0.019273s |  0.013195s |  1.4606 |
| HellingerDist | 0.018883s |  0.012921s |  1.4613 |
| WeightedSqEuclidean | 0.007559s |  0.002256s |  3.3504 |
| WeightedEuclidean | 0.007624s |  0.002325s |  3.2796 |
| WeightedCityblock | 0.007803s |  0.002248s |  3.4709 |
| WeightedMinkowski | 0.154231s |  0.147579s |  1.0451 |
| WeightedHamming | 0.009042s |  0.003182s |  2.8417 |
| SqMahalanobis | 0.070869s |  0.019199s |  3.6913 |
| Mahalanobis | 0.070980s |  0.019305s |  3.6768 |

We can see that using ``colwise`` instead of a simple loop yields considerable gain (2x - 4x), especially when the internal computation of each distance is simple. Nonetheless, when the computation of a single distance is heavy enough (e.g. *KLDivergence*,  *RenyiDivergence*), the gain is not as significant.

#### Pairwise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance in a pairwise manner between columns in a ``100-by-200`` and ``100-by-250`` matrices, which will result in a ``200-by-250`` distance matrix.

|  distance  |  loop  |  pairwise  |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.019217s |  0.000196s | **97.9576** |
| Euclidean | 0.019287s |  0.000505s | **38.1874** |
| Cityblock | 0.019376s |  0.002532s |  7.6521 |
| Chebyshev | 0.032814s |  0.014811s |  2.2155 |
| Minkowski | 0.382199s |  0.361059s |  1.0586 |
| Hamming | 0.019826s |  0.003047s |  6.5072 |
| CosineDist | 0.024012s |  0.000367s | **65.3661** |
| CorrDist | 0.041356s |  0.000421s | **98.3049** |
| ChiSqDist | 0.035105s |  0.017882s |  1.9632 |
| KLDivergence | 0.131773s |  0.117640s |  1.1201 |
| RenyiDivergence | 0.057569s |  0.042694s |  1.3484 |
| RenyiDivergence | 0.082862s |  0.067811s |  1.2220 |
| JSDivergence | 0.292014s |  0.276898s |  1.0546 |
| BhattacharyyaDist | 0.051302s |  0.033043s |  1.5526 |
| HellingerDist | 0.049518s |  0.031856s |  1.5545 |
| WeightedSqEuclidean | 0.019959s |  0.000218s | **91.7298** |
| WeightedEuclidean | 0.020336s |  0.000557s | **36.5405** |
| WeightedCityblock | 0.020391s |  0.003118s |  6.5404 |
| WeightedMinkowski | 0.387738s |  0.366898s |  1.0568 |
| WeightedHamming | 0.024456s |  0.007403s |  3.3033 |
| SqMahalanobis | 0.113107s |  0.000366s | **309.3621** |
| Mahalanobis | 0.114646s |  0.000686s | **167.0595** |

For distances of which a major part of the computation is a quadratic form (e.g. *Euclidean*, *CosineDist*, *Mahalanobis*), the performance can be drastically improved by restructuring the computation and delegating the core part to ``GEMM`` in *BLAS*. The use of this strategy can easily lead to 100x performance gain over simple loops (see the highlighted part of the table above).
