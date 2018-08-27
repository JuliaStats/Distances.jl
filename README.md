# Distances.jl

[![Build Status](https://travis-ci.org/JuliaStats/Distances.jl.svg?branch=master)](https://travis-ci.org/JuliaStats/Distances.jl)
[![Coverage Status](https://coveralls.io/repos/JuliaStats/Distances.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaStats/Distances.jl?branch=master)

[![Distances](http://pkg.julialang.org/badges/Distances_0.6.svg)](http://pkg.julialang.org/?pkg=Distances)

A Julia package for evaluating distances(metrics) between vectors.

This package also provides optimized functions to compute column-wise and pairwise distances, which are often substantially faster than a straightforward loop implementation. (See the benchmark section below for details).


## Supported distances

* Euclidean distance
* Squared Euclidean distance
* Periodic Euclidean distance
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
* Haversine distance
* Mean absolute deviation
* Mean squared deviation
* Root mean squared deviation
* Normalized root mean squared deviation
* Bray-Curtis dissimilarity
* Bregman divergence

For ``Euclidean distance``, ``Squared Euclidean distance``, ``Cityblock distance``, ``Minkowski distance``, and ``Hamming distance``, a weighted version is also provided.


## Basic use

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
|  RogersTanimoto      |  `rogerstanimoto(a, b)`    | `2(sum(a&!b) + sum(!a&b)) / (2(sum(a&!b) + sum(!a&b)) + sum(a&b) + sum(!a&!b))` |
|  Jaccard             |  `jaccard(x, y)`           | `1 - sum(min(x, y)) / sum(max(x, y))` |
|  BrayCurtis          |  `braycurtis(x, y)`        | `sum(abs(x - y)) / sum(abs(x + y))`  |
|  CosineDist          |  `cosine_dist(x, y)`       | `1 - dot(x, y) / (norm(x) * norm(y))` |
|  CorrDist            |  `corr_dist(x, y)`         | `cosine_dist(x - mean(x), y - mean(y))` |
|  ChiSqDist           |  `chisq_dist(x, y)`        | `sum((x - y).^2 / (x + y))` |
|  KLDivergence        |  `kl_divergence(p, q)`     | `sum(p .* log(p ./ q))` |
|  GenKLDivergence     |  `gkl_divergence(x, y)`    | `sum(p .* log(p ./ q) - p + q)` |
|  RenyiDivergence     | `renyi_divergence(p, q, k)`| `log(sum( p .* (p ./ q) .^ (k - 1))) / (k - 1)` |
|  JSDivergence        |  `js_divergence(p, q)`     | `KL(p, m) / 2 + KL(p, m) / 2 with m = (p + q) / 2` |
|  SpanNormDist        |  `spannorm_dist(x, y)`     | `max(x - y) - min(x - y)` |
|  BhattacharyyaDist   |  `bhattacharyya(x, y)`     | `-log(sum(sqrt(x .* y) / sqrt(sum(x) * sum(y)))` |
|  HellingerDist       |  `hellinger(x, y) `        | `sqrt(1 - sum(sqrt(x .* y) / sqrt(sum(x) * sum(y))))` |
|  Haversine           |  `haversine(x, y, r)`      | [Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula) |
|  Mahalanobis         |  `mahalanobis(x, y, Q)`    | `sqrt((x - y)' * Q * (x - y))` |
|  SqMahalanobis       |  `sqmahalanobis(x, y, Q)`  | `(x - y)' * Q * (x - y)` |
|  MeanAbsDeviation    |  `meanad(x, y)`            | `mean(abs.(x - y))` |
|  MeanSqDeviation     |  `msd(x, y)`               | `mean(abs2.(x - y))` |
|  RMSDeviation        |  `rmsd(x, y)`              | `sqrt(msd(x, y))` |
|  NormRMSDeviation    |  `nrmsd(x, y)`             | `rmsd(x, y) / (maximum(x) - minimum(x))` |
|  WeightedEuclidean   |  `weuclidean(x, y, w)`     | `sqrt(sum((x - y).^2 .* w))`  |
|  WeightedSqEuclidean |  `wsqeuclidean(x, y, w)`   | `sum((x - y).^2 .* w)`  |
|  WeightedCityblock   |  `wcityblock(x, y, w)`     | `sum(abs(x - y) .* w)`  |
|  WeightedMinkowski   |  `wminkowski(x, y, w, p)`  | `sum(abs(x - y).^p .* w) ^ (1/p)` |
|  WeightedHamming     |  `whamming(x, y, w)`       | `sum((x .!= y) .* w)`  |
|  PeriodicEuclidean   |  `peuclidean(x, y, p)`     | `sqrt(sum(min(mod(abs(x - y), p), p - mod(abs(x - y), p)).^2))`  |
|  Bregman             |  `bregman(F, ∇, x, y; inner = LinearAlgebra.dot)` | `F(x) - F(y) - inner(∇(y), x - y)` |

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

The implementation has been carefully optimized based on benchmarks. The script in `benchmark/benchmarks.jl` defines a benchmark suite
for a variety of distances, under column-wise and pairwise settings.

Here are benchmarks obtained running Julia 1.0 on a computer with a quad-core Intel Core i5-2300K processor @ 2.3 GHz.
The tables below can be replicated using the script in `benchmark/print_table.jl`.

#### Column-wise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance between corresponding columns in two ``200-by-10000`` matrices.

|  distance  |  loop  |  colwise  |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.005488s |  0.001032s |  5.3198 |
| Euclidean | 0.005455s |  0.001033s |  5.2790 |
| Cityblock | 0.005527s |  0.001047s |  5.2812 |
| Chebyshev | 0.009622s |  0.005049s |  1.9058 |
| Minkowski | 0.046888s |  0.040620s |  1.1543 |
| Hamming | 0.005506s |  0.001057s |  5.2077 |
| CosineDist | 0.007194s |  0.002282s |  3.1530 |
| CorrDist | 0.012108s |  0.014342s |  0.8442 |
| ChiSqDist | 0.006355s |  0.001259s |  5.0481 |
| KLDivergence | 0.031378s |  0.024853s |  1.2625 |
| RenyiDivergence | 0.017307s |  0.009944s |  1.7406 |
| RenyiDivergence | 0.051003s |  0.045333s |  1.1251 |
| JSDivergence | 0.046896s |  0.042365s |  1.1070 |
| BhattacharyyaDist | 0.008461s |  0.003804s |  2.2244 |
| HellingerDist | 0.008584s |  0.003678s |  2.3341 |
| WeightedSqEuclidean | 0.005481s |  0.001182s |  4.6370 |
| WeightedEuclidean | 0.005591s |  0.001198s |  4.6680 |
| WeightedCityblock | 0.005712s |  0.001177s |  4.8521 |
| WeightedMinkowski | 0.048870s |  0.041635s |  1.1738 |
| WeightedHamming | 0.005682s |  0.001183s |  4.8042 |
| PeriodicEuclidean | 0.012928s |  0.007914s |  1.6335 |
| SqMahalanobis | 0.094069s |  0.023761s |  3.9590 |
| Mahalanobis | 0.093612s |  0.021565s |  4.3409 |
| BrayCurtis | 0.005686s |  0.001111s |  5.1179 |

We can see that using ``colwise`` instead of a simple loop yields considerable gain (2x - 4x), especially when the internal computation of each distance is simple. Nonetheless, when the computation of a single distance is heavy enough (e.g. *KLDivergence*,  *RenyiDivergence*), the gain is not as significant.

#### Pairwise benchmark

The table below compares the performance (measured in terms of average elapsed time of each iteration) of a straightforward loop implementation and an optimized implementation provided in *Distances.jl*. The task in each iteration is to compute a specific distance in a pairwise manner between columns in a ``100-by-200`` and ``100-by-250`` matrices, which will result in a ``200-by-250`` distance matrix.

|  distance  |  loop  |  pairwise  |  gain  |
|----------- | -------| ----------| -------|
| SqEuclidean | 0.014776s |  0.000186s | **79.5505** |
| Euclidean | 0.014541s |  0.000273s | **53.2506** |
| Cityblock | 0.014449s |  0.000920s | 15.7060 |
| Chebyshev | 0.026325s |  0.011859s |  2.2198 |
| Minkowski | 0.120684s |  0.103045s |  1.1712 |
| Hamming | 0.014306s |  0.000797s | 17.9394 |
| CosineDist | 0.019469s |  0.000230s | 84.7057 |
| CorrDist | 0.032510s |  0.000273s | **119.2969** |
| ChiSqDist | 0.016795s |  0.003127s |  5.3714 |
| KLDivergence | 0.080376s |  0.061518s |  1.3066 |
| RenyiDivergence | 0.045634s |  0.023118s |  1.9740 |
| RenyiDivergence | 0.133878s |  0.113815s |  1.1763 |
| JSDivergence | 0.119839s |  0.105798s |  1.1327 |
| BhattacharyyaDist | 0.023843s |  0.009411s |  2.5334 |
| HellingerDist | 0.023501s |  0.008779s |  2.6768 |
| WeightedSqEuclidean | 0.014698s |  0.000200s | **73.6076** |
| WeightedEuclidean | 0.015542s |  0.000289s | **53.8548** |
| WeightedCityblock | 0.015417s |  0.001589s |  9.7047 |
| WeightedMinkowski | 0.126733s |  0.104740s |  1.2100 |
| WeightedHamming | 0.015610s |  0.001521s | 10.2655 |
| PeriodicEuclidean | 0.034584s |  0.019539s |  1.7700 |
| SqMahalanobis | 0.447340s |  0.000375s | **1192.7965** |
| Mahalanobis | 0.436488s |  0.000464s | **940.6720** |
| BrayCurtis | 0.015394s |  0.001522s | 10.1145 |

For distances of which a major part of the computation is a quadratic form (e.g. *Euclidean*, *CosineDist*, *Mahalanobis*), the performance can be drastically improved by restructuring the computation and delegating the core part to ``GEMM`` in *BLAS*. The use of this strategy can easily lead to 100x performance gain over simple loops (see the highlighted part of the table above).
